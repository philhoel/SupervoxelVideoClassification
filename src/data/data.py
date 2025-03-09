import os
import shutil
import torchvision
import cv2
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import multiprocessing
from torch.utils.data import Subset
import math
import matplotlib.pyplot as plt
from matplotlib import rc


class WindowTransform:
    """
    Selects a sequence of video frames by selecting every n-th frame from the middle of the video.
    If the video is too short, the frames are repeated if necessary and chosen with an even distribution across the whole video

    Attributes:
        n : Determines which frames to select, every n-th
        total : Total number of frames to select
    """

    def __init__(self, n, total=16):
        self.n = n
        self.total = total

    def __call__(self, x):

        # Take frames from the middle of the video
        frames = x.shape[1]
        start = (frames // 2) - ((self.total * self.n) // 2)
        end = (frames // 2) + ((self.total * self.n) // 2)

        # Interval fits, take every n-th element
        if start >= 0 and end <= frames:
            result = x[:, start:end:self.n, :, :]
            assert result.shape[1] == self.total
            return result
        else:
            # Not enough elements, copy frames
            if frames <= self.total:
                repeats = math.ceil(self.total / frames)
                x = torch.repeat_interleave(x, repeats=repeats, dim=1)
            # Get a distribution of frames across the video
            step = frames // self.total
            result = x[:, ::step, :, :][:, :self.total, :, :]
            assert result.shape[1] == self.total
            return result


def gaussian_noise(mean=0.0, std=0.1):
    return lambda x: torch.clamp(x + torch.normal(mean, std, x.shape), 0.0, 1.0)


def meta_information_dataset(dataset, start_idx=None, end_idx=None):
    """
    Prints the average, minimum, and maximum values for the number of frames, 
    height, and width of videos in a given dataset or its subset (determined my start_idx and end_idx)

    Args:
        dataset : Dataset object for which meta data is calculated
        start_idx : Starting index for subset of the dataset. Defaults to None, meaning the entire dataset is used
        end_idx : Ending index for subset of the dataset. Defaults to None, meaning the entire dataset is used
    """

    # Use subset of data (for parallel computation)
    if start_idx is not None and end_idx is not None:
        paths = dataset.paths[start_idx:end_idx]
    # Use whole dataset
    else:
        paths = dataset.paths

    video_data = []

    for path in paths:
        cap = cv2.VideoCapture(path)

        # Get meta data of video
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_data.append({'frames': frames, 'height': height, 'width': width})

        cap.release()

    # Convert to a DataFrame
    df = pd.DataFrame(video_data)

    # Calculate statistics
    stats = {data: {
        'avg': df[data].mean().item(),
        'min': df[data].min().item(),
        'max': df[data].max().item()
    } for data in df.columns}

    # Print results
    print(stats)
    print(len(paths))


def _clean_csv(root_dir, subdir, split, orig_csv, clean_csv, skiprows=0):
    """
    Cleans a dataset CSV by verifying that video files are not corrupted and can be used, 
    then saves a new CSV containing only valid entries.

    Args:
        root_dir : Root directory containing the dataset and annotations folder
        subdir : Subdirectory where video files are stored
        split : Used to construct paths to the videos
        orig_csv : Filename of original CSV containing video metadata
        clean_csv : Filename for new cleaned CSV that will contain only data of valid videos
        skiprows : Number of rows to skip when reading the original CSV (original dataset: use 1)
    """

    # Load the CSV file into a DataFrame
    csv_file_path_orig = os.path.join(root_dir, 'annotations', orig_csv)
    orig_df = pd.read_csv(csv_file_path_orig, header=None, skiprows=skiprows)
    orig_df.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

    # Output path for cleaned CSV
    csv_file_path_clean = os.path.join(root_dir, 'annotations', clean_csv)
    corrupted_videos = 0
    clean_df = pd.DataFrame(columns=orig_df.columns)

    if os.path.exists(csv_file_path_clean):
        print("Cleaned CSV file already exists, skipping creation.")
        return

    # Create paths including the correct filename
    orig_df['video_filename'] = (
        orig_df['video_id'] + '_'
        + orig_df['start'].apply(lambda x: f"{x:06}") + '_'
        + orig_df['end'].apply(lambda x: f"{x:06}") + '.mp4'
    )

    path_to_subdir = root_dir + '/' + subdir + '/' if subdir else root_dir + \
        '/'  # subdir None if original Kinetics classes dataset
    orig_df['paths'] = (
        path_to_subdir
        + orig_df['split'] + '/'
        + orig_df['label'] + '/'
        + orig_df['video_filename'])

    for _, row in orig_df.iterrows():
        path = row['paths']

        # Load video using OpenCV
        cap = cv2.VideoCapture(path)

        if cap.isOpened():
            clean_df = pd.concat(
                [clean_df, row.to_frame().T], ignore_index=True)
        else:
            corrupted_videos += 1

        cap.release()

    # Save data for valid videos
    clean_df = clean_df.drop('video_filename', axis='columns')
    clean_df = clean_df.drop('paths', axis='columns')
    clean_df.to_csv(csv_file_path_clean, header=False, index=False)

    print(f"Removed {corrupted_videos} corrupted videos.")
    print(f"Saved cleaned CSV to {csv_file_path_clean}")
    return


def _label_mapping(root_dir, overwrite=False, resized_dataset=False):
    """
    Creates and saves a label-to-number mapping for a dataset based on its CSV file. 
    Works for both the entire dataset and a customized subset.

    Args: 
        root_dir : Root directory containing the dataset and annotations folder
        overwrite : If True, overwrites the existing label mapping file
        resized_dataset : If True, uses CSV file specific to the customized dataset (e.g., `label_mapping_50.csv` and `train_clean_50.csv`)
    """

    # Label-to-number mapping
    if resized_dataset:
        label_mapping_file = os.path.join(
            root_dir, 'annotations', 'label_mapping_50.csv')
        csv_file_path = os.path.join(
            root_dir, 'annotations', 'train_clean_50.csv')
    else:
        label_mapping_file = os.path.join(
            root_dir, 'annotations', 'label_mapping.csv')
        csv_file_path = os.path.join(root_dir, 'annotations', 'val.csv')

    if not os.path.exists(label_mapping_file) or overwrite:

        # Load CSV file into a DataFrame to get all existing labels in dataset
        data = pd.read_csv(csv_file_path, header=None, skiprows=1)
        data.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

        unique_labels = sorted(data['label'].unique())

        # Save label mapping to CSV file
        pd.DataFrame(unique_labels).to_csv(
            label_mapping_file, index=False, header=False)
        print(f"Label mapping saved to {label_mapping_file}")
    else:
        print("Label mapping already exists, skipped mapping process.")


def augment(augs, resize=(160, 160)):
    """
    Composes augmentations, each applied with a random probability

    Args:
        augs : List of augmentation types (strings) 
            Possible augmentations:
            - "flip_h": Horizontal flip
            - "flip_v": Vertical flip
            - "noise": Gaussian noise with a random standard deviation
            - "crop": Random crop followed by resize
            - "brightness": Random color jitter for brightness
        resize : Target size to which images will be resized after cropping
    """

    # Possible augmentations to apply
    valid_augmentations = {
        "flip_h": v2.RandomHorizontalFlip,
        "flip_v": v2.RandomVerticalFlip,
        "noise": gaussian_noise(std=0.05 + (0.15 - 0.05) * torch.rand(1).item()),
        "crop": v2.Compose([
            v2.RandomCrop(size=(140, 140)),
            v2.Resize(size=resize)
        ]),
        "brightness": v2.ColorJitter(brightness=(0.7, 1.3))
    }

    augmentations = []

    # Rondom probabilities of augmentation
    p_values = torch.rand(len(augs))

    # Sample 0s and 1s based on each random probability p
    samples = torch.rand(len(augs)) < p_values

    # Augmentation based on sample with probability p_values
    for idx, aug in enumerate(augs):
        if aug == "flip_h" or aug == "flip_v":
            augmentations.append(valid_augmentations[aug](p_values[idx]))
        elif samples[idx]:
            augmentations.append(valid_augmentations[aug])
        else:
            continue

    return T.Compose(augmentations)


def store_new_dataset_pt(dataset, sub_dir, start_idx, end_idx):
    """
    Stores a subset of a given dataset (determined by start_idx and end_idx) as tensors in a specified 
    subdirectory of the datasets root directory

    Args:
        dataset
        sub_dir : Subdirectory within the dataset's root directory where the processed tensors will be stored
        start_idx : Starting index for subset of the dataset
        end_idx :  Ending index for subset of the dataset
    """
    batch_size = 16

    paths = dataset.paths
    path_to_new_split = os.path.join(dataset.root_dir, sub_dir, dataset.split)
    subset = Subset(dataset, range(start_idx, end_idx))
    # Create DataLoader with multiple workers
    dataloader = DataLoader(subset, batch_size=batch_size,
                            num_workers=19, prefetch_factor=4)

    # Iterate through dataset in batches
    for batch_idx, (x_batch, _) in enumerate(dataloader):
        for idx, x in enumerate(x_batch):
            overall_idx = start_idx + batch_idx * batch_size + idx
            file_name = os.path.splitext(
                os.path.basename(paths[overall_idx]))[0]
            # Last folder in path is semantic class label
            label = os.path.basename(os.path.dirname(paths[overall_idx]))

            new_path_to_label_dir = os.path.join(path_to_new_split, label)
            os.makedirs(new_path_to_label_dir, exist_ok=True)

            # Save the processed tensor
            torch.save(x, os.path.join(
                new_path_to_label_dir, file_name + '.pt'))


def store_new_dataset_mp4(dataset, sub_dir, start_idx, end_idx):
    """
    Stores a subset of a given dataset (determined by start_idx and end_idx) as mp4 files in a specified 
    subdirectory of the datasets root directory

    Args:
        dataset
        sub_dir : Subdirectory within the dataset's root directory where the processed videos will be stored
        start_idx : Starting index for subset of the dataset
        end_idx :  Ending index for subset of the dataset
    """
    batch_size = 16

    paths = dataset.paths
    path_to_new_split = os.path.join(dataset.root_dir, sub_dir, dataset.split)
    subset = Subset(dataset, range(start_idx, end_idx))
    # Create DataLoader with multiple workers
    dataloader = DataLoader(subset, batch_size=batch_size,
                            num_workers=19, prefetch_factor=4)
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Iterate through dataset in batches
    for batch_idx, (x_batch, _) in enumerate(dataloader):
        for idx, x in enumerate(x_batch):
            overall_idx = start_idx + batch_idx * batch_size + idx
            file_name = os.path.splitext(
                os.path.basename(paths[overall_idx]))[0]
            # Last folder in path is semantic class label
            label = os.path.basename(os.path.dirname(paths[overall_idx]))

            new_path_to_label_dir = os.path.join(path_to_new_split, label)
            os.makedirs(new_path_to_label_dir, exist_ok=True)

            # Save the processed tensor
            # (num_frames, height, width, channels)
            x = (x * 255).permute(1, 2, 3, 0).byte()
            out = cv2.VideoWriter(os.path.join(
                new_path_to_label_dir, file_name + '.mp4'), fourcc, 26, (x.shape[2], x.shape[1]))

            for frame in x:
                frame = frame.numpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)

            out.release()


def video_lengths(root_dir, split):

    path_to_csv = os.path.join(
        root_dir, 'annotations', f'{split}_clean_small.csv')
    # Load cleaned CSV file
    df = pd.read_csv(path_to_csv, header=None, skiprows=1)
    df.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

    # Calculate duration for each video
    df['duration'] = df['end'] - df['start']

    # Calculate average duration
    average_duration = df['duration'].mean()
    print(
        f"The average length of the video clips is {average_duration} seconds.")

    # Find number of rows where the duration is not 10 seconds
    count_different_than_10 = df[df['duration'] != 10].shape[0]
    print(
        f"The number of rows with a duration different than 10 seconds is {count_different_than_10}.")

    # Get the minimum and maximum video lengths
    min_length = df['duration'].min()
    max_length = df['duration'].max()
    print(f"The minimum length of the video clips is {min_length} seconds.")
    print(f"The maximum length of the video clips is {max_length} seconds.")


def remove_classes_mapping_csv(root_dir, orig_label_mapping, small_label_mapping):
    """
    Keeps a subset of labels from the original label mapping CSV, 
    saving the selected labels into a new CSV file for a smaller set of classes

    Args:
        root_dir
        orig_label_mapping : Filename of original label mapping CSV containing all classes
        small_label_mapping : Filename for new CSV to store the filtered list of classes
    """
    # Load CSV for 100 classes
    label_mapping_file = os.path.join(
        root_dir, 'annotations', orig_label_mapping)  # 'label_mapping_small.csv'

    labels_100 = pd.read_csv(label_mapping_file, header=None, skiprows=0)
    labels_100.columns = ['label']

    # Remove smallest labels
    keep_labels = ["abseiling", "archery", "arm wrestling", "baking cookies", "barbequing", "bartending", "belly dancing", "bench pressing", "catching or throwing frisbee", "catching or throwing softball", "changing oil", "cheerleading", "chopping wood", "clean and jerk", "cleaning shoes", "climbing ladder", "contact juggling", "diving cliff", "doing nails", "dribbling basketball", "driving tractor", "dunking basketball", "eating burger", "folding paper", "getting a haircut",
                   "getting a tattoo", "gymnastics tumbling", "hammer throw", "headbanging", "high kick", "hitting baseball", "hurdling", "making snowman", "marching", "massaging back", "milking cow", "motorcycling", "mowing lawn", "playing clarinet", "playing didgeridoo", "playing drums", "playing harmonica", "playing harp", "playing ice hockey", "playing monopoly", "playing paintball", "playing poker", "playing recorder", "playing squash or racquetball", "playing tennis"]
    # Only keep entries where label is not in the list remove_labels
    labels_50 = labels_100[labels_100['label'].isin(keep_labels)]

    # Save remaining labels
    label_mapping_file_small = os.path.join(
        root_dir, 'annotations', small_label_mapping)  # 'label_mapping_50.csv'
    labels_small.to_csv(label_mapping_file_small, header=False, index=False)

    print(f"New mapping CSV saved to {label_mapping_file_small}")
    print(f'Kept {len(remove_labels)} classes')


def remove_classes_csv(root_dir, orig_csv, small_csv, skiprows=0, label_mapping_csv='label_mapping_50.csv'):
    """
    Keeps video metadata from the original CSV where the labels are present in the specified label mapping CSV, 
    saves the filtered video information into a new CSV file

    Args:
        root_dir
        orig_csv : Filename of original CSV file that contains the video metadata and their labels
        small_csv : Filename for new CSV to store the filtered dataset meta information
        skiprows : Number of rows to skip when reading the original CSV,
            only f'{split}.csv's have a header, use skiprows=1
        label_mapping_csv : Filename of label mapping CSV that contains the list of labels to keep. 
            Only entries with labels found in this file will be included in the new CSV
    """
    csv_file_path_orig = os.path.join(root_dir, 'annotations', orig_csv)
    label_mapping_file = os.path.join(
        root_dir, 'annotations', label_mapping_csv)  # Keep only those classes

    labels = pd.read_csv(label_mapping_file, header=None, skiprows=0)
    labels.columns = ['label']

    df_orig = pd.read_csv(csv_file_path_orig, header=None, skiprows=skiprows)
    df_orig.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

    # Only keep entries where label is in the list of labels
    df_small = df_orig[df_orig['label'].isin(labels['label'])]

    # Save CSV of subset with labels from label_mapping_file
    csv_file_path_small = os.path.join(root_dir, 'annotations', small_csv)
    df_small.to_csv(csv_file_path_small, header=False, index=False)

    print(f"New CSV saved to {csv_file_path_small}")


def save_single_augmented_video(path, x, aug):
    """
    Visualizes a single augmentation applied to a single video and saves the result as a video file

    Args:
        path : Directory where the augmented video will be saved
        x : Input video tensor, expected to have shape (C, T, H, W)
        aug : Type of augmentation to apply. Must be one of the following:
            - "flip_h" for horizontal flip
            - "flip_v" for vertical flip
            - "noise" for adding Gaussian noise
            - "crop" for random cropping and resizing
            - "brightness" for random brightness adjustment
    """
    x = x.permute(1, 0, 2, 3)

    # Possible augmentations to apply
    valid_augmentations = {
        "flip_h": v2.RandomHorizontalFlip,
        "flip_v": v2.RandomVerticalFlip,
        "noise": gaussian_noise(std=0.05 + (0.15 - 0.05) * torch.rand(1).item()),
        "crop": v2.Compose([
            v2.RandomCrop(size=(140, 140)),
            v2.Resize(size=(160, 160))
        ]),
        "brightness": v2.ColorJitter(brightness=(0.7, 1.3))
    }
    assert aug in valid_augmentations, f"Invalid augmentation {aug}, must be in {list(valid_augmentations.keys())}"

    # Apply augmentation
    if aug == "flip_h" or aug == "flip_v":
        x_aug = valid_augmentations[aug](1)(x)
    else:
        x_aug = valid_augmentations[aug](x)

    x_aug = x_aug.permute(1, 0, 2, 3)

    # Save the processed tensor
    os.makedirs(path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    x_aug = (x_aug * 255).permute(1, 2, 3, 0).byte()
    out = cv2.VideoWriter(os.path.join(
        path, f'example_{aug}' + '.mp4'), fourcc, 26, (x_aug.shape[2], x_aug.shape[1]))

    for frame in x_aug:
        frame = frame.numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"example saved: {os.path.join(path, f"example_{aug}" + '.mp4')}")


def data_overview(root_dir, csv):
    """
    Visualizes the distribution of classes in a dataset and identifies the smallest and largest class 
    based on video count

    Args:
        root_dir
        csv : Name of CSV file containing the dataset meta information
    """

    # Load csv
    path_to_csv = os.path.join(root_dir, 'annotations', csv)
    df = pd.read_csv(path_to_csv, header=None, skiprows=0)
    df.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

    # Group by 'label', find smallest and largest class
    class_counts = df['label'].value_counts()

    smallest_class = class_counts.idxmin()
    smallest_class_count = class_counts.min()

    largest_class = class_counts.idxmax()
    largest_class_count = class_counts.max()

    # Print results
    print(class_counts)
    print(
        f"Smallest class: '{smallest_class}' with {smallest_class_count} videos")
    print(
        f"Largest class: '{largest_class}' with {largest_class_count} videos")

    # Plot class distribution with correct font for latex
    rc('font', family='serif')
    plt.figure(figsize=(15, 8))
    class_counts.plot(kind='bar')

    plt.xticks(rotation=45, ha='right')

    plt.title(r'Number of Videos per Class', fontsize=16)
    plt.xlabel(r'Class', fontsize=14)
    plt.ylabel(r'Number of Videos', fontsize=14)

    plt.tight_layout()
    plt.savefig("./visualization/class_dist.pdf", format='pdf')
