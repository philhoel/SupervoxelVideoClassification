import os
from time import perf_counter
from utils import format_time
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import pandas as pd
from data.data import _label_mapping, _clean_csv, augment, remove_classes_csv

# Personalized dataset class for Kintetics dataset


class KineticsDataset(Dataset):
    """
    Dataset class for the original Kinetics dataset

    Args:
        root_dir : Directory where the Kinetics dataset is stored
        split : Dataset split to load ('train', 'val', or 'test')
        transforms : Transforms to be applied to each video
        augmentations : List of augmentations to be applied on the video, 
            e.g. ['flip_h', 'flip_v', 'noise', 'crop', 'brightness']
        slurm_id : Identifier for slurm jobs to manage data storage in the scratch directory
        num_classes : Number of classes to use for classification (50 or 400)
    """

    def __init__(self, root_dir, split, transforms=None, augmentations=None, slurm_id=None, num_classes=50):

        scratch_dir = f"/scratch/{slurm_id}/data"
        if slurm_id is not None and not os.path.exists(scratch_dir):
            # Copy data over from root to scratch
            start_time = perf_counter()
            os.makedirs(scratch_dir, exist_ok=True)
            print(f"Copying data from {root_dir} to {scratch_dir}")
            os.system(f"cp -r {root_dir}/* {scratch_dir}/")
            print(
                f"Data copy took: {format_time(perf_counter() - start_time)}")
            root_dir = scratch_dir

        self.root_dir = root_dir  # Root directory containing dataset
        self.split = split  # 'train', 'val', 'test'
        self.split_dir = os.path.join(self.root_dir, split)
        self.transforms = transforms
        self.seed = 42

        self.augmentations = augmentations
        valid_augmentations = ["flip_h", "flip_v",
                               "noise", "crop", "brightness"]
        if self.augmentations:
            assert all(
                aug in valid_augmentations for aug in self.augmentations), "Invalid augmentations, check spelling"

        self.num_classes = num_classes
        valid_num_classes = [50, 400]
        assert self.num_classes in valid_num_classes, f"Invalid amount of classes, given {self.num_classes}, must be in {valid_num_classes}."

        self.paths, self.labels = self.__paths_and_labels_from_csv()

    def update_root_dir(self, root_dir):
        self.root_dir = root_dir
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.paths, self.labels = self.__paths_and_labels_from_csv()

    def __paths_and_labels_from_csv(self):
        """
        Uses CSV files to generate paths to video files and get corresponding numerical labels

        The method ensures that the correct CSV file for the specified number of classes (50 or 400) is used. It also cleans the data if 
        the cleaned CSV file is not found. Additionally, if the label mapping file does not exist, it will create it


        Returns:
            paths : List of full paths to the video files
            labels : List of numerical labels corresponding to each video
        """

        csv_name = self.split
        label_mapping_name = 'label_mapping_50' if self.num_classes == 50 else 'label_mapping'

        # Load correct CSV for num_classes:
        if self.num_classes == 50:
            csv_name = f"{self.split}_50"
            csv_path = os.path.join(
                self.root_dir, 'annotations', f"{csv_name}.csv")

            if not os.path.exists(csv_path):
                # Keep only the 50 classes from label mapping file
                remove_classes_csv(
                    root_dir=self.root_dir,
                    orig_csv=f"{self.split}.csv",
                    small_csv=f"{csv_name}.csv",
                    skiprows=1,
                    label_mapping_csv=f'{label_mapping_name}.csv'
                )

        # Load the cleaned CSV file
        clean_csv_name = f'{csv_name}_firstclean'
        csv_clean_path = os.path.join(
            self.root_dir, 'annotations', f"{clean_csv_name}.csv")

        if not os.path.exists(csv_clean_path):
            _clean_csv(
                root_dir=self.root_dir,
                subdir=None,
                split=self.split,
                orig_csv=f'{csv_name}.csv',
                clean_csv=f"{clean_csv_name}.csv",
                skiprows=0 if self.num_classes == 50 else 1
            )

        # cleaned csv has no header
        data = pd.read_csv(csv_clean_path, header=None, skiprows=0)

        # Assign names to columns
        data.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

        # Create paths including the correct filename
        data['video_filename'] = (
            data['video_id'] + '_'
            + data['start'].apply(lambda x: f"{x:06}") + '_'
            + data['end'].apply(lambda x: f"{x:06}") + '.mp4'
        )

        data['paths'] = (
            self.root_dir + '/'
            + data['split'] + '/'
            + data['label'] + '/'
            + data['video_filename'])

        # Load label to number mapping
        label_mapping_file = os.path.join(
            self.root_dir, 'annotations', f'{label_mapping_name}.csv')

        if not os.path.exists(label_mapping_file):
            _label_mapping(self.root_dir)

        # Read existing label mappings
        unique_labels = pd.read_csv(
            label_mapping_file, header=None)[0].tolist()

        label_to_number = {label: idx for idx,
                           label in enumerate(unique_labels)}
        data['label_numeric'] = data['label'].map(label_to_number)

        # Retrieve labels and paths
        return data['paths'].tolist(),  data['label_numeric'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a video and its corresponding label from the dataset.
        Reads a video file from disk, converts it to a tensor, ensures landscape orientation, 
        applies augmentations (if specified)

        Args:
            idx : Index of the video in the dataset

        Returns:
            x : Tensor of shape (channels, num_frames, height, width), pixel values are normalized to [0, 1]
            label : Numerical label corresponding to the video
        """

        path = self.paths[idx]
        label = self.labels[idx]

        # Load video using OpenCV
        cap = cv2.VideoCapture(path)

        # Disable cv2 multithreading
        # cv2.setNumThreads(0)

        # Get the number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Hold video frames (frame_count, height, width, channels)
        frames = np.empty((frame_count, frame_height,
                          frame_width, 3), dtype=np.uint8)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[i] = frame

        cap.release()  # Release the video capture object

        x = torch.from_numpy(frames) / 255.0
        x = x.permute(0, 3, 1, 2).float()  # (frames, channels, height, width)

        # Put all videos into landscape format
        if frame_height > frame_width:
            x = x.permute(0, 1, 3, 2)

        if self.augmentations:
            x = augment(self.augmentations)(x)

        x = x.permute(1, 0, 2, 3)  # (channels, num_frames, height, width)
        if self.transforms:
            x = self.transforms(x)

        return x, label

# Dataset class for prepocessed kinetics dataset class (resized and frame window, stored as tensors)


class KineticsDatasetPreprocessed(Dataset):
    """
    Dataset class for the preprocessed Kinetics sub dataset

    Args:
        root_dir : Directory where the preprocessed Kinetics dataset is stored
        split : Dataset split to load ('train', 'val', or 'test')
        transforms : Transforms to be applied to each video
        augmentations : List of augmentations to be applied on the video, 
            e.g. ['flip_h', 'flip_v', 'noise', 'crop', 'brightness']
        file_extension : File extension of videos in sub dataset, '.mp4' or '.pt'
    """

    def __init__(self, root_dir, sub_dir, split, transforms=None, augmentations=None, file_extension='.mp4'):
        self.root_dir = root_dir  # Root directory containing dataset
        self.sub_dir = sub_dir  # Subdirectory for resized dataset
        self.split = split  # 'train', 'val', 'test'
        self.split_dir = os.path.join(self.root_dir, self.sub_dir, self.split)
        self.file_extension = file_extension
        valid_file_extension = ['.mp4', '.pt']
        assert self.file_extension in valid_file_extension, f"Invalid file_extension. Got {self.file_extension}, must be in {valid_file_extension}."

        self.transforms = transforms
        self.augmentations = augmentations
        valid_augmentations = ["flip_h", "flip_v",
                               "noise", "crop", "brightness"]
        if self.augmentations:
            assert all(
                aug in valid_augmentations for aug in self.augmentations), "Invalid augmentations, check spelling"

        self.paths, self.labels = self.__paths_and_labels_from_csv()

    def __paths_and_labels_from_csv(self):
        """
        Uses CSV files to generate paths to video files and get corresponding numerical labels

        The method ensures that the correct CSV file for the specified split is used. It also cleans the data if 
        the cleaned CSV file is not found. Additionally, if the label mapping file does not exist, it will create it


        Returns:
            paths : List of full paths to the video files
            labels : List of numerical labels corresponding to each video
        """

        # Load the cleaned CSV file
        csv_file_path = os.path.join(
            self.root_dir, 'annotations', f'{self.split}_clean_50.csv')
        # Label-to-number mapping
        label_mapping_file = os.path.join(
            self.root_dir, 'annotations', 'label_mapping_50.csv')

        assert os.path.exists(
            csv_file_path), f"Cleaned CSV doesn't exist at {csv_file_path}, creation happens in KineticsDataset class call."
        assert os.path.exists(
            label_mapping_file), f"Label mapping doesn't exist at {label_mapping_file}, creation happens in KineticsDataset class call."

        data = pd.read_csv(csv_file_path, header=None, skiprows=1)

        # Assign names to columns
        data.columns = ['label', 'video_id', 'start', 'end', 'split', 'index']

        # Create paths including the correct filename
        data['video_filename'] = (
            data['video_id'] + '_'
            + data['start'].apply(lambda x: f"{x:06}") + '_'
            + data['end'].apply(lambda x: f"{x:06}") + self.file_extension
        )

        data['paths'] = (
            self.root_dir + '/'
            + self.sub_dir + '/'
            + data['split'] + '/'
            + data['label'] + '/'
            + data['video_filename'])

        # Read existing label mappings
        unique_labels = pd.read_csv(
            label_mapping_file, header=None)[0].tolist()

        label_to_number = {label: idx for idx,
                           label in enumerate(unique_labels)}
        data['label_numeric'] = data['label'].map(label_to_number)

        # Retrieve labels and paths
        return data['paths'].tolist(),  data['label_numeric'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a video and its corresponding label from the dataset.
        Depending on self.file_extension, either reads a .mp4 video file from disk and converts it to a tensor or it reads a tensor,
        applies transformations and augmentations (if specified)

        Args:
            idx : Index of the video in the dataset

        Returns:
            x : Tensor of shape (channels, num_frames, height, width), pixel values are normalized to [0, 1]
            label : Numerical label corresponding to the video
        """
        # Get video path and label
        path = self.paths[idx]
        label = self.labels[idx]

        if self.file_extension == '.mp4':
            # Load video using OpenCV
            cap = cv2.VideoCapture(path)

            # Get number of frames in video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Hold video frames (frame_count, height, width, channels)
            frames = np.empty((frame_count, frame_height,
                              frame_width, 3), dtype=np.uint8)

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[i] = frame

            cap.release()  # Release the video capture object

            x = torch.from_numpy(frames) / 255.0
            # (num_frames, channels, height, width)
            x = x.permute(0, 3, 1, 2).float()
        else:
            x = torch.load(path)  # (channels, num_frames, height, width)
            # (num_frames, channels, height, width)
            x = x.permute(1, 0, 2, 3).float()

        if self.augmentations:
            x = augment(self.augmentations)(x)

        x = x.permute(1, 0, 2, 3)  # (channels, num_frames, height, width)
        if self.transforms:
            x = self.transforms(x)

        return x, label
