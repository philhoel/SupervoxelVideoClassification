import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import v2


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
                repeats = np.ceil(self.total / frames)
                x = torch.repeat_interleave(x, repeats=repeats, dim=1)
            # Get a distribution of frames across the video
            step = frames // self.total
            result = x[:, ::step, :, :][:, :self.total, :, :]
            assert result.shape[1] == self.total
            return result


def gaussian_noise(mean=0.0, std=0.1):
    return lambda x: torch.clamp(x + torch.normal(mean, std, x.shape), 0.0, 1.0)


class GetData:

    def __init__(self, path: str, csv="train") -> None:

        self.path = path
        self.classes = []
        if csv == "train":
            self.files = os.path.join(path, "train_data.csv")
        elif csv == "val":
            self.files = os.path.join(path, "val_data.csv")
        elif csv == "test":
            self.files = os.path.join(path, "test_data.csv")

        temp_dict = {}

        with open(self.files, "r") as csv_file:
            for line in csv_file.readlines():
                filename, class_name = line.split(",")
                class_name = class_name.strip()
                temp_dict[filename] = class_name
                if class_name not in self.classes:
                    self.classes.append(class_name)

        self.class_to_idx = {}
        for idx, name in enumerate(self.classes):
            self.class_to_idx[name] = idx

        self.video_files = []
        self.labels = []
        for key in temp_dict:
            self.video_files.append(key)
            self.labels.append(self.class_to_idx[temp_dict[key]])

        self.classes = [None] * len(self.class_to_idx)
        for name in self.class_to_idx:
            self.classes[self.class_to_idx[name]] = name

    def get_dataset(self):
        return self.video_files, self.labels


class VideoData(Dataset):

    def __init__(self, X, y, path, class_names, augmentations=[], transforms=None, num_frames=20):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.class_names = class_names
        self.path = path
        self.augmentations = augmentations
        self.num_frames = num_frames

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(
            f"{self.path}/{self.class_names[self.y[idx]]}/{self.X[idx]}.mp4")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        

        frames = np.empty((self.num_frames, frame_height,
                          frame_width, 3), dtype=np.uint8)
        
        #print(self.num_frames)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i >= self.num_frames:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[i] = frame

        cap.release()

        if frame_count < self.num_frames:
            diff = self.num_frames - frame_count
            for i in range(1, diff):
                frames[frame_count + diff - 1] = frames[frame_count-1]

        x = torch.from_numpy(frames) / 255
        x = x.permute(0, 3, 1, 2).float()

        #print(f"x shape: {x.shape}")

        
        if len(self.augmentations) > 0:
            x = augment(self.augmentations, resize=(frame_height,frame_width))(x)
            

        x = x.permute(1, 0, 2, 3)

        if self.transforms:
            x = self.transforms(x)

        #print(f"shape: {x.shape}")

        return x.clone(), self.y[idx]
