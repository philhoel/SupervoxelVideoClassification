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

    def __init__(self, path: str, split: tuple[float, float, float]) -> None:

        self.path = path
        self.classes = []
        self.split = split

        self.video_files = []

        for class_name in sorted(os.listdir(path)):
            self.classes.append(class_name)

            videos = [v for v in os.listdir(f"{path}/{class_name}")]

            self.video_files.append(videos)

        self.num_of_classes = len(self.classes)

    def train_val_test_split(self):

        train_labels = []
        val_labels = []
        test_labels = []

        train_lst = []
        val_lst = []
        test_lst = []

        for i in range(self.num_of_classes):

            lst_temp = []

            for j, item in enumerate(self.video_files[i]):

                lst_temp.append(item)

            perm = torch.randperm(len(lst_temp))
            lst_temp = [lst_temp[k] for k in perm]

            n = len(self.video_files[i])
            n1 = int(n*self.split[1])
            n2 = int(n*self.split[2])

            val = lst_temp[:n1]
            test = lst_temp[n1:n2+n1]
            train = lst_temp[n2+n1:]

            for x in train:
                train_lst.append(x)
                train_labels.append(i)

            for y in val:
                val_lst.append(y)
                val_labels.append(i)

            for z in test:
                test_lst.append(z)
                test_labels.append(i)

        perm1 = torch.randperm(len(train_lst))
        train_lst = [train_lst[q] for q in perm1]
        train_labels = [train_labels[q] for q in perm1]

        perm2 = torch.randperm(len(val_lst))
        val_lst = [val_lst[q] for q in perm2]
        val_labels = [val_labels[q] for q in perm2]

        perm3 = torch.randperm(len(test_lst))
        test_lst = [test_lst[q] for q in perm3]
        test_labels = [test_labels[q] for q in perm3]

        self.train_lst = train_lst
        self.val_lst = val_lst
        self.test_lst = test_lst

        return train_lst, train_labels, val_lst, val_labels, test_lst, test_labels


class VideoData(Dataset):

    def __init__(self, X, y, transforms, path, class_names, crop=True):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.class_names = class_names
        self.path = path
        self.crop = crop
        self.augmentations = None

    def crop_videos(self, height, width, frames):
        pass

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(
            f"{self.path}/{self.class_names[self.y[idx]]}/{self.X[idx]}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frames = np.empty((frame_count, frame_height,
                          frame_width, 3), dtype=np.uint8)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[i] = frame

        cap.release()

        x = torch.from_numpy(frames) / 255
        x = x.permute(0, 3, 1, 2).float()

        x = x[:20, :, :, :]

        if self.augmentations:
            x = augment(self.augmentations)(x)

        x = x.permute(1, 0, 2, 3)

        if self.transforms:
            x = self.transforms(x)

        return x, self.y[idx]
