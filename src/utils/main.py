import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import random

from data import KineticsDataset, KineticsDatasetPreprocessed, store_new_dataset_mp4, WindowTransform, _clean_csv
from models.GATModel import GATModel
from torch.utils.data import DataLoader
from torchvision import transforms as T

# Store a transformed dataset (tensors), can be a subset of the whole dataset (chosen by start and end idx)


def create_sub_dataset(root_dir, subdir, from_mp4=True):

    parser = argparse.ArgumentParser(
        description="New (sub) dataset of tensors")
    parser.add_argument("--split", type=str, choices=[
                        'train', 'val', 'test'], default='train', help="One of ['train', 'val', 'test']")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start intex for partitioned dataset")
    parser.add_argument("--end_idx", type=int, default=7999,
                        help="End index for partitioned dataset")
    args = parser.parse_args()

    SPLIT = args.split
    START_IDX = args.start_idx
    END_IDX = args.end_idx

    resize = (160, 160)
    every_nth_frame = 2
    total_frames = 20

    transforms = T.Compose(
        [
            T.Resize(size=resize),
            WindowTransform(every_nth_frame, total_frames),
        ]
    )
    # Use original Kinetics dataset
    if from_mp4:
        orig_dataset = KineticsDataset(
            root_dir, SPLIT, num_classes=50, transforms=transforms)
    else:  # Use preprocessed dataset of tensors
        orig_dataset = KineticsDatasetPreprocessed(
            root_dir, "resized", SPLIT, transforms=transforms, file_extension='.pt')
    store_new_dataset_mp4(orig_dataset, subdir, START_IDX, END_IDX)
    print(f"New {SPLIT} dataset is stored")


# Clean csv for preprocessed and saved dataset
def create_new_csv(root_dir, subdir, split, num_classes=50):

    orig_name = f"{split}_50_firstclean" if num_classes == 50 else f"{split}_firstclean"
    clean_name = f"{split}_clean_50" if num_classes == 50 else f"{split}_clean"

    _clean_csv(
        root_dir,
        subdir,
        split,
        f"{orig_name}.csv",
        f"{clean_name}.csv"
    )


def main():
    # train_model()
    parser = argparse.ArgumentParser(
        description="Train GATModel on KineticsDataset")
    parser.add_argument("--in_features", type=int,
                        default=375, help="Number of input features")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--hidden_features", type=int,
                        default=256, help="Number of hidden features")
    parser.add_argument("--dropout", type=float,
                        default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--maxlvl", type=int, default=6,
                        help="Number of levels for supervoxel")
    parser.add_argument("--time_patch", type=int, default=12,
                        help="Patch size for image time dimension")
    parser.add_argument("--space_patch", type=int, default=5,
                        help="Patch size for image space dimension")
    parser.add_argument("--batch_size", type=int,
                        default=80, help="Batch size")
    parser.add_argument("--out_features", type=int,
                        default=128, help="Number of output features")
    parser.add_argument("--verbose", action='store_true',
                        help="Print training details")
    parser.add_argument("--slurm_id", type=str,
                        default=None, help="SLURM job ID")
    parser.add_argument("--slurm", action='store_true',
                        help="Not used yet, just to simplyfy SLURM scripts")
    parser.add_argument("--data_workers", type=int, default=6,
                        help="Should often be as high as possible, due to bottleneck. Scales with batch size")
    parser.add_argument("--pre_fetch", type=int, default=3,
                        help="Number of pre_fetchers")
    parser.add_argument("--create_new_data",
                        action='store_true', help="Create new dataset")
    parser.add_argument("--split", type=str, choices=[
                        'train', 'val', 'test'], default='train', help="One of ['train', 'val', 'test']")
    parser.add_argument("--load_model", type=str,
                        default=None, help="Load model from file")
    parser.add_argument("--checkpoint", type=int, default=5,
                        help="How often to save model checkpoint")
    parser.add_argument("--weight_init", type=str, default='kaiming',
                        help="Weight initialization, one of ['kaiming', 'xavier']")
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    NUM_HEADS = args.num_heads
    HIDDEN_FEATURES = args.hidden_features
    DROPOUT = args.dropout
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    TIME_PATCH = args.time_patch
    SPACE_PATCH = args.space_patch
    MAXLVL = args.maxlvl
    BATCH_SIZE = args.batch_size
    OUT_FEATURES = args.out_features
    VERBOSE = args.verbose
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SLURM_ID = args.slurm_id
    DATA_WORKERS = args.data_workers
    PRE_FETCH = args.pre_fetch
    SPLIT = args.split
    CREATE_NEW_DATA = args.create_new_data
    LOAD_MODEL = args.load_model
    CHECKPOINT = args.checkpoint
    WEIGHT_INIT = args.weight_init
    EVALUATE = args.evaluate

    IN_FEATURES = args.in_features
    IN_FEATURES = 3 * TIME_PATCH * SPACE_PATCH * SPACE_PATCH

    root_dir = "/cluster/work/projects/ec395/group10/videos"
    subdir = "resized_20frames"

    model = GATModel(
        in_features=IN_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        out_features=OUT_FEATURES,
        num_heads=NUM_HEADS,
        num_classes=50,
        dropout=DROPOUT,
        time_patch=TIME_PATCH,
        space_patch=SPACE_PATCH,
        weight_init=WEIGHT_INIT
    )

    if LOAD_MODEL:
        model.load(LOAD_MODEL)
        print(f"Model loaded from {LOAD_MODEL}", flush=True)

    if EVALUATE == True:
        import seaborn as sns
        test_dataset = KineticsDatasetPreprocessed(root_dir, subdir, "test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DATA_WORKERS,
            drop_last=True,
            prefetch_factor=PRE_FETCH,
            pin_memory=True
        )
        print("Starting evaluation on test set")
        average_loss, test_accuracy, test_top_5_accuracy, conf_mtx = model.validate(
            test_loader, device=DEVICE, maxlvl=MAXLVL, return_confusion_matrix=True, verbose=False)

        print(f"TEST SET EVALUTION SCORES: {'=' * 60}")
        print(f" * Average Loss: {average_loss:.4f}")
        print(f" * Accuracy: {test_accuracy:.4f}")
        print(f" * Top-5 Accuracy: {test_top_5_accuracy:.4f}")

        # Create class name mapping
        path_labels = "/cluster/work/projects/ec395/group10/videos/annotations/label_mapping_50.csv"
        unique_labels = pd.read_csv(path_labels, header=None)[0].tolist()

        conf_matrix_np = conf_mtx.numpy()
        df_conf_matrix = pd.DataFrame(conf_matrix_np,
                                      index=[f'{unique_labels[i]}' for i in range(
                                          conf_matrix_np.shape[0])],
                                      columns=[f'{unique_labels[i]}' for i in range(conf_matrix_np.shape[1])])

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(25, 25))
        sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title('Confusion Matrix')
        plt.show()

        os.makedirs('plots', exist_ok=True)
        os.makedirs('plots/confusion_matrices', exist_ok=True)

        short_id = format(random.randint(0, 99999), 'x')
        file_name = f"confusion_matrix_{short_id}"
        file_path = f"plots/confusion_matrices/{file_name}.png"
        plt.savefig(file_path)
        print(f"saved confusion matrix to {file_path}")

        return

    if CREATE_NEW_DATA:
        create_sub_dataset(root_dir, subdir)
        create_new_csv(root_dir, subdir, SPLIT)
        print(f"New {SPLIT} dataset is stored at {root_dir}/{subdir}")
        return

    # Load data
    train_dataset = KineticsDatasetPreprocessed(root_dir, subdir, "train", augmentations=[
                                                "flip_h", "flip_v", "noise", "crop", "brightness"])
    val_dataset = KineticsDatasetPreprocessed(root_dir, subdir, "val")

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # worker_init_fn=worker_init_fn,
        num_workers=DATA_WORKERS,
        drop_last=True,
        prefetch_factor=PRE_FETCH,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATA_WORKERS,
        drop_last=True,
        prefetch_factor=PRE_FETCH,
        pin_memory=True
    )

    print("Starting training:")
    model.fit(
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        optimizer_name="adam",
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        maxlvl=MAXLVL,
        time_patch=TIME_PATCH,
        space_patch=SPACE_PATCH,
        verbose=VERBOSE,
        device=DEVICE,
        checkpoints=CHECKPOINT,
    )


if __name__ == "__main__":
    main()


# ps -o pid,%mem,rsz,comm -p 3987350 && free -h
