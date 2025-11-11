import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import random

from train import fit, validate, load_model
from data import VideoData, GetData
from models.GATModel import GATModel
from torch.utils.data import DataLoader
from torchvision import transforms as T


def main():
    # train_model()
    parser = argparse.ArgumentParser(
        description="Train GATModel on KineticsDataset")

    # Model hyperparameters:

    # input features
    parser.add_argument("--in_features", type=int,
                        default=375, help="Number of input features")

    # number of heads
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")

    # hidden features
    parser.add_argument("--hidden_features", type=int,
                        default=256, help="Number of hidden features")

    # output features
    parser.add_argument("--out_features", type=int,
                        default=128, help="Number of output features")

    # dropout
    parser.add_argument("--dropout", type=float,
                        default=0.3, help="Dropout rate")

    # learning rate
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # segmentation level
    parser.add_argument("--maxlvl", type=int, default=6,
                        help="Number of levels for supervoxel")
    
    # time patch
    parser.add_argument("--time_patch", type=int, default=12,
                        help="Patch size for image time dimension")

    # space patch
    parser.add_argument("--space_patch", type=int, default=5,
                        help="Patch size for image space dimension")

    # Algorithm choice
    parser.add_argument("--sv", type=str, default='orig', help="Which algorithm to use, orig for original, alt for alternative")
    
    # space level
    parser.add_argument("--space_lvl", type=int, default=4, help="How many iterations in space dimension")

    # time level
    parser.add_argument("--time_lvl", type=int, default=2, help="How many iterations in time dimension")
    
    # weight initialization
    parser.add_argument("--weight_init", type=str, default='kaiming',
                        help="Weight initialization, one of ['kaiming', 'xavier']")

    
    # training parameters

    # epochs
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")

    # batch size
    parser.add_argument("--batch_size", type=int,
                        default=80, help="Batch size")

    # verbose
    parser.add_argument("--verbose", action='store_true',
                        help="Print training details")

    # slurm id
    parser.add_argument("--slurm_id", type=str,
                        default=None, help="SLURM job ID")

    # slurm
    parser.add_argument("--slurm", action='store_true',
                        help="Not used yet, just to simplyfy SLURM scripts")

    # data workers
    parser.add_argument("--data_workers", type=int, default=6,
                        help="Should often be as high as possible, due to bottleneck. Scales with batch size")

    # pre fetch
    parser.add_argument("--pre_fetch", type=int, default=3,
                        help="Number of pre_fetchers")

    # create new data
    parser.add_argument("--create_new_data",
                        action='store_true', help="Create new dataset")

    # training, test split
    parser.add_argument("--split", type=str, choices=[
                        'train', 'val', 'test'], default='train', help="One of ['train', 'val', 'test']")

    # load model
    parser.add_argument("--load_model", type=str,
                        default=None, help="Load model from file")
    parser.add_argument("--checkpoint", type=int, default=5,
                        help="How often to save model checkpoint")

    # evaluate
    parser.add_argument("--evaluate", action="store_true")

    # dataset
    parser.add_argument("--dataset", type=int, default=1,
                        help="1 for less f and less r, 2 for less f, 3 for less r")

    args = parser.parse_args()

    # Model hyperparameters
    
    NUM_HEADS = args.num_heads
    HIDDEN_FEATURES = args.hidden_features
    OUT_FEATURES = args.out_features
    DROPOUT = args.dropout
    LEARNING_RATE = args.lr
    MAXLVL = args.maxlvl
    TIME_PATCH = args.time_patch
    SPACE_PATCH = args.space_patch
    SV = args.sv
    SPACE_LVL = args.space_lvl
    TIME_LVL = args.time_lvl
    WEIGHT_INIT = args.weight_init
    IN_FEATURES = args.in_features
    IN_FEATURES = 3 * TIME_PATCH * SPACE_PATCH * SPACE_PATCH

    # training parameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    VERBOSE = args.verbose
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SLURM_ID = args.slurm_id
    DATA_WORKERS = args.data_workers
    PRE_FETCH = args.pre_fetch
    SPLIT = args.split
    LOAD_MODEL = args.load_model
    EVALUATE = args.evaluate
    DATASET = args.dataset

    path = "/cluster/work/projects/ec35/ec-philipth/"

    num_frames = 0
    if DATASET == 1:
        dataset = "UCF_less_f_less_r"
        num_frames = 20
    elif DATASET == 2:
        dataset = "UCF_less_f"
        num_frames = 20
    elif DATASET == 3:
        dataset = "UCF_less_r"
        num_frames = 50
    else:
        raise Exception("No dataset chosen")

    print(f"Path to data: {path}/{dataset}")

    model = GATModel(
        in_features=IN_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        out_features=OUT_FEATURES,
        num_heads=NUM_HEADS,
        num_classes=101,
        dropout=DROPOUT,
        time_patch=TIME_PATCH,
        space_patch=SPACE_PATCH,
        weight_init=WEIGHT_INIT
    )

    #optimizer_scheduler = None

    if LOAD_MODEL:
        #print("Before loading:", model.classifier.weight.abs().mean())
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.normpath(os.path.join(base_dir, '..', 'models', LOAD_MODEL))
        #optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_MAX=100)
        load_model(model, model_path)
        print(f"Model loaded from {LOAD_MODEL}", flush=True)

    if EVALUATE == True:
        import seaborn as sns
        te_data = GetData(path, csv="test")
        X_test, y_test = te_data.get_dataset()
        test_dataset = VideoData(X_test, y_test, os.path.join(path, dataset), te_data.classes, num_frames=num_frames)
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
        average_loss, test_accuracy, test_top_5_accuracy, conf_mtx = validate(
            model,
            test_loader,
            device=DEVICE,
            maxlvl=MAXLVL,
            sv=SV,
            space_patch=SPACE_PATCH,
            time_patch=TIME_PATCH,
            space_lvl=SPACE_LVL,
            time_lvl=TIME_LVL,
            return_confusion_matrix=True,
            verbose=False
        )

        print(f"TEST SET EVALUTION SCORES: {'=' * 60}")
        print(f" * Average Loss: {average_loss:.4f}")
        print(f" * Top-1 Accuracy: {test_accuracy:.4f}")
        print(f" * Top-5 Accuracy: {test_top_5_accuracy:.4f}")

    else:
        # Load data
        t_data = GetData(path, csv="train")
        X_train, y_train = t_data.get_dataset()
        train_dataset = VideoData(X_train, y_train, os.path.join(path, dataset), t_data.classes, augmentations=["flip_h", "flip_v", "noise", "crop", "brightness"], num_frames=num_frames)
        v_data = GetData(path, csv="val")
        X_val, y_val = v_data.get_dataset()
        val_dataset = VideoData(X_val, y_val, os.path.join(path, dataset), v_data.classes, num_frames=num_frames)
    
        #print(len(X_train))
    
        
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
        fit(
            model,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            optimizer_name="adamW",
            device=DEVICE,
            maxlvl=MAXLVL,
            time_patch=TIME_PATCH,
            space_patch=SPACE_PATCH,
            sv=SV,
            space_lvl=SPACE_LVL,
            time_lvl=TIME_LVL,
            verbose=VERBOSE,
        )


if __name__ == "__main__":
    main()


# ps -o pid,%mem,rsz,comm -p 3987350 && free -h
