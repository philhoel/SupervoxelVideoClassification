import torch
import torch.nn as nn
from time import perf_counter
import random
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from utils import add_cls_tokens, accuracy, format_time, top_5_accuracy, confusion_matrix, CosineDecay
import numpy as np




def fit(
    model,
    train_data_loader,
    val_data_loader,
    epochs=100,
    lr=0.005,
    #optimizer_scheduler=None,
    optimizer_name="adamW",
    device: torch.device = None,
    maxlvl = 4,
    time_patch = 5,
    space_patch = 12,
    sv = 'orig',
    space_lvl = 4,
    time_lvl = 2,
    verbose = True,
):
    flush_interval = 20

    # Variables
    epochs_trained = 0
    train_losses = []
    train_acc = []
    train_top_5_acc = []
    val_losses = []
    val_acc = []
    val_top_5_acc = []
    

    optimizers = {
        'adam': optim.Adam,
        'adamW': optim.AdamW,
        'sgd': optim.SGD,
        'adagrad': optim.Adagrad
    }

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optimizers.get(optimizer_name, optim.AdamW)( model.parameters(), lr=lr, weight_decay=3e-3)

    scheduler = CosineDecay(
        optimizer=optimizer,
        lr_start=lr,
        lr_stop=0.001 * lr,
        epochs=epochs,
        warmup_ratio=0.1,
        num_steps=len(train_data_loader),
        warmup_mode='sinusoidal'
    )

    #if optimizer_scheduler is None:
    #     optimizer = optimizers.get(optimizer_name, optim.AdamW)( model.parameters(), lr=lr, weight_decay=3e-3)

    #    scheduler = CosineDecay(
    #        optimizer=optimizer,
    #        lr_start=lr,
    #        lr_stop=0.001 * lr,
    #        epochs=epochs,
    #        warmup_ratio=0.1,
    #        num_steps=len(train_data_loader),
    #        warmup_mode='sinusoidal'
    #    )
    #else:
    #    optimizer = optimizer_scheduler[0]
    #    scheduler = optimizer_scheduler[1]

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    num_vids = len(train_data_loader)

    print(f"Training model for {epochs} epochs with {num_vids} batches each...")

    start_time = perf_counter()
    predictions = []
    all_labels = []

    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch+1}/{epochs}\n{'='*60}")
        epoch_start_time = perf_counter()
        epoch_loss = []
        epoch_predictions = []
        epoch_labels = []
        prev_time = perf_counter()

        model.train()
        for i, (vid, labels) in enumerate(train_data_loader):

            if i == 1:
                epoch_start_time = perf_counter()

            data_load_time = perf_counter() - prev_time
            forward_time = perf_counter()

            # Call supervoxel algorithm
            segs, edges, features, seg_indexes = model.supervoxel.process(
                vid=vid.to(device),
                maxlvl=maxlvl,
                sv=sv,
                space_patch=space_patch,
                time_patch=time_patch,
                space_lvl=space_lvl,
                time_lvl=time_lvl
            )

            features = features.to(device)
            features = torch.flatten(features, start_dim=1, end_dim=-1)
            flattened_features = features.view(-1, features.size(-1))
            features_mean = flattened_features.mean(dim=0, keepdim=True)
            features_std = flattened_features.std(dim=0, keepdim=True) + 1e-6
            features = (features - features_mean) / features_std

            features, edges, cls_indexes = add_cls_tokens(features, edges, seg_indexes, model.cls_token, device)
            labels = labels.to(device)
            edges = edges.to(device)
            edges = edges.unsqueeze(0)
            features = features.unsqueeze(0)

            pre_time = perf_counter()
            preprocessing_time = pre_time - forward_time
            model.train()
            optimizer.zero_grad()
            output = model(features, edges, cls_indexes)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            model_time = perf_counter() - pre_time

            model.train_losses.append(loss.item())
            epoch_loss.append(loss.item())
            epoch_predictions.extend(output)
            epoch_labels.extend(labels)

            predictions.append(output)
            all_labels.append(labels)


            if (i + 1) % 400 == 0 or i in [1,4,24,54,99]:

                print(f" * ({i+1}/{num_vids}), Epoch Loss: {np.mean(epoch_loss):.4f}", flush=True)
                print(f"  --- Data load time: {format_time(data_load_time)}")
                print(f"  --- Supervoxel time: {format_time(preprocessing_time)}")
                print(f"  --- Model time: {format_time(model_time)}")
                print(f"  --- Total time for batch: {format_time(perf_counter() - prev_time)}")
                print(f"  --- Expected time left for epoch: {format_time((perf_counter() - epoch_start_time) / (i+1) * (num_vids - i))}", flush=True)

            prev_time = perf_counter()

        epochs_trained += 1

        if val_data_loader:
            val_loss, val_accuracy, val_top_5_accuracy = validate(
                model,
                val_data_loader,
                device,
                maxlvl,
                sv=sv,
                space_patch=space_patch,
                time_patch=time_patch,
                space_lvl=space_lvl,
                time_lvl=time_lvl
            )

            val_losses.append(val_loss)
            val_acc.append(val_accuracy)
            val_top_5_acc.append(val_top_5_accuracy)

        epoch_accuracy = accuracy(epoch_predictions, epoch_labels)
        epoch_top_5_accuracy = top_5_accuracy(epoch_predictions, epoch_labels)

        train_losses.append(np.mean(epoch_loss))
        train_acc.append(epoch_accuracy)
        train_top_5_acc.append(epoch_top_5_accuracy)

        elapsed_time = perf_counter() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, _ = divmod(rem, 60)
        print(f"\nEpoch Loss: {np.mean(epoch_loss):.4f}, Epoch top-1 accuracy: {epoch_accuracy:.4f}, Epoch top-5 accuracy: {epoch_top_5_accuracy:.4f}, Total training time: {int(hours)}H {int(minutes)}M, Total epochs: {epochs_trained}")

    short_id = format(random.randint(0,99999), 'x')
    print(f"short_id: id_{short_id}")

    stats = {
        'train_loss': train_losses,
        'train_acc': train_acc,
        'train_top_5': train_top_5_acc,
        'val_loss': val_losses,
        'val_acc': val_acc,
        'val_top_5': val_top_5_acc
    }
    
    save_model(model, f"id_{short_id}")
    save_stats_to_csv(stats, f"id_{short_id}")

################################################
    
###### End of fit function #############

################################################

def validate(
    model,
    val_data_loader,
    device = None,
    maxlvl = 4,
    sv = 'orig',
    space_patch = 12,
    time_patch = 5,
    space_lvl = 4,
    time_lvl = 2,
    return_confusion_matrix = False,
    verbose = True
):

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    device = device or model.device
    model = model.to(device)

    total_loss = 0
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i, (vid, labels) in enumerate(val_data_loader):
            segs, edges, features, seg_indexes = model.supervoxel.process(
                vid=vid.to(device),
                maxlvl=maxlvl,
                sv=sv,
                space_patch=space_patch,
                time_patch=time_patch,
                space_lvl=space_lvl,
                time_lvl=time_lvl
            )

            features = features.to(device)
            features = torch.flatten(features, start_dim=1, end_dim=-1)
            
            flattened_features = features.view(-1, features.size(-1))
            features_mean = flattened_features.mean(dim=0, keepdim=True)
            features_std = flattened_features.std(dim=0, keepdim=True) + 1e-6
            features = (features - features_mean) / features_std

            features, edges, cls_indexes = add_cls_tokens(features, edges, seg_indexes, model.cls_token, device)

            labels = labels.to(device)
            edges = edges.to(device)
            edges = edges.unsqueeze(0)
            features = features.unsqueeze(0)

            output = model(features, edges, cls_indexes)

            loss = loss_fn(output, labels)
            total_loss += loss.item()

            all_predictions.extend(output)
            all_labels.extend(labels)

    average_loss = total_loss / len(val_data_loader)
    conf_mtx = confusion_matrix(all_predictions, all_labels)
    val_accuracy = accuracy(all_predictions, all_labels)
    val_top_5_accuracy = top_5_accuracy(all_predictions, all_labels)

    if verbose:
        print(f"\n\nValidation Loss: {average_loss:.4f}, Validation top-1 Accuracy: {val_accuracy:.4f}, top-5 Accuracy: {val_top_5_accuracy:.4f}")

    if return_confusion_matrix:
        return average_loss, val_accuracy, val_top_5_accuracy, conf_mtx

    return average_loss, val_accuracy, top_5_accuracy

#########################################

####    End of Validate function   #####

#########################################


#def save_model(model, optimizer, scheduler, epochs, name="model"):
def save_model(model, name="model"):
    os.makedirs('models', exist_ok=True)

    path = os.path.join('models', f"{name}.pth")
    print(f"\nSaving to {path}\n")
    torch.save({
        "model_state_dict": model.state_dict()
    }, path)

    #torch.save({
    #    "model_state_dict":  model.state_dict(),
    #    "optimizer_state_dict": optimizer.state_dict(),
    #    "scheduler_state_dict": scheduler.state_dict(),
    #    "epochs": epochs
    #    }, path)

#def load_model(model, optimizer, scheduler, epochs, file):
def load_model(model, file):

    state_dicts = torch.load(file)
    model.load_state_dict(state_dicts["model_state_dict"])
    #optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
    #scheduler.load_state_dict(state_dicts["scheduler_state_dict"])
    #epochs = state_dicts["epochs"]

def save_stats_to_csv(stats, name='stats'):
    os.makedirs('csv_stats', exist_ok=True)

    path = os.path.join('csv_stats', f"{name}.csv")
    print(f"\nSaving stats to {path}\n")

    df = pd.DataFrame(stats)
    df.insert(0, 'epoch', range(1, len(df)+1))

    df.to_csv(path, index=False)
            
