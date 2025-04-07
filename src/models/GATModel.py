import random
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from tqdm import tqdm

from modules import GATLayerSparse
from supervoxel import Supervoxel
from utils import add_cls_tokens, accuracy, format_time, top_5_accuracy, confusion_matrix
import os


# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)


class GATModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        num_heads,
        num_classes,
        dropout,
        time_patch,
        space_patch,
        weight_init='kaiming',
    ):
        super().__init__()
        self.gat1 = GATLayerSparse(
            in_feat=in_features,
            out_feat=hidden_features * num_heads,
            concat_hidden=True,
            num_heads=num_heads,
            dropout=dropout,
            activation='lrelu',
            v2=True,
            weight_init=weight_init
        )
        self.gat2 = GATLayerSparse(
            in_feat=hidden_features * num_heads,
            out_feat=out_features,
            concat_hidden=False,
            num_heads=1,
            dropout=dropout,
            activation='none',  # No activation in the output layer
            v2=True,
            weight_init=weight_init
        )

        # Get supervoxels
        self.sv = Supervoxel(
            device="cuda", time_patch=time_patch, space_patch=space_patch)
        node_feature_size = 3 * time_patch * space_patch ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, node_feature_size))
        if weight_init == 'kaiming':
            nn.init.kaiming_uniform_(
                self.cls_token, a=0.2, nonlinearity='leaky_relu')
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.cls_token)
        else:
            raise ValueError(f"Invalid weight initialization: {weight_init}")

        self.batch_norm_1 = nn.LayerNorm(hidden_features * num_heads)
        self.batch_norm_2 = nn.LayerNorm(out_features)

        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(
            in_features=out_features, out_features=num_classes)

        self.train_losses = []
        self.train_top_5_accuracies = []

        self.epoch_losses = []
        self.epoch_accuracies = []

        self.validation_accuracies = []
        self.validation_losses = []
        self.val_top_5_accuracies = []

        self.model_details = (
            f"init_{weight_init}_in_feat_{in_features}_hid_feat_{hidden_features}_"
            f"out_feat_{out_features}_num_head_{num_heads}_"
            f"num_cls_{num_classes}_dropout_{dropout}"
        )

        self.epochs_trained = 0
        self.best_weights = None
        self.best_val_accuracy = 0
        self.best_epoch = 0

    def forward(self, x, adj, seg_index):
        x = self.gat1(x, adj)
        x = self.batch_norm_1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)

        x = self.gat2(x, adj)  # [batch_size, num_nodes, out_features]
        x = self.batch_norm_2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        # Get CLS tokens
        x = x.squeeze(0)[seg_index]

        x = self.classifier(x)

        return x

    def predict(self, x):
        ...

    def validate(self, data_loader, device=None, maxlvl=4, return_confusion_matrix=False, verbose=True):
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Ensure model is on the correct device
        device = device or self.device
        model = self.to(device)

        model.eval()

        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for i, (vid, labels) in enumerate(data_loader):
                segs, edges, features, seg_indexes = self.sv.process(
                    vid=vid.to(device),
                    maxlvl=maxlvl,
                )

                features = features.to(device)
                features = torch.flatten(features, start_dim=1, end_dim=-1)

                # Normalize features
                flattened_features = features.view(-1, features.size(-1))
                features_mean = flattened_features.mean(dim=0, keepdim=True)
                features_std = flattened_features.std(
                    dim=0, keepdim=True) + 1e-6
                features = (features - features_mean) / features_std

                features, edges, cls_indexes = add_cls_tokens(
                    features, edges, seg_indexes, self.cls_token, device)

                labels = labels.to(device)
                edges = edges.to(device)
                edges = edges.unsqueeze(0)
                features = features.unsqueeze(0)

                output = model(features, edges, cls_indexes)

                loss = loss_fn(output, labels)
                total_loss += loss.item()

                all_predictions.extend(output)
                all_labels.extend(labels)

        average_loss = total_loss / len(data_loader)
        conf_mtx = confusion_matrix(all_predictions, all_labels)
        val_accuracy = accuracy(all_predictions, all_labels)
        val_top_5_accuracy = top_5_accuracy(all_predictions, all_labels)

        if verbose:
            print(
                f"\n\nValidation Loss: {average_loss:.4f}, Validation top-1 Accuracy: {val_accuracy:.4f}, top-5 Accuracy: {val_top_5_accuracy:.4f}")

        if return_confusion_matrix:
            return average_loss, val_accuracy, val_top_5_accuracy, conf_mtx

        return average_loss, val_accuracy, val_top_5_accuracy

    def fit(
        self,
        train_data_loader,
        val_data_loader=None,
        epochs=100,
        lr=0.005,
        optimizer_name='adam',
        device: torch.device = None,
        maxlvl=4,
        time_patch=5,
        space_patch=12,
        verbose=True,
        checkpoints=7,
    ):

        flush_interval = 20

        optimizers = {
            'adam': optim.Adam,
            'adamW': optim.AdamW,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
        }

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optimizers.get(optimizer_name, optim.Adam)(
            self.parameters(), lr=lr, weight_decay=5e-4)

        # Move stuff to device
        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        model = self.to(device)

        num_vids = len(train_data_loader)

        # early_stop = 20
        # num_vids = early_stop

        print(
            f"Training model for {epochs} epochs with {num_vids} batches each...")
        train_details = f"space_patch_{space_patch}_time_patch_{time_patch}_LR_{lr}_Optimizer_{optimizer_name}_MaxLvl_{maxlvl}_current_run_epochs_{epochs}"
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
                # Do not calculate the time for the first iteration for average, as it is much slower
                if i == 1:
                    epoch_start_time = perf_counter()

                data_load_time = perf_counter() - prev_time

                forward_time = perf_counter()

                # vid = vid.to(device)
                segs, edges, features, seg_indexes = self.sv.process(
                    vid=vid.to(device),
                    maxlvl=maxlvl,
                )

                features = features.to(device)

                features = torch.flatten(features, start_dim=1, end_dim=-1)

                # Normalize values
                flattened_features = features.view(-1, features.size(-1))
                features_mean = flattened_features.mean(dim=0, keepdim=True)
                features_std = flattened_features.std(
                    dim=0, keepdim=True) + 1e-6
                features = (features - features_mean) / features_std

                features, edges, cls_indexes = add_cls_tokens(
                    features, edges, seg_indexes, self.cls_token, device)

                labels = labels.to(device)
                edges = edges.to(device)
                edges = edges.unsqueeze(0)
                features = features.unsqueeze(0)

                # print(f"trilinear: {trilinear.shape}, Adj: {adj.shape}, Labels: {labels.shape}")

                
                pre_time = perf_counter()
                preprocessing_time = pre_time - forward_time
                model.train()
                optimizer.zero_grad()
                output = model(features, edges, cls_indexes)
                #print(f"Labels: {torch.unique(labels)}", flush=True)
                #print(f"labels shape: {labels.shape}")
                #print(f"output shape: {output.shape}")
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                model_time = perf_counter() - pre_time

                self.train_losses.append(loss.item())
                epoch_loss.append(loss.item())
                epoch_predictions.extend(output)
                epoch_labels.extend(labels)

                predictions.append(output)
                all_labels.append(labels)
                # if (i + 1) % num_vids // 10 == 0:
                if (i + 1) % 400 == 0 or i in [1, 4, 24, 54, 99]:
                    # pred_labels = output.argmax(dim=1)
                    print(
                        f" * ({i+1}/{num_vids}), Epoch Loss: {np.mean(epoch_loss):.4f}", flush=True)
                    # print(f" ---- pred: {pred_labels.tolist()}")
                    # print(f" ---- true: {labels.tolist()}")
                    print(
                        f"  --- Data load time: {format_time(data_load_time)}")
                    print(
                        f"  --- Supervoxel time: {format_time(preprocessing_time)}")
                    print(f"  --- Model time: {format_time(model_time)}")
                    print(
                        f"  --- Total time for batch: {format_time(perf_counter() - prev_time)}")
                    print(
                        f"  --- Expected time left for epoch: {format_time((perf_counter() - epoch_start_time) / (i + 1) * (num_vids - i))}", flush=True)
                # if i + 1 >= early_stop:
                #     print(f"Ending epoch early after {i+1} iterations")
                #     break

                # Flush the output, to force print if SLURM is used
                # if i % flush_interval == 0:
                #     print("", flush=True)

                prev_time = perf_counter()

            if (epoch+1) % checkpoints == 0:
                self.save_model(
                    name=f"checkpoints/{train_details}_total_epochs_{self.epochs_trained}_{self.model_details}.pt")

            self.epochs_trained += 1

            if val_data_loader:
                val_loss, val_accuracy, val_top_5_accuracy = self.validate(
                    val_data_loader,
                    device,
                    maxlvl=maxlvl
                )
                self.validation_losses.append(val_loss)
                self.validation_accuracies.append(val_accuracy)
                self.val_top_5_accuracies.append(val_top_5_accuracy)

                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_weights = self.state_dict()
                    self.best_epoch = self.epochs_trained

            epoch_accuracy = accuracy(epoch_predictions, epoch_labels)
            epoch_top_5_accuracy = top_5_accuracy(
                epoch_predictions, epoch_labels)

            self.epoch_accuracies.append(epoch_accuracy)
            self.train_top_5_accuracies.append(epoch_top_5_accuracy)
            self.epoch_losses.append(np.mean(epoch_loss))

            elapsed_time = perf_counter() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, _ = divmod(rem, 60)
            print(f"\nEpoch Loss: {np.mean(epoch_loss):.4f}, Epoch top-1 accuracy: {epoch_accuracy:.4f}, Epoch top-5 accuracy: {epoch_top_5_accuracy}, Total training time: {int(hours)}H {int(minutes)}M, Total epochs: {self.epochs_trained}")

        short_id = format(random.randint(0, 99999), 'x')

        train_details = f"total_epochs_{self.epochs_trained}_{train_details}"
        print(f"{train_details}_{self.model_details}_id_{short_id}")
        self.save_model(f"id_{short_id}")
        self.save_best(
            f"best_{train_details}_{self.model_details}_id_{short_id}")
        self.save_iteration_loss_plot(
            f"ITERATION_LOSS_{train_details}_{self.model_details}_id_{short_id}")
        self.save_epoch_accuracy_loss_plots(
            f"{train_details}_{self.model_details}_id_{short_id}")

    def save_iteration_loss_plot(self, name="loss"):

        os.makedirs('plots', exist_ok=True)

        plt.plot(self.train_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f"plots/{name}.png")
        plt.show()
        plt.clf()

    def save_epoch_accuracy_loss_plots(self, name="training"):

        # Ensure the 'plots' directory exists
        os.makedirs('plots', exist_ok=True)

        # Plot Training and Validation Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch_accuracies, label='Training Accuracy',
                 color='blue', marker='o')
        plt.plot(self.validation_accuracies,
                 label='Validation Accuracy', color='green', marker='s')
        plt.plot(self.train_top_5_accuracies,
                 label='Training Top-5 Accuracy', color='red', marker='x')
        plt.plot(self.val_top_5_accuracies,
                 label='Validation Top-5 Accuracy', color='orange', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Set y axis of accuracy plot to be between 0 and 1
        ymin = 0
        ymax = max(1, max(self.epoch_accuracies + self.validation_accuracies))
        plt.ylim(ymin, ymax)
        plt.yticks(np.arange(ymin, ymax + 0.1, 0.1))

        max_accuracy = f"{max(self.validation_accuracies, default=0):.4f}"
        accuracy_filename = f"plots/accuracy_{max_accuracy}_{self.epochs_trained}_{name}.png"
        plt.savefig(accuracy_filename)
        print(f"Saved Accuracy Plot as {accuracy_filename}")
        plt.show()
        plt.close()

        # Plot Training Loss
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch_losses, label='Training Loss',
                 color='red', marker='x')
        plt.plot(self.validation_losses, label='Validation Loss',
                 color='orange', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_filename = f"plots/loss_{max_accuracy}_({self.epochs_trained})_{name}.png"
        plt.savefig(loss_filename)
        print(f"Saved Loss Plot as {loss_filename}")
        plt.show()
        plt.close()

    def save_model(self, name="model"):
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/checkpoints', exist_ok=True)

        path = os.path.join('models', f"{name}.pth")

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_details': self.model_details,
            'epochs_trained': self.epochs_trained,
            'epoch_accuracies': self.epoch_accuracies,
            'epoch_losses': self.epoch_losses,
            'validation_accuracies': self.validation_accuracies,
            'validation_losses': self.validation_losses,
            'train_top_5_accuracies': self.train_top_5_accuracies,
            'val_top_5_accuracies': self.val_top_5_accuracies,
            'losses': self.train_losses,
            'best_weights': self.best_weights,
            'best_epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_accuracy,
        }
        print(f"\nSaving to {path}\n")

        torch.save(checkpoint, path)

    def save_best(self, name="best_model"):
        if self.best_weights is None:
            print("No best weights to save")
            return

        os.makedirs('models', exist_ok=True)

        path = os.path.join('models', f"{name}.pth")

        checkpoint = {
            'model_state_dict': self.best_weights,
            'model_details': self.model_details,
            'epochs_trained': self.best_epoch,
            'epoch_accuracies': self.epoch_accuracies,
            'epoch_losses': self.epoch_losses,
            'validation_accuracies': self.validation_accuracies,
            'validation_losses': self.validation_losses,
            'train_top_5_accuracies': self.train_top_5_accuracies,
            'val_top_5_accuracies': self.val_top_5_accuracies,
            'losses': self.train_losses,
            'best_epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'best_weights': self.best_weights,
        }
        print(f"\nSaving to {path}\n")

        torch.save(checkpoint, path)

    def load(self, file):
        # Load the checkpoint from the specified file
        checkpoint = torch.load(file)

        # Load the model's state dictionary
        state_dict = checkpoint['model_state_dict']
        self.load_state_dict(state_dict)

        # Load additional model details if they exist in the checkpoint
        self.model_details = checkpoint.get(
            'model_details', self.model_details)
        self.epochs_trained = checkpoint.get('epochs_trained', 0)

        self.epoch_accuracies = checkpoint.get(
            'epoch_accuracies', [0*self.epochs_trained])
        self.validation_accuracies = checkpoint.get(
            'validation_accuracies', [0*self.epochs_trained])

        self.train_top_5_accuracies = checkpoint.get(
            'train_top_5_accuracies', [0*self.epochs_trained])
        self.val_top_5_accuracies = checkpoint.get(
            'val_top_5_accuracies', [0*self.epochs_trained])

        self.epoch_losses = checkpoint.get('epoch_losses', [])
        self.train_losses = checkpoint.get('losses', [])
        self.validation_losses = checkpoint.get(
            'validation_losses', [0*self.epochs_trained])

        self.best_weights = checkpoint.get('best_weights', None)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0)
