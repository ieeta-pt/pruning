import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.profilers import Profiler, SimpleProfiler, PyTorchProfiler

import numpy as np
import matplotlib.pyplot as plt

from Custom_Profiler import Custom_Profiler
from LRP import LRP
from callback import LRP_callback

import torch
from torch.utils.data import DataLoader, TensorDataset

# Dummy dataset parameters
num_samples = 10
num_features = 5
num_classes = 2

# Create random dummy data
X_train = torch.randn(num_samples, num_features)
y_train = torch.randint(0, num_classes, (num_samples,))

X_val = torch.randn(num_samples, num_features)
y_val = torch.randint(0, num_classes, (num_samples,))

X_test = torch.randn(num_samples, num_features)
y_test = torch.randint(0, num_classes, (num_samples,))

# Wrap tensors into TensorDataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)


# This section serves just to setup the dataloaders for the example usage.
# In practice, you would replace this with your actual dataset loading logic.

func_to_time = [
    '_compute_tau_gradient', '_compute_r_gradient', '_update_masks', '_compute_dtdr',
    '_update_weight_gradient', '_compute_global_t','forward', '_update_sorted_weights', '_incremental_sort', '_apply_hard_pruning',
]
profiler = Custom_Profiler(functions_to_profile=func_to_time)

model = LRP(r0 = 0.99, tau0 = 1e-2, profiler = profiler, lr_r0 = 1e-4, lr_tau0 = 1e-6, lamb = 1e1)
callback_two_PBDP = LRP_callback()
early_stopping = EarlyStopping(monitor="r", patience=5, mode="max")

early_stopping = EarlyStopping(monitor="r", patience=10, mode="max", min_delta=1e-4)



trainer = pl.Trainer(max_epochs=25, accelerator="gpu", devices="auto", callbacks = [callback_two_PBDP])#, profiler = profiler


# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader)