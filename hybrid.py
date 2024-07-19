from IPython.display import Image, SVG, display
import os
from pathlib import Path

import random
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torchvision
import torch
import contextlib
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print('Cuada device: ', torch.cuda.get_device_name(0))

from data_utils.download_mnist import *

train_set, valid_set, test_set = load_mnist()

### Plotting examples
from plotting.plot_examples import *
# plot_examples(train_set)

# Model
from models.MultiLayerPerceptron import *
from parameters import *
from plotting.plot_results import *
from training.BasicOptimizer_class import *
from training.training import *
from models.HebbianBackpropMultiLayerPerceptron import *

# Dataloaders

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

HEBB_LR = 1e-4 # lower, since Hebbian gradients are much bigger than backprop gradients
N_CLASSES = 3

print('*** Hybrid ***')

HybridMLP, Hybrid_results_dict = train_model_extended(
    train_set=train_set, valid_set=valid_set,
    model_type="hebbian",
    keep_num_classes=N_CLASSES,
    num_epochs=50,
    partial_backprop=True, # backprop on the final layer
    lr=[HEBB_LR / 5, LR], # learning rates for each layer
)

plot_results(Hybrid_results_dict)
plot_scores_per_class(Hybrid_results_dict)
plot_weights(HybridMLP)
plt.show()
