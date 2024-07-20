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

# Dataloaders

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

from learning_rules.HebbianFunction_class import *
from models.HebbianMultiLayerPerceptron import *
from data_utils.restrict_classes import *

train_set_2_classes = restrict_classes(train_set, [0, 1])
valid_set_2_classes = restrict_classes(valid_set, [0, 1])

plot_class_distribution(train_set_2_classes, valid_set_2_classes)

train_loader_2cls = torch.utils.data.DataLoader(train_set_2_classes, batch_size=BATCH_SIZE, shuffle=True)
valid_loader_2cls = torch.utils.data.DataLoader(valid_set_2_classes, batch_size=BATCH_SIZE, shuffle=False)

HEBB_LR = 1e-4 # lower, since Hebbian gradients are much bigger than backprop gradients

HebbianMLP_2cls = HebbianMultiLayerPerceptron(
    num_hidden=NUM_HIDDEN,
    num_outputs=2,
    clamp_output=False,
)

Hebb_optimizer_2cls = BasicOptimizer(HebbianMLP_2cls.parameters(), lr=HEBB_LR)


## -------------------------------------------------------
# print('*** Hebbian with 3 classes ***')

# HebbianMLP_3cls, Hebbian_results_dict_3cls = train_model_extended(
#     train_set=train_set, valid_set=valid_set,
#     model_type="hebbian",
#     keep_num_classes=3,
#     num_epochs=50,
#     lr=HEBB_LR,
# )

# plot_results(Hebbian_results_dict_3cls, num_classes=3)
# plot_scores_per_class(Hebbian_results_dict_3cls, num_classes=3)
# plot_weights(HebbianMLP_3cls)
# plt.show()
## -------------------------------------------------------

print('*** Hebbian with 3 classes different LR in layers ***')

HebbianMLP_3cls, Hebbian_results_dict_3cls = train_model_extended(
    train_set=train_set, valid_set=valid_set,
    model_type="hebbian",
    keep_num_classes=3,
    num_epochs=100,
    lr=[HEBB_LR/4/10, HEBB_LR*8/10], # learning rate for each layer
)

plot_results(Hebbian_results_dict_3cls, num_classes=3)
plot_scores_per_class(Hebbian_results_dict_3cls, num_classes=3)
plot_weights(HebbianMLP_3cls)
plt.show()
