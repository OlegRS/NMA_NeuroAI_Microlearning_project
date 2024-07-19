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

from data_utils.download_mnist import *

train_set, valid_set, test_set = download_mnist()

### Plotting examples
from plotting.plot_examples import *
# plot_examples(train_set)
# plt.show()

# Model
from models.MultiLayerPerceptron import *
from parameters import *

MLP = MultiLayerPerceptron(
    num_hidden=NUM_HIDDEN,
    activation_type=ACTIVATION,
    bias=BIAS,
    )

# Dataloaders

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

from training.BasicOptimizer_class import *

backprop_optimizer = BasicOptimizer(MLP.parameters(), lr=LR)

from training.training import *

NUM_EPOCHS = 5

MLP_results_dict = train_model(
    MLP,
    train_loader,
    valid_loader,
    optimizer=backprop_optimizer,
    num_epochs=NUM_EPOCHS
    )

from plotting.plot_results import *

# plot_results(MLP_results_dict)
# plot_scores_per_class(MLP_results_dict)
# plot_examples(valid_loader.dataset, MLP=MLP);
# plot_weights(MLP=MLP);
# plt.show()



############ HEBBIAN ################
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

Hebb_results_dict_2cls = train_model(
    HebbianMLP_2cls,
    train_loader_2cls,
    valid_loader_2cls,
    Hebb_optimizer_2cls,
    num_epochs=10
    );

plot_results(Hebb_results_dict_2cls, num_classes=2)
plot_scores_per_class(Hebb_results_dict_2cls, num_classes=2)
plot_examples(valid_loader_2cls.dataset, MLP=HebbianMLP_2cls, num_classes=2)
plot_weights(HebbianMLP_2cls)

HebbianMLP_2cls = HebbianMultiLayerPerceptron(
    num_hidden=NUM_HIDDEN,
    num_outputs=2,
    clamp_output=True, # clamp output to targets
)

Hebb_optimizer_2cls = BasicOptimizer(HebbianMLP_2cls.parameters(), lr=HEBB_LR)

Hebb_results_dict_2cls = train_model(
    HebbianMLP_2cls,
    train_loader_2cls,
    valid_loader_2cls,
    Hebb_optimizer_2cls,
    num_epochs=10
    );

plot_results(Hebb_results_dict_2cls, num_classes=2)
plot_scores_per_class(Hebb_results_dict_2cls, num_classes=2)
plot_examples(valid_loader_2cls.dataset, MLP=HebbianMLP_2cls, num_classes=2)
plot_weights(HebbianMLP_2cls);

# from train_model_extended import *

MLP_3cls, results_dict_3cls = train_model_extended(
    train_set=train_set, valid_set=valid_set,
    model_type="backprop",
    keep_num_classes=3,
    num_epochs=5,
    lr=LR,
    plot_distribution=True
)

plot_results(results_dict_3cls, num_classes=3)
plot_scores_per_class(results_dict_3cls, num_classes=3)
plot_weights(MLP_3cls)

HebbianMLP_3cls, Hebbian_results_dict_3cls = train_model_extended(
    train_set=train_set, valid_set=valid_set,
    model_type="hebbian",
    keep_num_classes=3,
    num_epochs=15,
    lr=HEBB_LR,
)

plot_results(Hebbian_results_dict_3cls, num_classes=3)
plot_scores_per_class(Hebbian_results_dict_3cls, num_classes=3)
plot_weights(HebbianMLP_3cls)

HebbianMLP_3cls, Hebbian_results_dict_3cls = train_model_extended(
    train_set=train_set, valid_set=valid_set,
    model_type="hebbian",
    keep_num_classes=3,
    num_epochs=15,
    lr=[HEBB_LR / 4, HEBB_LR * 8], # learning rate for each layer
)

plot_results(Hebbian_results_dict_3cls, num_classes=3)
plot_scores_per_class(Hebbian_results_dict_3cls, num_classes=3)
plot_weights(HebbianMLP_3cls)

### HebbianBackpropMultiLayerPerceptron
from models.HebbianBackpropMultiLayerPerceptron import *

HybridMLP, Hybrid_results_dict = train_model_extended(
    train_set=train_set, valid_set=valid_set,
    model_type="hebbian",
    num_epochs=5,
    partial_backprop=True, # backprop on the final layer
    lr=[HEBB_LR / 5, LR], # learning rates for each layer
)

plot_results(Hybrid_results_dict)
plot_scores_per_class(Hybrid_results_dict)
plot_weights(HybridMLP);

print('Finish')
