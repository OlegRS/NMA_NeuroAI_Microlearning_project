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

from learning_rules.HebbianFunction_class import *
from models.MultiLayerPerceptron import *

class HebbianBackpropMultiLayerPerceptron(MultiLayerPerceptron):
  """
  Hybrid backprop/Hebbian multilayer perceptron with one hidden layer.
  """

  def forward(self, X, y=None):
    """
    Runs a forward pass through the network.

    Arguments:
    - X (torch.Tensor): Batch of input images.
    - y (torch.Tensor, optional): Batch of targets, not used here.

    Returns:
    - y_pred (torch.Tensor): Predicted targets.
    """

    # Hebbian layer
    h = HebbianFunction.apply(
        X.reshape(-1, self.num_inputs),
        self.lin1.weight,
        self.lin1.bias,
        self.activation,
    )

    # backprop layer
    y_pred = self.softmax(self.lin2(h))

    return y_pred
