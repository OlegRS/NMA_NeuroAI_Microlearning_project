import torch
import numpy as np
import matplotlib.pyplot as plt

from data_utils.download_mnist import *

train_set, valid_set, test_set = load_mnist()

N_classes = 10

def target_output_from_label(n_classes, label):
    output = np.zeros(n_classes)
    output[label] = 1
    return output


### Prepating the data
inputs, labels, outputs = [], [], []
for i in range(len(train_set.indices)):
    if train_set[i][1] < N_classes:
        inputs.append(train_set[i][0].numpy().reshape(28*28))
        labels.append(train_set[i][1])
        outputs.append(target_output_from_label(n_classes=N_classes,
                                                label=train_set[i][1]))

inputs_val, labels_val = [], []
for i in range(len(valid_set.indices)):
    if valid_set[i][1] < N_classes:
        inputs_val.append(valid_set[i][0].numpy().reshape(28*28))
        labels_val.append(valid_set[i][1])

inputs_test, labels_test = [], []
for i in range(len(test_set.indices)):
    if test_set[i][1] < N_classes:
        inputs_test.append(test_set[i][0].numpy().reshape(28*28))
        labels_test.append(test_set[i][1])

W = np.zeros([N_classes, 28*28])

### Training
lr = 1
for i in range(len(outputs)):
    W += lr * np.outer(outputs[i], inputs[i])

### Testing
def prediction(x):
    return np.argmax(W @ x)

def softmax(x):
    # Subtract the maximum value from the input array for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

predictions_train = np.array([prediction(inputs[i]) for i in range(len(inputs))])
predictions_val = np.array([prediction(inputs_val[i]) for i in range(len(inputs_val))])
predictions_test = np.array([prediction(inputs_test[i]) for i in range(len(inputs_test))])

# accuracy := 1 - (fraction of misclassified digits)
accuracy_train = 1 - np.count_nonzero(predictions_train - np.array(labels))/len(labels)
accuracy_val = 1 - np.count_nonzero(predictions_val - np.array(labels_val))/len(labels_val)
accuracy_test = 1 - np.count_nonzero(predictions_test - np.array(labels_test))/len(labels_test)

print("accuracy_train=", accuracy_train)
print("accuracy_validation=", accuracy_val)
print("accuracy_test=", accuracy_test)

### "Receptive fields" (generation from labels)
def generate_from_label(n_classes, label):
    output_layer_activity = target_output_from_label(n_classes, label)
    return (W.T @ output_layer_activity).reshape(28,28)
    # return (weights_numpy.T @ output_layer_activity).reshape(28,28)

fig, axs = plt.subplots(ncols=N_classes, figsize=(10*1.7, 1.3*1.7))
for i in range(N_classes):
    axs[i].imshow(generate_from_label(N_classes, i), cmap='gray')
    axs[i].set_xticks([])
    axs[i].set_yticks([])

fig.suptitle('Learned receptive fields of the output units', fontsize=20)
plt.tight_layout()
plt.show()

# Cosyne similarity
# (weights_numpy.flatten() @ W.flatten())/(np.linalg.norm(weights_numpy.flatten(), ord=2)*np.linalg.norm(W.flatten(), ord=2))
