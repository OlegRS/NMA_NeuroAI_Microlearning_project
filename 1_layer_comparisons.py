import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Nicer fonts for matplotlib
import matplotlib
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['mathtext.fontset']='cm'

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

############### SGD softmax ################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print('Cuada device: ', torch.cuda.get_device_name(0))

# Hyperparameters
input_size = 784  # 28x28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to GPU if available
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Reshape data
        data = data.reshape(data.shape[0], -1)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent or Adam step
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Check accuracy on training and test set
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
    model.train()
    return num_correct / num_samples

weights_tensor = model.linear.weight

# Convert to numpy array
W_SGD = weights_tensor.cpu().detach().numpy()

train_acc = check_accuracy(train_loader, model)
test_acc = check_accuracy(test_loader, model)
print(f'Training Accuracy: {train_acc*100:.2f}%')
print(f'Test Accuracy: {test_acc*100:.2f}%')

### Cosine similarity
cos = (W_SGD.flatten() @ W.flatten())/(np.linalg.norm(W_SGD.flatten(), ord=2)*np.linalg.norm(W.flatten()))
print("Cosine similarity:", cos)


###### Comparing weight matrices
### "Reverse inference" (generation from labels)
### What we get when the network is run backwards
def reverse_inference(n_classes, label, weights):
    output_layer_activity = target_output_from_label(n_classes, label)
    return (weights.T @ output_layer_activity).reshape(28,28)

fig, axs = plt.subplots(nrows=2, ncols=N_classes, figsize=(17, 3.8))
for j in range(N_classes):
    axs[0,j].imshow(reverse_inference(N_classes, j, weights=W))#, cmap='gray')
    axs[0,j].set_xticks([])
    axs[0,j].set_yticks([])
    axs[1,j].imshow(reverse_inference(N_classes, j, weights=W_SGD))#, cmap='gray')
    axs[1,j].set_xticks([])
    axs[1,j].set_yticks([])
fig.suptitle('Reverse inference (top:Hebbian, bottom:SGD)', fontsize=20)


### Comparing weight matrices
fig_weights, axs_weights = plt.subplots(nrows=2, figsize=(17/1.3, 8/1.3))
axs_weights[0].imshow(W, aspect='auto', interpolation='none')
axs_weights[0].set_xticks([])
axs_weights[0].set_yticks(np.arange(10))
axs_weights[1].set_yticks(np.arange(10))
axs_weights[1].imshow(W_SGD - W_SGD.flatten().min(), aspect='auto', interpolation='none')

axs_weights[0].set_ylabel("Classes", fontsize=18)
axs_weights[1].set_ylabel("Classes", fontsize=18)
axs_weights[1].set_xlabel("Flattened weights (vertical lines separate different rows of the weight matrix)", fontsize=18)

for i in range(28):
    axs_weights[0].axvline(28*i,color='red')
    axs_weights[1].axvline(28*i,color='red')

for i in range(10):
    axs_weights[0].axhline(i-.5,color='red')
    axs_weights[1].axhline(i-.5,color='red')

cos = (W_SGD.flatten() @ W.flatten())/(np.linalg.norm(W_SGD.flatten(), ord=2)*np.linalg.norm(W.flatten()))
title = f'Weights (top:Hebbian, bottom:SGD); cosine_similarity={cos:.3f}'
fig_weights.suptitle(title, fontsize=19)

plt.tight_layout()
plt.show()
