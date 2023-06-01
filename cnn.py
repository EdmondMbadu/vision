import argparse
from collections import defaultdict
import ssl

import PIL

ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np

from PIL import Image
import matplotlib

matplotlib.use('Agg')  # use Agg backend so that we don't require an X server
from matplotlib import pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

# We're not using the GPU.
use_gpu = False

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

default_train_transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize rescales and shifts the data so that it has a zero mean 
    # and unit variance. This reduces bias and makes it easier to learn!
    # The values here are the mean and variance of our inputs.
    # This will change the input images to be centered at 0 and be 
    # between -1 and 1.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

default_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Download the data from internet if not present

def get_train_loader(batch_size, transform=default_train_transform):
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform)
    return torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_test_loader(batch_size, transform=default_test_transform):
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform)
    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)


# The function we'll call to train the network each epoch
def train(net, loader, optimizer, criterion, epoch, use_gpu=False):
    running_loss = 0.0
    total_loss = 0.0

    # Send the network to the correct device
    if use_gpu:
        net = net.cuda()
    else:
        net = net.cpu()

    # tqdm is a useful package for adding a progress bar to your loops
    pbar = tqdm(loader)
    for i, data in enumerate(pbar):
        inputs, labels = data

        # If we're using the GPU, send the data to the GPU
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()  # Set the gradients of the parameters to zero.
        outputs = net(inputs)  # Forward pass (send the images through the network)
        loss = criterion(outputs, labels)  # Compute the loss w.r.t the labels.
        loss.backward()  # Backward pass (compute gradients).
        optimizer.step()  # Use the gradients to update the weights of the network.

        running_loss += loss.item()
        total_loss += loss.item()
        pbar.set_description(f"[epoch {epoch + 1}] loss = {running_loss / (i + 1):.03f}")

    average_loss = total_loss / (i + 1)
    tqdm.write(f"Epoch {epoch} summary -- loss = {average_loss:.03f}")

    return average_loss


def show_hard_negatives(hard_negatives, label, nrow=10):
    """Visualizes hard negatives"""
    grid = make_grid([(im + 1) / 2 for im, score in hard_negatives[label]],
                     nrow=nrow, padding=1)
    grid = grid.permute(1, 2, 0).mul(255).byte().numpy()
    # ipd.display(Image.fromarray((grid)))


# The function we'll call to test the network
def test(net, loader, tag='', use_gpu=False, num_hard_negatives=10):
    correct = 0
    total = 0

    # Send the network to the correct device
    net = net.cuda() if use_gpu else net.cpu()

    # Compute the overall accuracy of the network
    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {tag}"):
            images, labels = data

            # If we're using the GPU, send the data to the GPU
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            # Forward pass (send the images through the network)
            outputs = net(images)

            # Take the output of the network, and extract the index 
            # of the largest prediction for each example
            _, predicted = torch.max(outputs.data, 1)

            # Count the number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    average_accuracy = correct / total
    tqdm.write(f'{tag} accuracy of the network: {100 * average_accuracy:.02f}%')

    # Repeat above, but estimate the testing accuracy for each of the labels
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    hard_negatives = defaultdict(list)
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            predicted_scores, predicted_labels = torch.max(outputs, 1)
            correct_mask = (predicted_labels == labels).squeeze()
            incorrect_mask = ~correct_mask
            unique_labels, unique_counts = torch.unique(labels, return_counts=True)
            for l, c in zip(unique_labels, unique_counts):
                l = l.item()
                label_mask = (labels == l)
                predicted_mask = (predicted_labels == l)
                # This keeps track of the most hardest negatives
                # i.e. mistakes with the highest confidence.
                hard_negative_mask = (~correct_mask & predicted_mask)
                if hard_negative_mask.sum() > 0:
                    hard_negatives[l].extend([
                        (im, score.item())
                        for im, score in zip(images[hard_negative_mask],
                                             predicted_scores[hard_negative_mask])])
                    hard_negatives[l].sort(key=lambda x: x[1], reverse=True)
                    hard_negatives[l] = hard_negatives[l][:num_hard_negatives]
                class_correct[l] += (correct_mask & label_mask).sum()
                class_total[l] += c

    for i in range(10):
        tqdm.write(f'{tag} accuracy of {classes[i]} = {100 * class_correct[i] / class_total[i]:.02f}%')
        # if len(hard_negatives[i]) > 0:
        #    print(f'Hard negatives for {classes[i]}')
        #    show_hard_negatives(hard_negatives, i, nrow=10)
        # else:
        #    print("There were no hard negatives--perhaps the model got 0% accuracy?")

    return average_accuracy


def train_network(net,
                  lr,
                  epochs,
                  batch_size,
                  criterion=None,
                  optimizer=None,
                  lr_func=None,
                  train_transform=default_train_transform,
                  eval_interval=10,
                  use_gpu=use_gpu):
    assert optimizer is not None

    # Initialize the loss function
    if criterion is None:
        # Note that CrossEntropyLoss has the Softmax built in!
        # This is good for numerical stability. 
        # Read: https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

    # Initialize the data loaders
    train_loader = get_train_loader(batch_size, transform=train_transform)
    test_loader = get_test_loader(batch_size)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        if lr_func is not None:
            lr_func(optimizer, epoch, lr)

        train_loss = train(net, train_loader, optimizer, criterion, epoch, use_gpu=use_gpu)
        train_losses.append(train_loss)

        # Evaluate the model every `eval_interval` epochs.
        if (epoch + 1) % eval_interval == 0:
            print(f"Evaluating epoch {epoch + 1}")
            train_accuracy = test(net, train_loader, 'Train', use_gpu=use_gpu)
            test_accuracy = test(net, test_loader, 'Test', use_gpu=use_gpu)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

    return train_losses, train_accuracies, test_accuracies


# A function to plot the losses over time
def plot_results(train_losses, train_accuracies, test_accuracies):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(train_accuracies)
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    axes[2].plot(test_accuracies)
    axes[2].set_title('Testing Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')


#
# The images in CIFAR-10 are 32x32.
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        # TODO: Implement forward pass for LinearNet
        x = self.fc1(x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]))
        return x


#
class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for MLPNet
        x = self.fc1(x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]))
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 15, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(15, 75, 5)
        self.fc1 = nn.Linear(75 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for ConvNet
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def set_learning_rate(optimizer, epoch, base_lr):
    # TODO: adjust learning rate here
    lr = base_lr - (.1 * base_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


custom_train_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class MSELossClassification(nn.Module):
    def forward(self, output, labels):
        one_hot_encoded_labels = \
            torch.nn.functional.one_hot(labels, num_classes=output.shape[1]).float()
        return nn.functional.mse_loss(output, one_hot_encoded_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains and evaluates a neural network.')
    parser.add_argument('output', type=str, help='Output file path')
    parser.add_argument('network', type=str, help='Network architecture to use')
    parser.add_argument('--optim', type=str, default='SGDmomentum', help='Optimizer used to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--lr_sched', action='store_true', help='Use learning rate schedule')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Size of each minibatch')
    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--mse', action='store_true', help='Use MSE loss')

    args = parser.parse_args()

    if args.network == "linear":
        net = LinearNet()
    elif args.network == "mlp":
        net = MLPNet()
    elif args.network == "convnet":
        net = ConvNet()
    else:
        print("Unknown network")
        # raise

    if args.optim == "SGDmomentum":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        print("Unknown optimizer")
        # raise

    if args.aug:
        train_transform = custom_train_transform
    else:
        train_transform = default_train_transform

    if args.mse:
        criterion = MSELossClassification()
    else:
        criterion = nn.CrossEntropyLoss()

    lr_func = None
    if args.lr_sched:
        lr_func = set_learning_rate

    train_losses, train_accuracies, test_accuracies = train_network(
        net,
        optimizer=optimizer,
        lr=args.lr,
        lr_func=lr_func,
        criterion=criterion,
        train_transform=train_transform,
        epochs=args.epochs,
        batch_size=args.batch_size)

    plot_results(train_losses, train_accuracies, test_accuracies)
    plt.savefig(args.output)
