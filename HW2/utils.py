import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    # CIFAR-10 Dataset
    train_dataset = dsets.CIFAR10(root='./data/', train=True, transform=train_transform, download=True)
    test_dataset = dsets.CIFAR10(root='./data/', train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def calc_error(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    incorrect = (predicted != labels).sum()
    return incorrect


def save_model(net, results, accuracy):
    torch.save(net.state_dict(), 'results/%.2f.pkl' % accuracy)
    df = pd.DataFrame(results, columns=['train_error', 'train_loss', 'test_error', 'test_loss'])
    df.to_csv('results/%.2f.csv' % accuracy)


def show_plots(csv_path):
    df = pd.read_csv(csv_path)
    error = df[['train_error', 'test_error']]
    loss = df[['train_loss', 'test_loss']]

    error.plot()
    plt.title('Error')
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

    loss.plot()
    plt.title('Loss')
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
