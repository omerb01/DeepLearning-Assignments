import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 14),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes))

    def forward(self, x):
        return self.layer(x)


def loadDataset(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3015,))])

    train_dataset = dsets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = dsets.FashionMNIST(root='./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def createModel(input_size, num_classes, learning_rate):
    net = NeuralNet(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    print('Number of parameters:', sum(param.numel() for param in net.parameters()))
    print('Num of trainable parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

    return net, criterion, optimizer


def trainModel(net, criterion, optimizer, train_loader):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        labels = labels

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return loss


def testModel(net, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100 * float(correct) / float(total)
    print('Accuracy of the model on the 10000 test images: %.2f %%' % accuracy)
    return accuracy


if __name__ == '__main__':

    while True:

        # Hyper Parameters
        input_size = 784
        num_classes = 10
        num_epochs = 75
        batch_size = 100
        learning_rate = 0.2

        train_loader, test_loader = loadDataset(batch_size)
        net, criterion, optimizer = createModel(input_size, num_classes, learning_rate)

        reduce_rate = 0.05
        accuracies = []
        for epoch in range(num_epochs):
            print('Learning Rate: %f' % learning_rate)
            loss = trainModel(net, criterion, optimizer, train_loader)
            print('Epoch: [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))

            accuracy = testModel(net, test_loader)
            accuracies.append(accuracy)
            if accuracy >= 88.9:
                torch.save(net, 'models/%.2f.pt' % accuracy)
                with open('models/%.2f.txt' % accuracy, 'w') as out_file:
                    out_file.write(str(accuracies))

            learning_rate *= (1 - reduce_rate)
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
