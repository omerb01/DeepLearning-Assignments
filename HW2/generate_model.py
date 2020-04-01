import torch
import torch.nn as nn
from tqdm import tqdm

from models.omer_cnn import CNN
from utils import load_dataset, to_gpu, save_model, calc_error


def train_model(net, criterion, optimizer, train_loader):
    incorrect = 0
    total = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = to_gpu(images)
        labels = to_gpu(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        incorrect += calc_error(outputs, labels)
        total += labels.size(0)

    error = 100 * float(incorrect) / float(total)

    return error, loss.item()


def test_model(net, criterion, test_loader):
    net.eval()
    incorrect = 0
    total = 0

    for images, labels in test_loader:
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = net(images)
        loss = criterion(outputs, labels)
        incorrect += calc_error(outputs, labels)
        total += labels.size(0)

    error = 100 * float(incorrect) / float(total)
    return error, loss.item()


def lr_schedule(epoch):
    learning_rate = 0.001
    if epoch > 45:
        learning_rate = 0.0005
    if epoch > 70:
        learning_rate = 0.0003
    return learning_rate


if __name__ == '__main__':

    num_epochs = 100

    best_accuracy = 0
    while True:
        net = to_gpu(CNN())
        n_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Num of trainable parameters : ', n_trainable_parameters)
        assert n_trainable_parameters < 50000

        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader = load_dataset(batch_size=100)

        results = []
        for epoch in range(num_epochs):
            optimizer = torch.optim.Adam(net.parameters(), lr=lr_schedule(epoch))
            train_error, train_loss = train_model(net, criterion, optimizer, train_loader)
            test_error, test_loss = test_model(net, criterion, test_loader)
            results.append((train_error, train_loss, test_error, test_loss))
            accuracy = 100 - test_error
            print('Epoch: [%d/%d], Accuracy: %.2f %%' % (epoch + 1, num_epochs, accuracy))

            if accuracy > best_accuracy and accuracy >= 80:
                print('Saving...')
                save_model(net, results, accuracy)
                best_accuracy = accuracy
