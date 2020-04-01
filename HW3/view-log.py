import sys
import torch
import matplotlib
import numpy as np

matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1]
    results = torch.load(path)

    train_acc = torch.FloatTensor(results['tracker']['train_acc'])
    train_acc = train_acc.mean(dim=1).numpy()
    train_acc = np.apply_along_axis(lambda acc: (1 - acc) * 100, 0, train_acc)
    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()
    val_acc = np.apply_along_axis(lambda acc: (1 - acc) * 100, 0, val_acc)

    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Error')
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig('acc.png')

    train_loss = torch.FloatTensor(results['tracker']['train_loss'])
    train_loss = train_loss.mean(dim=1).numpy()
    val_loss = torch.FloatTensor(results['tracker']['val_loss'])
    val_loss = val_loss.mean(dim=1).numpy()

    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Loss')
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')


if __name__ == '__main__':
    main()
