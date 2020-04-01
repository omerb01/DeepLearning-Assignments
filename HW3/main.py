from models import model
from train import start
import torch
import torch.nn as nn
import torch.optim as optim
import data
import utils
import preprocess_vocab
import preprocess_images
import train


def calc_accuracy(accs):
    return (torch.stack(accs, dim=0).sum(dim=0).data.item() / len(accs)) * 100


def evaluate_hw3():
    # all data files should be inside "coco" folder in the project directory
    preprocess_images.run(is_evaluate=True)
    preprocess_vocab.run()
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model.Net(val_loader.dataset.num_tokens)).cuda()
    net.load_state_dict(torch.load('model.pkl', map_location=lambda storage, loc: storage))

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    answers, accuracies, idx = start(net, val_loader, optimizer, tracker, train=False, prefix='val')
    acc = calc_accuracy(accuracies)
    print('%.2f' % acc)


def main():
    preprocess_images.run()
    preprocess_vocab.run()
    train.run()


if __name__ == '__main__':
    main()
