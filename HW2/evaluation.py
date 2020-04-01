import torch
from utils import load_dataset, calc_error
from models.omer_cnn import CNN


def test_model(net, test_loader):
    net.eval()
    incorrect = 0
    total = 0

    for images, labels in test_loader:
        images = images
        labels = labels
        outputs = net(images)
        incorrect += calc_error(outputs, labels)
        total += labels.size(0)

    accuracy = 100 - (100 * float(incorrect) / float(total))
    return accuracy


def evaluate_hw2():
    _, test_loader = load_dataset(batch_size=100)
    model = CNN()
    model.load_state_dict(torch.load('model.pkl', map_location=lambda storage, loc: storage))
    accuracy = test_model(model, test_loader)
    return accuracy
