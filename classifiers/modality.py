import argparse
import os

import torch
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models
from utils.dataloaders import modality_classifier


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(classifier, data_train_loader, optimizer, criterion, epoch):
    classifier.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.cuda(), labels.cuda().float()

        optimizer.zero_grad()
        output = classifier(images)[:, 0]
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))


def test(classifier, data_test_loader, criterion):
    classifier.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = classifier(images)[:, 0]
            avg_loss += criterion(output, labels).sum()
            total_correct += ((output > 0.5).int() == labels).sum()

    avg_loss /= len(data_test_loader.dataset)
    acc = float(total_correct) / len(data_test_loader.dataset)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))


def main():
    parser = argparse.ArgumentParser(description='train-teacher-network')

    parser.add_argument('--data_root', type=str, default='/home/ac1151/Datasets/Sketchy/reduced/')
    parser.add_argument('--output_dir', type=str, default='./cache/models/')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trainloader, testloader = modality_classifier(args)

    classifier = models.resnet34(pretrained=True)
    classifier.fc = nn.Linear(classifier.fc.in_features, 1)
    classifier = classifier.cuda()
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=5e-4)

    for e in range(args.num_epochs):
        train(classifier, trainloader, optimizer, criterion, e)
        test(classifier, testloader, criterion)
        torch.save(classifier.state_dict(), args.output_dir + 'modality_classifier.pth')


if __name__ == '__main__':
    main()
