import argparse
import os

import torch
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models


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
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        output = classifier(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()


def test(classifier, data_test_loader, criterion):
    classifier.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = classifier(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test_loader.dataset)
    acc = float(total_correct) / len(data_test_loader.dataset)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))


def main():
    parser = argparse.ArgumentParser(description='train-teacher-network')

    parser.add_argument('--data', type=str, default='/home/ac1151/Datasets/Sketchy/reduced/')
    parser.add_argument('--output_dir', type=str, default='./cache/models/')
    parser.add_argument('--num_epochs', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_train = datasets.ImageFolder(root=os.path.join(args.data, 'train', 'photo'), transform=transforms)
    data_test = datasets.ImageFolder(root=os.path.join(args.data, 'test', 'photo'), transform=transforms)

    data_train_loader = DataLoader(data_train, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    data_test_loader = DataLoader(data_test, batch_size=100, shuffle=False, pin_memory=True, num_workers=4)

    classifier = models.resnet34(pretrained=True)
    classifier.fc = nn.Linear(classifier.fc.in_features, 10)
    classifier = classifier.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=5e-4)

    for e in range(args.num_epochs):
        train(classifier, data_train_loader, optimizer, criterion, e)
        test(classifier, data_test_loader, criterion)
        torch.save(classifier.state_dict(), args.output_dir + 'photo_classifier.pth')


if __name__ == '__main__':
    main()
