import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


def test_loaders_sbir(args):
    test_transform = T.Compose([
        T.Resize((args.im_size, args.im_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_test_photo = datasets.ImageFolder(root=args.path_photo, transform=test_transform)
    test_loader_photo = DataLoader(data_test_photo, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    data_test_sketch = datasets.ImageFolder(root=args.path_sketch, transform=test_transform)
    test_loader_sketch = DataLoader(data_test_sketch, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return test_loader_photo, test_loader_sketch


class ModalityClassifier(Dataset):
    def __init__(self, data_root):
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.photos = datasets.ImageFolder(root=os.path.join(data_root, 'photo')).imgs
        self.sketches = datasets.ImageFolder(root=os.path.join(data_root, 'sketch')).imgs

    def __getitem__(self, idx):
        if idx < len(self.photos):
            return self.transforms(Image.open(self.photos[idx][0]).convert('RGB')), 0.0
        return self.transforms(Image.open(self.sketches[idx - len(self.photos)][0]).convert('RGB')), 1.0

    def __len__(self):
        return len(self.photos) + len(self.sketches)


def modality_classifier(args):
    trainset = ModalityClassifier(os.path.join(args.data_root, 'train'))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    testset = ModalityClassifier(os.path.join(args.data_root, 'test'))
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    return trainloader, testloader


def merge_photos_sketches(data_root):
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    photos = datasets.ImageFolder(root=os.path.join(data_root, 'photo'), transform=transforms)
    sketches = datasets.ImageFolder(root=os.path.join(data_root, 'sketch'), transform=transforms)
    merged = photos
    merged.samples = photos.samples + sketches.samples
    return merged


def unified_classifier(args):
    trainset = merge_photos_sketches(os.path.join(args.data_root, 'train'))
    testset = merge_photos_sketches(os.path.join(args.data_root, 'test'))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    return trainloader, testloader
