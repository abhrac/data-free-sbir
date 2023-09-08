import os

import torch
import torch.nn as nn
from torchvision import models

from networks.generator import Generator


def init_classifiers(args, device, unified=False):
    classifier_modality = models.resnet34()
    classifier_modality.fc = nn.Linear(classifier_modality.fc.in_features, 1)
    classifier_modality.load_state_dict(torch.load(args.teacher_dir + 'modality_classifier.pth'))
    classifier_modality.eval()

    if not unified:
        classifier_photo = models.resnet34()
        classifier_photo.fc = nn.Linear(classifier_photo.fc.in_features, args.num_classes)
        classifier_photo.load_state_dict(torch.load(args.teacher_dir + 'photo_classifier.pth'))
        classifier_photo.eval()

        classifier_sketch = models.resnet34()
        classifier_sketch.fc = nn.Linear(classifier_sketch.fc.in_features, args.num_classes)
        classifier_sketch.load_state_dict(torch.load(args.teacher_dir + 'sketch_classifier.pth'))
        classifier_sketch.eval()

        return classifier_photo.to(device), classifier_sketch.to(device), classifier_modality.to(device)

    classifier_unified = models.resnet34()
    classifier_unified.fc = nn.Linear(classifier_unified.fc.in_features, args.num_classes)
    classifier_unified.load_state_dict(torch.load(args.teacher_dir + 'unified_classifier.pth'))
    classifier_unified.eval()

    return classifier_unified.to(device), classifier_modality.to(device)


def init_generators(args, device):
    nz = args.latent_dim
    nd = args.im_size
    nc = args.channels

    generator_photo = Generator(nz, nd, nc)
    generator_sketch = Generator(nz, nd, nc)

    return generator_photo.to(device), generator_sketch.to(device)


def init_encoders(args, device):
    encoder_photo = models.resnet34()
    encoder_photo = torch.nn.Sequential(*list(encoder_photo.children())[:-1])
    # encoder_photo.load_state_dict(torch.load(args.output_dir + 'photo_encoder.pth'))

    encoder_sketch = models.resnet34()
    encoder_sketch = torch.nn.Sequential(*list(encoder_sketch.children())[:-1])
    # encoder_sketch.load_state_dict(torch.load(args.output_dir + 'sketch_encoder.pth'))

    return encoder_photo.to(device), encoder_sketch.to(device)


def auto_resume(args, g_photo, g_sketch, e_photo, e_sketch, optimizer_g, optimizer_e, scheduler):
    start_epoch = 0
    if os.path.exists(os.path.join(args.output_dir + 'encoders.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir + 'encoders.pth'))
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}...')
        e_photo.load_state_dict(checkpoint['encoder_photo'])
        e_sketch.load_state_dict(checkpoint['encoder_sketch'])
        g_photo.load_state_dict(checkpoint['generator_photo'])
        g_sketch.load_state_dict(checkpoint['generator_sketch'])
        optimizer_g.load_state_dict(checkpoint['optimizer_generator'])
        optimizer_e.load_state_dict(checkpoint['optimizer_encoders'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    return start_epoch
