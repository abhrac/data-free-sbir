#!/usr/bin/env python

import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Training SBIR encoders in a data-free manner")
        parser.add_argument('--dataset', type=str, default='Sketchy', choices=['Sketchy'])
        parser.add_argument('--data', type=str, default='/home/ac1151/Datasets/Sketchy/reduced/photo/')
        parser.add_argument('--path_sketch', type=str, default='/home/ac1151/Datasets/Sketchy/reduced/test/sketch/')
        parser.add_argument('--path_photo', type=str, default='/home/ac1151/Datasets/Sketchy/reduced/test/photo/')
        parser.add_argument('--teacher_dir', type=str, default='./cache/models/')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes in the dataset')
        parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
        parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
        parser.add_argument('--gen_iters', type=int, default=500, help='number of generator iterations/epoch')
        parser.add_argument('--lr_G', type=float, default=1e-5, help='learning rate')
        parser.add_argument('--lr_E', type=float, default=1e-5, help='learning rate')
        parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
        parser.add_argument('--im_size', type=int, default=224, help='size of each image dimension')
        parser.add_argument('--channels', type=int, default=3, help='number of image channels')
        parser.add_argument('--oh', type=float, default=3, help='one hot loss')
        parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
        parser.add_argument('--a', type=float, default=0.1, help='activation loss')
        parser.add_argument('--output_dir', type=str, default='./cache/models/')
        parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
        parser.add_argument('--gpu', default='0', type=str, help='GPU id in case of multiple GPUs')
        parser.add_argument('--eval_every', default='10', type=int, help='Evaluation frequency in epochs')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
