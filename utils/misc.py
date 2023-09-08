import os
import random

import numpy as np
import torch

from evaluation import metrics


def seeder(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[INFO] Setting SEED: " + str(args.seed))
    else:
        print("[INFO] Setting SEED: None")


def select_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type: " + str(device))
    return device


def evaluate(test_loader_photo, test_loader_sketch, e_photo, e_sketch):
    prec_at_100 = metrics.mAP(test_loader_photo, test_loader_sketch, e_photo, e_sketch, k=100)
    mAP = metrics.mAP(test_loader_photo, test_loader_sketch, e_photo, e_sketch)
    print(f"Precision: {prec_at_100} | Mean average precision: {mAP}")


def save(args, epoch, g_photo, g_sketch, e_photo, e_sketch, optimizer_g, optimizer_e, scheduler):
    print('Saving checkpoint')
    torch.save({
        'epoch': epoch,
        'generator_photo': g_photo.state_dict(),
        'generator_sketch': g_sketch.state_dict(),
        'encoder_photo': e_photo.state_dict(),
        'encoder_sketch': e_sketch.state_dict(),
        'optimizer_generator': optimizer_g.state_dict(),
        'optimizer_encoders': optimizer_e.state_dict(),
        'scheduler': scheduler.state_dict()
    }, os.path.join(args.output_dir + 'encoders.pth'))
