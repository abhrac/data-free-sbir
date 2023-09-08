import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed


def acc_at_1(photo_loader, sketch_loader, encoder_photo, encoder_sketch):
    encoder_photo.eval(), encoder_sketch.eval()
    gallery_reprs = []
    gallery_labels = []
    with torch.no_grad():
        for photo, label in photo_loader:
            photo, label = photo.cuda(), label.cuda()
            photo_reprs = encoder_photo(photo)[:, :, 0, 0]
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)

        total = 0
        total_correct = 0
        for sketch, label in sketch_loader:
            sketch, label = sketch.cuda(), label.cuda()
            sketch_reprs = F.normalize(encoder_sketch(sketch)[:, :, 0, 0])
            ranks = torch.argsort(torch.matmul(sketch_reprs, gallery_reprs.T), dim=1, descending=True)
            num_correct = torch.sum(gallery_labels[ranks[:, 0]] == label).item()

            total = total + len(sketch)
            total_correct = total_correct + num_correct
            print(f"Correct/Total: {total_correct}/{total}")


def compute_avg_prec(sketch_label, retrieval, tgt_set_size=10):
    num_correct = 0
    avg_prec = 0
    for photo_idx, photo_class in enumerate(retrieval, start=1):
        if photo_class == sketch_label:
            num_correct += 1
            avg_prec = avg_prec + (num_correct / photo_idx)
            if num_correct == tgt_set_size:
                break
    if num_correct > 0:
        avg_prec = avg_prec / num_correct

    return avg_prec


def mAP(photo_loader, sketch_loader, encoder_photo, encoder_sketch, k=None):
    encoder_photo.eval(), encoder_sketch.eval()
    num_cores = min(multiprocessing.cpu_count(), 32)
    gallery_reprs = []
    gallery_labels = []
    with torch.no_grad():
        for photo, label in photo_loader:
            photo, label = photo.cuda(), label.cuda()
            photo_reprs = encoder_photo(photo)[:, :, 0, 0]
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)

        aps_all = []
        for sketch, label in sketch_loader:
            sketch, label = sketch.cuda(), label.cuda()
            sketch_reprs = F.normalize(encoder_sketch(sketch)[:, :, 0, 0])
            ranks = torch.argsort(torch.matmul(sketch_reprs, gallery_reprs.T), dim=1, descending=True)
            # num_correct = torch.sum(gallery_labels[ranks[:, 0]] == label).item()
            retrievals = gallery_labels[ranks]
            if k is not None:
                retrievals = gallery_labels[ranks[:, :k]]

            aps = Parallel(n_jobs=num_cores)(
                delayed(compute_avg_prec)(label[sketch_idx].item(), retrieval.cpu().numpy()) for sketch_idx, retrieval
                in enumerate(retrievals))
            aps_all.extend(aps)

        return np.mean(aps_all)
