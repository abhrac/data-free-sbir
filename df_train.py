from itertools import chain

import torch

import utils.initializers as I
from cycles.encoding.pairwise_encoding import epoch_encoders
from cycles.generation.independent_classifiers import epoch_generators
from losses.moco import MoCo
from utils import dataloaders
from utils.misc import evaluate, save, seeder, select_device
from utils.options import Options


def main():
    args = Options().parse()
    seeder(args)
    device = select_device(args)

    # Test set
    test_loader_photo, test_loader_sketch = dataloaders.test_loaders_sbir(args)

    # Models
    c_photo, c_sketch, c_modality = I.init_classifiers(args, device)
    g_photo, g_sketch = I.init_generators(args, device)
    e_photo, e_sketch = I.init_encoders(args, device)

    # Contrastive Criteria
    contrast_ps = MoCo(dim=512, K=128).to(device)
    contrast_sp = MoCo(dim=512, K=128).to(device)

    # Generator Optimizer
    optimizer_g = torch.optim.SGD(chain(g_photo.parameters(), g_sketch.parameters()), momentum=0.9, lr=args.lr_G)

    # Encoder Optimizer
    optimizer_e = torch.optim.SGD(chain(e_photo.parameters(), e_sketch.parameters()), momentum=0.9, lr=args.lr_E)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_e, T_max=args.n_epochs, eta_min=0, last_epoch=-1)

    start_epoch = I.auto_resume(args, g_photo, g_sketch, e_photo, e_sketch, optimizer_g, optimizer_e, scheduler)

    for epoch in range(start_epoch, args.n_epochs):
        gen_photos, gen_sketches = epoch_generators(args, g_photo, g_sketch, c_photo, c_sketch, c_modality, optimizer_g, epoch, n_iters=args.gen_iters)
        epoch_encoders(e_photo, e_sketch, gen_photos, gen_sketches, optimizer_e, scheduler, contrast_sp, contrast_ps, epoch)

        if epoch % args.eval_every == 0:
            evaluate(test_loader_photo, test_loader_sketch, e_photo, e_sketch)
            save(args, epoch, g_photo, g_sketch, e_photo, e_sketch, optimizer_g, optimizer_e, scheduler)


if __name__ == '__main__':
    main()
