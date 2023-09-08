from itertools import chain

import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss

import utils.initializers as I
from cycles.encoding.proxy_encoding import epoch_encoders
from cycles.generation.unified_classifier import epoch_generators
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
    c_unified, c_modality = I.init_classifiers(args, device, unified=True)
    g_photo, g_sketch = I.init_generators(args, device)
    e_photo, e_sketch = I.init_encoders(args, device)

    # Contrastive Criteria
    l_proxy = ProxyAnchorLoss(num_classes=args.num_classes, embedding_size=512).to(device)
    l_proxy.proxies = c_unified.fc.weight

    # Generator Optimizer
    optimizer_g = torch.optim.SGD(chain(g_photo.parameters(), g_sketch.parameters()), momentum=0.9, lr=args.lr_G)

    # Encoder Optimizer
    optimizer_e = torch.optim.SGD(chain(e_photo.parameters(), e_sketch.parameters()), momentum=0.9, lr=args.lr_E)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_e, T_max=args.n_epochs, eta_min=0, last_epoch=-1)

    start_epoch = I.auto_resume(args, g_photo, g_sketch, e_photo, e_sketch, optimizer_g, optimizer_e, scheduler)

    for epoch in range(start_epoch, args.n_epochs):
        gen_photo, gen_sketch, pred_photo, pred_sketch = epoch_generators(
            args, g_photo, g_sketch, c_unified, c_modality, e_photo, e_sketch, optimizer_g, epoch, n_iters=args.gen_iters)
        epoch_encoders(e_photo, e_sketch, gen_photo, gen_sketch, pred_photo, pred_sketch, optimizer_e, scheduler, l_proxy, epoch)

        if epoch % args.eval_every == 0:
            evaluate(test_loader_photo, test_loader_sketch, e_photo, e_sketch)
            save(args, epoch, g_photo, g_sketch, e_photo, e_sketch, optimizer_g, optimizer_e, scheduler)


if __name__ == '__main__':
    main()
