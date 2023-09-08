import torch
import torch.nn.functional as F
import torchvision.transforms as T


def epoch_encoders(e_photo, e_sketch, gen_photos, gen_sketches, optimizer_e, scheduler, contrast_sp, contrast_ps, epoch):
    # Training Encoders
    e_photo.train(), e_sketch.train()
    loss_ce = torch.nn.CrossEntropyLoss()

    # Normalize with mini-batch statistics
    gen_photos = T.Normalize(gen_photos.view(3, -1).mean(dim=1), gen_photos.view(3, -1).std(dim=1))(gen_photos)
    gen_sketches = T.Normalize(gen_sketches.view(3, -1).mean(dim=1), gen_sketches.view(3, -1).std(dim=1))(gen_sketches)
    # Normalize with ImageNet statistics
    imagenet_normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    gen_photos, gen_sketches = imagenet_normalize(gen_photos), imagenet_normalize(gen_sketches)

    optimizer_e.zero_grad()
    photo_repr = F.normalize(e_photo(gen_photos)[:, :, 0, 0])
    sketch_repr = F.normalize(e_sketch(gen_sketches)[:, :, 0, 0])

    loss_encoders = loss_ce(*contrast_ps(photo_repr, sketch_repr.detach())) + loss_ce(
        *contrast_sp(sketch_repr, photo_repr.detach()))
    loss_encoders.backward()
    optimizer_e.step()
    scheduler.step()

    loss_encoders += loss_encoders.item()
    print(f"Epoch: {epoch}, Encoder loss: {loss_encoders}")
