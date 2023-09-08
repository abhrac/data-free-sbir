import torch.nn.functional as F
import torchvision.transforms as T


def epoch_encoders(e_photo, e_sketch, gen_photo, gen_sketch, pred_photo, pred_sketch, optimizer_e, scheduler, l_proxy, epoch):
    # Training Encoders
    e_photo.train(), e_sketch.train()

    # Normalize with mini-batch statistics
    gen_photo = T.Normalize(gen_photo.view(3, -1).mean(dim=1), gen_photo.view(3, -1).std(dim=1))(gen_photo)
    gen_sketch = T.Normalize(gen_sketch.view(3, -1).mean(dim=1), gen_sketch.view(3, -1).std(dim=1))(gen_sketch)
    # Normalize with ImageNet statistics
    imagenet_normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    gen_photo, gen_sketch = imagenet_normalize(gen_photo), imagenet_normalize(gen_sketch)

    optimizer_e.zero_grad()
    photo_repr = F.normalize(e_photo(gen_photo)[:, :, 0, 0])
    sketch_repr = F.normalize(e_sketch(gen_sketch)[:, :, 0, 0])

    loss_encoders = l_proxy(photo_repr, pred_photo) + l_proxy(sketch_repr, pred_sketch)
    loss_encoders.backward()
    optimizer_e.step()
    scheduler.step()

    loss_encoders += loss_encoders.item()
    print(f"Epoch: {epoch}, Encoder loss: {loss_encoders}")
