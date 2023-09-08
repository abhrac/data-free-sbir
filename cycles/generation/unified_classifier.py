import torch
from tqdm import tqdm

from losses.distributions import kld


def epoch_generators(args, g_photo, g_sketch, c_unified, c_modality, e_photo, e_sketch, optimizer_g, epoch, n_iters):
    fe = torch.nn.Sequential(*list(c_unified.children())[:-1])
    proxies = c_unified.fc
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    for i in tqdm(range(n_iters)):
        z = torch.randn(args.batch_size, args.latent_dim, 1, 1).cuda()
        optimizer_g.zero_grad()
        gen_photo = g_photo(z)
        gen_sketch = g_sketch(z)

        # Photo Classifier
        outputs_photo_classifier = c_unified(gen_photo)
        features_T = fe(gen_photo).flatten(1)
        pred_photo = outputs_photo_classifier.data.max(1)[1]

        loss_activation = -features_T.abs().mean()
        loss_one_hot = loss_ce(outputs_photo_classifier, pred_photo)
        softmax_o_T = torch.nn.functional.softmax(outputs_photo_classifier, dim=1).mean(dim=0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss_modality = loss_bce(c_modality(gen_photo), torch.zeros(args.batch_size, 1).to(gen_photo.device))

        # Sketch Classifier
        outputs_sketch_classifier = c_unified(gen_sketch)
        features_T = fe(gen_sketch).flatten(1)
        pred_sketch = outputs_sketch_classifier.data.max(1)[1]

        loss_activation = loss_activation - features_T.abs().mean()
        loss_one_hot = loss_one_hot + loss_ce(outputs_sketch_classifier, pred_sketch)
        softmax_o_T = torch.nn.functional.softmax(outputs_sketch_classifier, dim=1).mean(dim=0)
        loss_information_entropy = loss_information_entropy + (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a
        loss_modality += loss_bce(c_modality(gen_sketch), torch.ones(args.batch_size, 1).to(gen_sketch.device))
        loss += loss_modality

        # Ensure same class generation given same noise vector
        loss_kld = kld(outputs_photo_classifier, outputs_sketch_classifier.detach())
        loss += loss_kld

        # Adversarial criterion - Generate images for which the encoders (students) diverge most from the classifiers (teachers)
        outputs_photo_encoder = proxies(e_photo(gen_photo).flatten(1))
        outputs_sketch_encoder = proxies(e_sketch(gen_sketch).flatten(1))
        loss_adv = - loss_ce(outputs_photo_encoder, pred_photo) - loss_ce(outputs_sketch_encoder, pred_sketch)
        loss += loss_adv

        loss.backward()
        optimizer_g.step()
        if i == 1:
            print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_modality: %f] [loss_kld: %f] [loss_adv: %f]" % (
                epoch, args.n_epochs, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(),
                loss_modality.item(), loss_kld.item(), loss_adv.item()))

    # The first few serve as warmup iterations. Returning results from only the last.
    return gen_photo.detach(), gen_sketch.detach(), pred_photo, pred_sketch
