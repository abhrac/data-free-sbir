import torch.nn.functional as F


def kld(y1, y2):
    p = F.log_softmax(y1, dim=1)
    q = F.softmax(y2, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) / y1.shape[0]
    return l_kl
