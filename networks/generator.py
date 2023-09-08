import torch.nn as nn


class Generator(nn.Module):
    '''
        Args:
            nz: Input noise dimension
            nd: Output image dimension
            nc: Number of channels in output image
    '''
    def __init__(self, nz, nd, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, nd * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nd * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nd * 8, nd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nd * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(nd * 4, nd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nd * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(nd * 2, nd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nd),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(nd, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
