from torch import nn

# Simple convolutional encoder
# Extracts a z_dim-dimensional feature vector from each input image
class Encoder(nn.Module):

    def __init__(self, z_dim, in_channels=1):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(4),
            nn.LeakyReLU(0.02),

            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.02),

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.02),

            nn.Conv2d(16, z_dim, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        return self.layers(x).view(x.shape[0], -1)