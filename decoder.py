from torch import nn 
import torch.nn.functional as F
import torch

# Fully connected implicit decoder
class IMDecoder(nn.Module):

    def __init__(self, z_dim):
        super(IMDecoder, self).__init__()
        z_dim_plus_coord = z_dim + 2
        self.layer1 = nn.Linear(z_dim_plus_coord, 512)
        self.layer2 = nn.Linear(512 + z_dim_plus_coord, 128)
        self.layer3 = nn.Linear(128 + z_dim_plus_coord, 64)
        self.layer4 = nn.Linear(64 + z_dim_plus_coord, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, features, coords):
        # Features has shape [batch_size, z_dim]
        # coord has shape [batch_size, num_coords, 2]
        features = features.unsqueeze(1).repeat(1, coords.shape[1], 1)
        x = torch.cat([features, coords], dim = -1)

        output = F.leaky_relu(self.layer1(x), negative_slope=0.02)
        output = torch.cat([output, x], dim=-1)

        output = F.leaky_relu(self.layer2(output), negative_slope=0.02)
        output = torch.cat([output, x], dim=-1)

        output = F.leaky_relu(self.layer3(output), negative_slope=0.02)
        output = torch.cat([output, x], dim=-1)

        output = F.leaky_relu(self.layer4(output), negative_slope=0.02)
        
        return F.sigmoid(self.output_layer(output).squeeze(-1))