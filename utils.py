import torch
from encoder import Encoder
from decoder import IMDecoder
from torch import nn

def create_coord_map(resolution):
    # creates a grid where each element is the coordinate of the point
    # coordinates are normalized to be between 0 and 1
    # returns a flattened form of the grid that can be used for model training
    x_coords = torch.arange(0, 1, step=1/resolution).repeat(resolution).unsqueeze(-1)
    y_coords = torch.arange(0, 1, step=1/resolution).unsqueeze(-1).repeat(1, resolution).view(-1, 1)
    coords = torch.cat([x_coords, y_coords], dim=-1)
    return coords

def initialize_enc_dec(z_dim, dev):
    enc = Encoder(z_dim)
    dec = IMDecoder(z_dim)
    for p in enc.parameters():
        nn.init.xavier_uniform_(p)

    for p in dec.named_parameters():
        if 'weight' in p[0]:
            nn.init.normal_(p[1], mean=0.0, std=0.02)
        elif 'bias' in p[0]:
            nn.init.constant_(p[1], 0.0)
    return enc.to(dev), dec.to(dev)