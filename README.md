# IM-NET PyTorch
A PyTorch implementation of "Learning Implicit Fields for Generative Shape Modeling" by Zhiqin Chen and Hao Zhang

- [Paper](https://arxiv.org/abs/1812.02822) </br>
- [MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

# Demo
## Interpolation between digits

The implicit network learns shape boundaries rather than pixel distributions, so interpolation between digits looks like one digit morphing into another. In a regular autoencoder, interpolation would look like the first digit fading out and the second digit fading in.

https://github.com/gursi26/IM-NET-pytorch/assets/75204369/08b69099-c4b7-4e46-9785-e6297ff93683

https://github.com/gursi26/IM-NET-pytorch/assets/75204369/e7ada431-f129-4412-bb9f-b075e981fe6f

https://github.com/gursi26/IM-NET-pytorch/assets/75204369/dd19d35d-853b-4cd4-8871-dccfd9b62603

## Super-resolution + interpolation

