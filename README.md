# IM-NET PyTorch
A PyTorch implementation of "Learning Implicit Fields for Generative Shape Modeling" by Zhiqin Chen and Hao Zhang

- [Paper](https://arxiv.org/abs/1812.02822) </br>
- [MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

# Demo
## Interpolation between digits

The implicit network learns shape boundaries rather than pixel distributions, so interpolation between digits looks like one digit morphing into another. In a regular autoencoder, interpolation would look like the first digit fading out and the second digit fading in.

<img src="/outputs/output.gif" width="250" height="250"/> <nobr>
<img src="/outputs/output2.gif" width="250" height="250"/> <nobr>
<img src="/outputs/output3.gif" width="250" height="250"/>

## Super-resolution + interpolation

We can also sample outputs at a higher resolution than the training data. Here is an MNIST interpolation at 128x128 pixels instead of the regular 28x28. This looks significantly less pixelated than the above interpolations.

<img src="/outputs/superres_output.gif" width="250" height="250"/>

