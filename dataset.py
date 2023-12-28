from torch.utils.data import Dataset
from torchvision import datasets, transforms


# A dataset class that allows variable resolutions for target reconstruction images
# Can be used for progressive training with increasing resolutions
class ReconstructionDataset(Dataset):

    def __init__(self):
        self.mnist = datasets.MNIST(root = "./data", transform=transforms.ToTensor(), download=True)
        self.set_target_resolution(28)

    def set_target_resolution(self, resolution):
        self.target_resolution = resolution
        self.transforms = transforms.Compose([
            transforms.Resize((self.target_resolution, self.target_resolution)),
            transforms.ToTensor()
        ])
        self.dataset = datasets.MNIST(root = "./data", transform=self.transforms, download=True)

    def __getitem__(self, index):
        return self.mnist[index][0], self.dataset[index][0]
    
    def __len__(self):
        return len(self.mnist)