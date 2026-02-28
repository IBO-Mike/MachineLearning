import torchvision
import torchvision.transforms as transforms
import os

def download_cifar10(data_dir):
    transform = transforms.ToTensor()
    torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

if __name__ == "__main__":
    download_cifar10("../../data/raw")