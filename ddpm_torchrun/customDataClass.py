
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

class CosmosImageData(Dataset):
    """
    A custom dataset class for loading and transforming images from a specified directory.

    This class is designed to load images from a given directory, apply any specified transformations,
    and provide an interface compatible with PyTorch's data loading utilities.

    Attributes:
        path (str): The directory containing image files.
        paths (List[Path]): A sorted list of Path objects pointing to image files.
        transform (Optional[Callable]): A callable transformation to apply to each image.
    """

    def __init__(self, path: str, transform: Optional[Callable] = None):
        """
        Initializes the dataset with the path to the images and an optional transform.

        Args:
            path (str): The directory containing image files.
            transform (Optional[Callable]): A transformation to apply to each image, such as resizing or normalization.
        """
        self.path = path
        self.paths = sorted(Path(self.path).glob('*.png'))
        self.transform = transform
        
    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Image.Image:
        """
        Retrieves an image from the dataset and applies the specified transformations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Image.Image: The transformed image.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        img = Image.open(self.paths[idx])
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img
