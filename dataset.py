import glob
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# saranga: dataset class for the BBBBC021 dataset
class BBBCDataset(Dataset):
    def __init__(
        self,
        path,
        img_size=128,
        split=None,
        as_tensor: bool = True,
        do_augment: bool = True,
        do_normalize: bool = True,
    ):
        self.path = path
        # Store image paths
        if split:
            data_path = glob.glob(os.path.join(self.path, split, "*/*"))
        else:
            data_path = glob.glob(os.path.join(self.path, "*/*/*"))

        self.data = [path for path in data_path]

        # Image transformation
        transform = [
            transforms.Resize((img_size, img_size)),
        ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index < len(self.data)
        img_path = self.data[index]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return {"img": img, "index": index}


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg"],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [
                p.relative_to(folder)
                for ext in exts
                for p in Path(f"{folder}").glob(f"**/*.{ext}")
            ]
        else:
            self.paths = [
                p.relative_to(folder)
                for ext in exts
                for p in Path(f"{folder}").glob(f"*.{ext}")
            ]
        if sort_names:
            self.paths = sorted(self.paths)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"img": img, "index": index}


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


if __name__ == "__main__":
    root = "/projects/deepdevpath/Anis/diffusion-comparison-experiments/datasets/bbbc021_simple"
    dataset = BBBCDataset(root)
    print(len(dataset))
    print(dataset[231]["img"].shape)
