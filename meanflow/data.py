import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import urllib 
import tarfile 
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.images[idx]
        return self.images[idx], self.labels[idx]


def get_transform(image_size=256):
    return transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def load_imagenette(image_size=256):
    
    data_dir = Path("./")
    data_dir.mkdir(parents=True,exist_ok=True)
    
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tar_path = data_dir / "imagenette2-320.tgz"
    extract_path = data_dir /"imagenette2-320"

    if not extract_path.exists():
        print("Downloading ImageNette")
        
        urllib.request.urlretrieve(url, tar_path)

        print("Extracting")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

    train_data = extract_path / "train"
    transform = get_transform(image_size)

    return train_data, transform


def make_three_image_dataset(full_dataset):
    selected_images = []
    selected_labels = []
    seen_classes = set()

    for img, label in full_dataset:
        if label not in seen_classes:
            selected_images.append(img)
            selected_labels.append(len(selected_labels))  # remap to 0,1,2
            seen_classes.add(label)

        if len(selected_images) == 3:
            break

    selected_images = torch.stack(selected_images, dim=0)
    selected_labels = torch.tensor(selected_labels, dtype=torch.long)

    dataset = ImageDataset(selected_images, selected_labels)

    return dataset, selected_images, selected_labels


def make_dataloader(dataset, batch_size=3, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )