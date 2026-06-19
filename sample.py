import torch
from torchvision import datasets

from meanflow.model import MeanFlowUNet
from meanflow.utils import plot_original_vs_generated
from meanflow.data import load_imagenette, make_three_image_dataset, make_dataloader


@torch.no_grad()
def generate_meanflow_samples(model, sample_shape, device, y=None):
    model.eval()

    z = torch.randn(sample_shape, device=device)

    B = sample_shape[0]
    r = torch.zeros(B, device=device)
    t = torch.ones(B, device=device)

    u = model(z, r, t, y)

    x_gen = z - u

    return x_gen.clamp(-1, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    image_size = 256

    model = MeanFlowUNet(
        img_channels=3,
        base_channels=64,
        time_emb_dim=128,
        num_classes=3,
    ).to(device)

    model.load_state_dict(
        torch.load("checkpoints/meanflow_3_image_overfit.pt", map_location=device)
    )

    train_data, transform = load_imagenette(
        image_size=image_size,
    )

    full_dataset = datasets.ImageFolder(root=train_data, transform=transform)

    three_image_dataset, _, _ = make_three_image_dataset(full_dataset)

    dataloader = make_dataloader(
        three_image_dataset,
        batch_size=3,
        shuffle=False,
    )

    x_orig, y_orig = next(iter(dataloader))
    x_orig = x_orig.to(device)
    y_orig = y_orig.to(device)

    assert x_orig.shape[-2:] == (256, 256)

    x_gen = generate_meanflow_samples(
        model=model,
        sample_shape=x_orig.shape,
        device=device,
        y=y_orig,
    )

    print("Original shape:", x_orig.shape)
    print("Generated shape:", x_gen.shape)

    assert x_gen.shape == x_orig.shape

    plot_original_vs_generated(
        x_orig,
        x_gen,
        save_path="outputs/sample_original_vs_generated.png",
    )


if __name__ == "__main__":
    main()