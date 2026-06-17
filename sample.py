import torch

from meanflow.model import MeanFlowUNet
from meanflow.utils import plot_original_vs_generated
from meanflow.data import load_imagenette, make_three_image_dataset, make_dataloader


@torch.no_grad()
def generate_meanflow_samples(model, num_samples, device, y=None):
    model.eval()

    z = torch.randn(num_samples, 3, 32, 32, device=device)

    r = torch.zeros(num_samples, device=device)
    t = torch.ones(num_samples, device=device)

    u = model(z, r, t, y)

    x_gen = z - u

    return x_gen


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MeanFlowUNet(
        img_channels=3,
        base_channels=64,
        time_emb_dim=64,
        num_classes=3,
    ).to(device)

    model.load_state_dict(
        torch.load("checkpoints/meanflow_3_image_overfit.pt", map_location=device)
    )

    full_dataset = load_imagenette(
        root="./data/imagenette2-160/train",
        image_size=32,
    )

    three_image_dataset, _, _ = make_three_image_dataset(full_dataset)

    dataloader = make_dataloader(
        three_image_dataset,
        batch_size=3,
        shuffle=False,
    )

    x_orig, y_orig = next(iter(dataloader))
    x_orig = x_orig.to(device)
    y_orig = y_orig.to(device)

    x_gen = generate_meanflow_samples(
        model=model,
        num_samples=x_orig.shape[0],
        device=device,
        y=y_orig,
    )

    plot_original_vs_generated(
        x_orig,
        x_gen,
        save_path="outputs/sample_original_vs_generated.png",
    )


if __name__ == "__main__":
    main()