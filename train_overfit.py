import os
import random

import torch
from tqdm.auto import tqdm
from torchvision import datasets

from meanflow.data import load_imagenette, make_three_image_dataset, make_dataloader
from meanflow.model import MeanFlowUNet
from meanflow.loss import meanflow_loss
from meanflow.utils import plot_loss, plot_original_vs_generated


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
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    image_size = 256

    train_data, transform = load_imagenette(
        image_size=image_size,
    )

    full_dataset = datasets.ImageFolder(root=train_data, transform=transform)

    three_image_dataset, selected_images, selected_labels = make_three_image_dataset(full_dataset)

    print("Selected image tensor shape:", selected_images.shape)
    print("Selected labels:", selected_labels.tolist())

    assert selected_images.shape == (3, 3, 256, 256), (
        f"Expected selected_images shape [3, 3, 256, 256], got {selected_images.shape}"
    )

    dataloader = make_dataloader(
        three_image_dataset,
        batch_size=3,
        shuffle=True,
    )

    model = MeanFlowUNet(
        img_channels=3,
        base_channels=64,
        time_emb_dim=128,
        num_classes=3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.0,
    )

    num_steps = 20000
    loss_history = []

    model.train()
    pbar = tqdm(range(num_steps))

    for step in pbar:
        x, y = next(iter(dataloader))
        x = x.to(device)
        y = y.to(device)

        loss, mf_loss = meanflow_loss(
            model,
            x,
            y,
            same_time_prob=0.4,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            pbar.set_description(
                f"step={step} loss={loss.item():.6f} mf={mf_loss:.6f} grad={float(grad_norm):.4f}"
            )

        if step % 500 == 0:
            print(
                f"Step {step:05d} | "
                f"Loss: {loss.item():.6f} | "
                f"MF: {mf_loss:.6f} | "
                f"Grad norm: {float(grad_norm):.4f}"
            )

    torch.save(model.state_dict(), "checkpoints/meanflow_3_image_overfit.pt")

    plot_loss(
        loss_history,
        save_path="outputs/loss_curve_3_image_overfit.png",
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
        save_path="outputs/original_vs_generated_3_image_overfit.png",
    )


if __name__ == "__main__":
    main()