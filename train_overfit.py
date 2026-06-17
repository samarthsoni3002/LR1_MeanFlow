import os
import torch
from tqdm.auto import tqdm

from meanflow.data import load_imagenette, make_three_image_dataset, make_dataloader
from meanflow.model import MeanFlowUNet
from meanflow.loss import meanflow_loss
from meanflow.utils import plot_loss, plot_original_vs_generated

from torchvision import datasets, transforms


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
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = 32
    train_data, transform = load_imagenette(
        image_size=image_size,
    ) 
    
    full_dataset = datasets.ImageFolder(root=train_data, transform=transform)

    three_image_dataset, selected_images, selected_labels = make_three_image_dataset(full_dataset)

    dataloader = make_dataloader(
        three_image_dataset,
        batch_size=3,
        shuffle=True,
    )

    model = MeanFlowUNet(
        img_channels=3,
        base_channels=64,
        time_emb_dim=64,
        num_classes=3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.0,
    )

    num_steps = 5000
    loss_history = []

    model.train()
    pbar = tqdm(range(num_steps))

    for step in pbar:
        x, y = next(iter(dataloader))
        x = x.to(device)
        y = y.to(device)

        loss = meanflow_loss(model, x, y, same_time_prob=0.75)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if step % 50 == 0:
            pbar.set_description(f"loss={loss.item():.6f}")

    torch.save(model.state_dict(), "checkpoints/meanflow_3_image_overfit.pt")

    plot_loss(
        loss_history,
        save_path="outputs/loss_curve_3_image_overfit.png",
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
        save_path="outputs/original_vs_generated_3_image_overfit.png",
    )


if __name__ == "__main__":
    main()