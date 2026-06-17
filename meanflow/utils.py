import torch
import matplotlib.pyplot as plt


def denormalize_img(x):
    x = (x + 1) / 2
    return x.clamp(0, 1)


def plot_loss(loss_history, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch / Step")
    plt.ylabel("Loss")
    plt.title("MeanFlow Training Loss")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_original_vs_generated(originals, generated, save_path=None):
    num_samples = originals.shape[0]

    fig, axes = plt.subplots(num_samples, 2, figsize=(5, 2.5 * num_samples))

    for i in range(num_samples):
        orig_img = denormalize_img(originals[i].detach().cpu()).permute(1, 2, 0)
        gen_img = denormalize_img(generated[i].detach().cpu()).permute(1, 2, 0)

        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gen_img)
        axes[i, 1].set_title(f"Generated {i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()