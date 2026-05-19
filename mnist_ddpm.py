import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- config ---
T = 100
BETA_START, BETA_END = 1e-4, 0.02  # DDPM paper values for T_ref=1000
T_REF = 1000  # reference steps for the schedule above
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def make_schedule(T_steps: int, device: torch.device):
    # Paper betas assume T=1000. For smaller T, scale up so x_T ~ N(0,I):
    #   x_T = sqrt(ab_T) x_0 + sqrt(1-ab_T) eps  =>  want ab_T ~ 0, 1-ab_T ~ 1
    scale = T_REF / T_steps
    betas = torch.linspace(BETA_START, BETA_END, T_steps, device=device) * scale
    betas = betas.clamp(1e-5, 0.999)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    ab_T = alpha_bar[-1].item()
    print(
        f"schedule: T={T_steps}, beta in [{betas[0]:.2e}, {betas[-1]:.4f}], "
        f"alpha_bar_T={ab_T:.2e}, sqrt(ab_T)={ab_T**0.5:.2e}, noise_var={1-ab_T:.4f}"
    )
    return betas, alphas, alpha_bar


def gather(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Pick a[t] per batch item and reshape for broadcasting over x."""
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


# --- data: MNIST in [-1, 1] ---
def get_dataloader():
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
        ]
    )
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


# --- small time-conditioned CNN ---
class EpsCNN(nn.Module):
    def __init__(self, n_steps: int):
        super().__init__()
        self.t_emb = nn.Embedding(n_steps, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_map = self.t_emb(t).view(-1, 1, 28, 28)
        return self.net(torch.cat([x, t_map], dim=1))


# --- forward: x_t = sqrt(a_bar) x_0 + sqrt(1-a_bar) eps ---
def forward_diffuse(x0, t, noise, alpha_bar):
    a = gather(alpha_bar, t, x0.shape)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


# --- reverse one step (DDPM) ---
@torch.no_grad()
def reverse_step(x, t, model, betas, alphas, alpha_bar):
    eps = model(x, t)
    a = gather(alphas, t, x.shape)
    ab = gather(alpha_bar, t, x.shape)
    b = gather(betas, t, x.shape)
    mean = (1.0 / torch.sqrt(a)) * (x - (b / torch.sqrt(1.0 - ab)) * eps)
    if (t == 0).all():
        return mean
    noise = torch.randn_like(x)
    return mean + torch.sqrt(b) * noise


@torch.no_grad()
def sample(model, betas, alphas, alpha_bar, n=16):
    x = torch.randn(n, 1, 28, 28, device=DEVICE)
    for ti in range(T - 1, -1, -1):
        t = torch.full((n,), ti, device=DEVICE, dtype=torch.long)
        x = reverse_step(x, t, model, betas, alphas, alpha_bar)
    return x


def to_img(x):
    """[-1,1] -> (H, W) numpy for imshow."""
    arr = ((x.clamp(-1, 1) + 1) * 0.5).detach().cpu().numpy()
    return arr.squeeze()  # (1,1,28,28) or (1,28,28) -> (28,28)


def plot_grid(tensors, title, path, nrow=8):
    n = len(tensors)
    ncol = min(nrow, n)
    nrow_g = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_g, ncol, figsize=(ncol * 1.2, nrow_g * 1.2))
    axes = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(to_img(tensors[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forward_grid(x0_one, alpha_bar, path):
    """One image -> progressively noised x_t."""
    steps = [0, 10, 20, 30, 50, 70, 99]
    imgs = []
    for ti in steps:
        t = torch.tensor([ti], device=DEVICE)
        if ti == 0:
            imgs.append(x0_one)
        else:
            eps = torch.randn_like(x0_one)
            imgs.append(forward_diffuse(x0_one, t, eps, alpha_bar))
    plot_grid(imgs, "Forward noising", path, nrow=8)


@torch.no_grad()
def plot_reverse_trajectory(model, betas, alphas, alpha_bar, path):
    """Pure noise -> denoise snapshots."""
    x = torch.randn(1, 1, 28, 28, device=DEVICE)
    snaps = [x.clone()]
    show_at = {99, 80, 60, 40, 20, 10, 5, 0}
    for ti in range(T - 1, -1, -1):
        t = torch.tensor([ti], device=DEVICE, dtype=torch.long)
        x = reverse_step(x, t, model, betas, alphas, alpha_bar)
        if ti in show_at:
            snaps.append(x.clone())
    snaps = list(reversed(snaps))  # noisy -> clean
    plot_grid(snaps, "Reverse denoising", path, nrow=8)


def train():
    betas, alphas, alpha_bar = make_schedule(T, torch.device(DEVICE))
    loader = get_dataloader()
    model = EpsCNN(T).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for x0, _ in loader:
            x0 = x0.to(DEVICE)
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
            eps = torch.randn_like(x0)
            xt = forward_diffuse(x0, t, eps, alpha_bar)
            pred = model(xt, t)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        losses.append(ep_loss)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={ep_loss:.4f}")

    # loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "loss_curve.png", dpi=120)
    plt.close()

    torch.save(model.state_dict(), OUT / "eps_cnn.pt")
    return model, betas, alphas, alpha_bar, loader


def main():
    print(f"device={DEVICE}, T={T}")
    model, betas, alphas, alpha_bar, loader = train()

    model.eval()
    x0, _ = next(iter(loader))
    x0 = x0[:1].to(DEVICE)
    plot_forward_grid(x0, alpha_bar, OUT / "forward_noise.png")
    plot_reverse_trajectory(model, betas, alphas, alpha_bar, OUT / "reverse_denoise.png")

    gens = sample(model, betas, alphas, alpha_bar, n=16)
    plot_grid([gens[i] for i in range(16)], "Generated digits", OUT / "generated_grid.png", nrow=4)

    print(f"saved to {OUT.resolve()}/")


if __name__ == "__main__":
    main()
