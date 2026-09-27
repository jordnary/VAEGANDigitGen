"""Reproduce two-dimensional MNIST VAE-GAN training and evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "results"


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU())
        self.mu = nn.Linear(128, 2)
        self.logvar = nn.Linear(128, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    @staticmethod
    def reparameterize(mu, logvar, eps):
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x, eps):
        mu, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mu, logvar, eps)), mu, logvar


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.Flatten(), spectral_norm(nn.Linear(64 * 7 * 7, 128)),
            nn.LeakyReLU(0.2), spectral_norm(nn.Linear(128, 1)))

    def forward(self, x):
        return self.net(x.reshape(-1, 1, 28, 28)).flatten()


class Classifier(nn.Module):
    """Independent evaluator; never supplies gradients to the generator."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 64), nn.ReLU())
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        features = self.conv(x.reshape(-1, 1, 28, 28))
        return self.head(features), features


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss_terms(pred, x, mu, logvar):
    l1 = F.l1_loss(pred, x, reduction="none").sum(1)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)
    return l1.mean(), kl.mean()


def d_loss(real, reconstruction, prior):
    return (0.5 * (real - 1).square().mean()
            + 0.25 * reconstruction.square().mean()
            + 0.25 * prior.square().mean())


def g_loss(reconstruction, prior):
    return 0.25 * ((reconstruction - 1).square().mean()
                   + (prior - 1).square().mean())


def load_data(device):
    train = MNIST(ROOT / "data", train=True, download=True)
    test = MNIST(ROOT / "data", train=False, download=True)
    x = train.data.flatten(1).to(device, torch.float32).div_(255)
    xt = test.data.flatten(1).to(device, torch.float32).div_(255)
    assert x.shape == (60000, 784) and xt.shape == (10000, 784)
    return x, xt, train.targets.to(device), test.targets.to(device)


def load_vae(name, device):
    checkpoint = torch.load(ART / name, map_location=device, weights_only=True)
    model = VAE().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def pretrain(x, args):
    """Standalone route matching the original VAE initialization and schedule."""
    seed_all(42)
    model = VAE().to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.pretrain_epochs, eta_min=5e-5)
    rows = []
    for epoch in range(1, args.pretrain_epochs + 1):
        total = torch.zeros(2, device=x.device)
        beta = min(1.0, epoch / 15)
        for ids in torch.randperm(len(x), device=x.device).split(args.batch_size):
            batch = x[ids]
            pred, mu, lv = model(batch, torch.randn(len(ids), 2, device=x.device))
            l1, kl = loss_terms(pred, batch, mu, lv)
            optimizer.zero_grad(set_to_none=True)
            (l1 + beta * kl).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            total += torch.stack((l1.detach(), kl.detach())) * len(ids)
        row = dict(epoch=epoch, l1=float(total[0] / len(x)),
                   kl=float(total[1] / len(x)), beta=beta)
        rows.append(row)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"Pretrain {epoch:03d}: L1={row['l1']:.4f}", flush=True)
    torch.save(dict(model=model.state_dict(), history=rows,
                    config=dict(epochs=args.pretrain_epochs, batch_size=args.batch_size,
                                lr=1e-3, warmup=15, seed=42)), ART / "vae_base.pt")


def train_variant(x, args, adversarial):
    seed_all(args.seed)
    model, _ = load_vae("vae_base.pt", x.device)
    disc = Discriminator().to(x.device)
    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs, eta_min=args.lr * 0.1) for opt in (opt_g, opt_d)]
    order_rng = torch.Generator(device=x.device).manual_seed(args.seed + 1)
    noise_rng = torch.Generator(device=x.device).manual_seed(args.seed + 2)
    prior_rng = torch.Generator(device=x.device).manual_seed(args.seed + 3)
    disc_rng = torch.Generator(device=x.device).manual_seed(args.seed + 4)
    name = args.run_name if adversarial else "vae_control"
    def noisy(images):
        noise = torch.randn(images.shape, device=x.device, generator=disc_rng)
        return images + args.instance_noise * noise
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        weight = args.adv_weight * min(1.0, epoch / args.adv_warmup)
        weight = weight if adversarial else 0.0
        total = torch.zeros(7, device=x.device)
        order = torch.randperm(len(x), device=x.device, generator=order_rng)
        for ids in order.split(args.batch_size):
            batch = x[ids]
            eps = torch.randn(len(ids), 2, device=x.device, generator=noise_rng)
            pred, mu, lv = model(batch, eps)
            l1, kl = loss_terms(pred, batch, mu, lv)
            zero = torch.zeros((), device=x.device)
            ld, adv, real_score, fake_score = zero, zero, zero, zero
            if adversarial:
                z = torch.randn(len(ids), 2, device=x.device, generator=prior_rng)
                prior = model.decoder(z)
                disc.train().requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                dr = disc(noisy(batch))
                dq, dp = disc(noisy(pred.detach())), disc(noisy(prior.detach()))
                ld = d_loss(dr, dq, dp)
                ld.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), 10)
                opt_d.step()
                opt_d.zero_grad(set_to_none=True)
                disc.eval().requires_grad_(False)
                adv = g_loss(disc(noisy(pred)), disc(noisy(prior)))
                real_score = dr.detach().mean()
                fake_score = 0.5 * (dq.detach().mean() + dp.detach().mean())
            objective = l1 + kl + weight * adv
            if not torch.isfinite(objective + ld):
                raise RuntimeError("Non-finite training objective")
            opt_g.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            opt_g.step()
            values = (l1, kl, ld, adv, objective, real_score, fake_score)
            total += torch.stack([v.detach() for v in values]) * len(ids)
        keys = ("l1", "kl", "d_loss", "g_adv", "objective", "d_real", "d_fake")
        row = dict(epoch=epoch, adv_weight=weight, lr=opt_g.param_groups[0]["lr"])
        row.update(zip(keys, (total / len(x)).tolist()))
        rows.append(row)
        schedulers[0].step()
        if adversarial:
            schedulers[1].step()
        write_csv(OUT / f"{name}_history.csv", rows)
        torch.save(dict(model=model.state_dict(), discriminator=disc.state_dict(),
                        config=vars(args), history=rows, epoch=epoch,
                        baseline_sha256=digest(ART / "vae_base.pt")),
                   ART / f"{name}.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"{name} {epoch:03d}: L1={row['l1']:.4f}, KL={row['kl']:.4f}, "
                  f"D={row['d_loss']:.4f}, Gadv={row['g_adv']:.4f}", flush=True)


def train_classifier(x, labels, args):
    seed_all(123)
    model = Classifier().to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    for epoch in range(1, args.classifier_epochs + 1):
        total = torch.zeros(2, device=x.device)
        for ids in torch.randperm(len(x), device=x.device).split(256):
            logits, _ = model(x[ids])
            loss = F.cross_entropy(logits, labels[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total[0] += loss.detach() * len(ids)
            total[1] += (logits.argmax(1) == labels[ids]).sum()
        rows.append(dict(epoch=epoch, loss=float(total[0] / len(x)),
                         accuracy=float(total[1] / len(x))))
        print(f"Classifier {epoch:02d}: accuracy={rows[-1]['accuracy']:.4f}",
              flush=True)
    torch.save(dict(model=model.state_dict(), history=rows), ART / "classifier.pt")
    write_csv(OUT / "classifier_history.csv", rows)


@torch.inference_mode()
def classify(model, x):
    probabilities, features = [], []
    for batch in x.split(512):
        logits, feature = model(batch)
        probabilities.append(logits.softmax(1).cpu())
        features.append(feature.cpu())
    return torch.cat(probabilities), torch.cat(features).numpy().astype(np.float64)


def feature_distance(a, b):
    """Frechet distance in this evaluator's 64-D features; not Inception FID."""
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(ca)
    sqrt_a = (eigenvectors * np.sqrt(eigenvalues.clip(0))) @ eigenvectors.T
    middle = sqrt_a @ cb @ sqrt_a
    cross = np.sqrt(np.linalg.eigvalsh((middle + middle.T) / 2).clip(0)).sum()
    value = np.square(a.mean(0) - b.mean(0)).sum() + np.trace(ca + cb) - 2 * cross
    return float(max(value, 0))


@torch.inference_mode()
def evaluate(model, test):
    rng = torch.Generator().manual_seed(43)
    draws = torch.zeros(10, device=test.device)
    mean_error = torch.zeros((), device=test.device)
    kl_total = torch.zeros_like(mean_error)
    mus = []
    for batch in test.split(512):
        mu, lv = model.encode(batch)
        mus.append(mu.cpu())
        deterministic = model.decoder(mu)
        mean_error += (deterministic - batch).abs().sum()
        kl_total += (-0.5 * (1 + lv - mu.square() - lv.exp()).sum())
        for draw in range(10):
            eps = torch.randn(mu.shape, generator=rng).to(test.device)
            pred = model.decoder(model.reparameterize(mu, lv, eps))
            draws[draw] += (pred - batch).abs().sum()
    draws = draws / len(test)
    metrics = dict(posterior_l1=float(draws.mean()),
                   posterior_l1_mc_se=float(draws.std() / math.sqrt(10)),
                   posterior_mae=float(draws.mean() / 784),
                   mean_mae=float(mean_error / len(test) / 784),
                   kl=float(kl_total / len(test)))
    return metrics, torch.cat(mus)


def make_grid():
    coordinates = torch.linspace(-3, 3, 21)
    yy, xx = torch.meshgrid(coordinates.flip(0), coordinates, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), 1)


def mosaic(images, rows, cols, gap=2):
    images = np.asarray(images).reshape(rows, cols, 28, 28)
    canvas = np.zeros((rows * (28 + gap) - gap, cols * (28 + gap) - gap))
    for row in range(rows):
        for col in range(cols):
            y, x = row * (28 + gap), col * (28 + gap)
            canvas[y:y + 28, x:x + 28] = images[row, col]
    return canvas


def save_figure(fig, name):
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def show_digits(axes, images, titles):
    for ax, pixels, title in zip(np.asarray(axes).flat, images, titles):
        ax.imshow(np.asarray(pixels).reshape(28, 28), cmap="gray_r",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


def make_figures(arrays, histories, test_y):
    grid = arrays["vaegan_grid"]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mosaic(grid, 21, 21), cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ticks = np.arange(0, 21, 2)
    ax.set_xticks(ticks * 30 + 13.5, [f"{v:.1f}" for v in np.linspace(-3, 3, 21)[ticks]])
    ax.set_yticks(ticks * 30 + 13.5, [f"{v:.1f}" for v in np.linspace(3, -3, 21)[ticks]])
    ax.set(xlabel="$z_1$", ylabel="$z_2$", title="VAE-GAN: 21 x 21 latent grid")
    save_figure(fig, "latent_grid.png")
    fig, axes = plt.subplots(2, 6, figsize=(11, 4.2))
    pictures = [arrays["first_five"], *arrays["vaegan_five"][:5],
                arrays["vaegan_five_mean"], *arrays["vaegan_five"][5:]]
    titles = ["Input: first 5", *[f"Sample {i}" for i in range(1, 6)],
              "Decode mean", *[f"Sample {i}" for i in range(6, 11)]]
    show_digits(axes, pictures, titles)
    fig.tight_layout()
    save_figure(fig, "five_samples.png")
    names = ["vae_base", "vae_control", "vaegan"]
    labels = ["Pretrained VAE", "VAE continuation", "VAE-GAN"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, name, label in zip(axes, names, labels):
        ax.imshow(mosaic(arrays[f"{name}_grid"], 21, 21), cmap="gray_r",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "grid_comparison.png")
    fig, axes = plt.subplots(4, 10, figsize=(11, 4.9))
    show_digits(axes[0], arrays["class_inputs"], [str(i) for i in range(10)])
    axes[0, 0].set_ylabel("Input", fontsize=9)
    for row, (name, label) in enumerate(zip(names, labels), 1):
        show_digits(axes[row], arrays[f"{name}_class_recon"], [""] * 10)
        axes[row, 0].set_ylabel(label.replace(" ", "\n"), fontsize=8)
    fig.tight_layout()
    save_figure(fig, "reconstructions.png")
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.5))
    for ax, name, label in zip(axes, names, labels):
        ax.imshow(mosaic(arrays[f"{name}_prior"][:32], 2, 16), cmap="gray_r",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, "prior_comparison.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key, title in zip(axes.flat[:2], ["l1", "kl"], ["L1 per image", "KL per image"]):
        for name, label in zip(names[1:], labels[1:]):
            rows = histories[name]
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
        ax.set_title(title)
        ax.legend(fontsize=8)
    rows = histories["vaegan"]
    for key, label in [("d_loss", "Discriminator"), ("g_adv", "Generator adversarial")]:
        axes[1, 0].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
    axes[1, 0].set_title("Unweighted adversarial losses")
    axes[1, 0].legend(fontsize=8)
    for key, label in [("d_real", "Real"), ("d_fake", "Fake mixture")]:
        axes[1, 1].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
    axes[1, 1].set_title("Discriminator scores")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Continuation epoch")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "training_curves.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    mu = arrays["vaegan_test_mu"]
    scatter = ax.scatter(mu[:, 0], mu[:, 1], c=test_y, cmap="tab10",
                         vmin=-0.5, vmax=9.5, s=3, alpha=0.55, rasterized=True)
    fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit label")
    ax.set(xlabel="$\\mu_1(x)$", ylabel="$\\mu_2(x)$", title="VAE-GAN posterior means")
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, "latent_scatter.png")


@torch.inference_mode()
def generate(x, test, labels, test_y):
    classifier = Classifier().to(x.device).eval()
    checkpoint = torch.load(ART / "classifier.pt", map_location=x.device, weights_only=True)
    classifier.load_state_dict(checkpoint["model"])
    real_prob, real_features = classify(classifier, test)
    accuracy = float((real_prob.argmax(1) == test_y.cpu()).float().mean())
    grid_z = make_grid()
    prior_z = torch.randn(10000, 2, generator=torch.Generator().manual_seed(2027))
    eps = torch.randn(10, 2, generator=torch.Generator().manual_seed(2026))
    index = int(torch.where(labels == 5)[0][0])
    original = x[index:index + 1]
    ids = [int(torch.where(test_y == label)[0][0]) for label in range(10)]
    arrays = dict(grid_z=grid_z.numpy(), prior_z=prior_z.numpy(), epsilon=eps.numpy(),
                  first_five=original.cpu().numpy(), class_inputs=test[ids].cpu().numpy())
    metrics, histories = {}, {}
    for name in ["vae_base", "vae_control", "vaegan_no_noise", "vaegan_strong", "vaegan"]:
        model, checkpoint = load_vae(f"{name}.pt", x.device)
        model.eval()
        result, test_mu = evaluate(model, test)
        result["config"] = checkpoint["config"]
        prior = torch.cat([model.decoder(z.to(x.device)) for z in prior_z.split(512)])
        probabilities, features = classify(classifier, prior)
        counts = torch.bincount(probabilities.argmax(1), minlength=10)
        real_counts = torch.bincount(test_y.cpu(), minlength=10)
        proportions = counts.double() / len(prior)
        target_proportions = real_counts.double() / len(test)
        confidence = probabilities.max(1).values
        result.update(feature_frechet=feature_distance(real_features, features),
                      prior_confidence=float(confidence.mean()),
                      prior_confident_fraction=float((confidence >= 0.9).float().mean()),
                      prior_class_counts=counts.tolist(),
                      prior_class_entropy=float(-(
                          proportions * proportions.clamp_min(1e-12).log()).sum()),
                      prior_class_tv=float((proportions - target_proportions).abs().sum() / 2))
        grid = model.decoder(grid_z.to(x.device))
        mu, lv = model.encode(original)
        z = model.reparameterize(mu, lv, eps.to(x.device))
        five = model.decoder(z)
        five_prob, _ = classify(classifier, five)
        result["first_five"] = dict(index=index, split="train", noise_seed=2026,
            mu=mu[0].cpu().tolist(), logvar=lv[0].cpu().tolist(),
            std=(0.5 * lv[0]).exp().cpu().tolist(),
            predictions=five_prob.argmax(1).tolist(),
            class5_probabilities=five_prob[:, 5].tolist(),
            pairwise_mae=float(torch.pdist(five, p=1).mean() / 784),
            input_mae=float((five - original).abs().mean()))
        mean_recons = torch.cat([model.decoder(m.to(x.device)) for m in test_mu.split(512)])
        recon_prob, _ = classify(classifier, mean_recons)
        result["reconstruction_label_accuracy"] = float(
            (recon_prob.argmax(1) == test_y.cpu()).float().mean())
        for suffix, value in dict(grid=grid, prior=prior[:32], five=five, five_z=z,
                five_mu=mu, five_logvar=lv, five_mean=model.decoder(mu), test_mu=test_mu,
                class_recon=mean_recons[ids]).items():
            arrays[f"{name}_{suffix}"] = value.cpu().numpy()
        histories[name] = checkpoint["history"]
        write_csv(OUT / f"{name}_history.csv", checkpoint["history"])
        metrics[name] = result
        print(f"{name}: MAE={result['posterior_mae']:.6f}, "
              f"feature distance={result['feature_frechet']:.3f}, "
              f"confidence={result['prior_confidence']:.4f}", flush=True)
    model, checkpoint = load_vae("vaegan.pt", x.device)
    metrics["experiment"] = dict(config=checkpoint["config"],
        baseline_config=load_vae("vae_base.pt", x.device)[1]["config"],
        classifier_test_accuracy=accuracy,
        vae_parameters=sum(p.numel() for p in model.parameters()),
        discriminator_parameters=sum(p.numel() for p in Discriminator().parameters()),
        prior_samples=10000, prior_seed=2027, posterior_mc_draws=10, posterior_seed=43,
        grid_count=441, grid_min=-3, grid_max=3, grid_step=0.3,
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      torchvision=torchvision.__version__, numpy=np.__version__,
                      matplotlib=matplotlib.__version__))
    metrics["experiment"]["classifier_epochs"] = len(torch.load(
        ART / "classifier.pt", map_location="cpu", weights_only=True)["history"])
    make_figures(arrays, histories, test_y.cpu().numpy())
    rows = [dict(sample=i + 1, epsilon_1=float(e[0]), epsilon_2=float(e[1]),
                 z_1=float(z[0]), z_2=float(z[1]))
            for i, (e, z) in enumerate(zip(eps, arrays["vaegan_five_z"]))]
    write_csv(OUT / "five_samples.csv", rows)
    write_json(OUT / "metrics.json", metrics)
    np.savez_compressed(OUT / "samples.npz", **arrays)
    files = [Path(__file__), *sorted(ART.glob("*.pt")),
             *sorted(OUT.glob("*.png")), *sorted(OUT.glob("*.csv")),
             OUT / "samples.npz", OUT / "metrics.json"]
    write_json(OUT / "sha256.json", {p.relative_to(ROOT).as_posix(): digest(p) for p in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=[
        "all", "pretrain", "gan", "ablation", "control", "classifier", "generate"])
    parser.add_argument("--pretrain-epochs", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--classifier-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adv-weight", type=float, default=5)
    parser.add_argument("--adv-warmup", type=int, default=10)
    parser.add_argument("--instance-noise", type=float, default=0.05)
    parser.add_argument("--run-name", default="vaegan", choices=[
        "vaegan", "vaegan_no_noise", "vaegan_strong", "vaegan_balanced"])
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()
    if min(args.pretrain_epochs, args.epochs, args.classifier_epochs,
           args.batch_size, args.adv_warmup) < 1 or args.lr <= 0 or args.adv_weight < 0:
        parser.error("Invalid training configuration")
    if args.instance_noise < 0:
        parser.error("Instance noise must be nonnegative")
    seed_all(args.seed)
    ART.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, test, labels, test_y = load_data(device)
    if args.mode == "pretrain" or (args.mode == "all" and not (ART / "vae_base.pt").exists()):
        pretrain(x, args)
    if args.mode in ("all", "gan"):
        train_variant(x, args, True)
    if args.mode in ("all", "ablation"):
        ablation_args = copy.copy(args)
        ablation_args.adv_weight = 20.0
        ablation_args.run_name = "vaegan_strong"
        train_variant(x, ablation_args, True)
        ablation_args.instance_noise = 0.0
        ablation_args.run_name = "vaegan_no_noise"
        train_variant(x, ablation_args, True)
    if args.mode in ("all", "control"):
        train_variant(x, args, False)
    if args.mode in ("all", "classifier"):
        train_classifier(x, labels, args)
    if args.mode in ("all", "generate"):
        generate(x, test, labels, test_y)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Execution failed ({type(error).__name__}). Check configuration, "
              "dependencies, data and memory.", file=sys.stderr)
        sys.exit(1)
