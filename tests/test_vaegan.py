"""Check objective units, gradient boundaries, sampling and report integrity."""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vaegan import (VAE, Discriminator, d_loss, g_loss, loss_terms,
                    make_grid, feature_distance, np, torch, seed_all)
import vaegan


class MathematicsTests(unittest.TestCase):
    def setUp(self):
        seed_all(7)

    def test_loss_scale_and_lsgan_equilibrium(self):
        x = torch.zeros(2, 784)
        l1, kl = loss_terms(x + 0.25, x, torch.tensor([[0., 0.], [1., 2.]]),
                            torch.zeros(2, 2))
        self.assertAlmostEqual(float(l1), 196.)
        self.assertAlmostEqual(float(kl), 1.25)
        scores = torch.full((4,), 0.5)
        self.assertAlmostEqual(float(d_loss(scores, scores, scores)), 0.25)
        self.assertAlmostEqual(float(g_loss(scores, scores)), 0.125)

    def test_reparameterization_derivatives(self):
        mu = torch.tensor([[1., 2.]], requires_grad=True)
        lv = torch.zeros_like(mu, requires_grad=True)
        eps = torch.tensor([[2., -1.]])
        z = VAE.reparameterize(mu, lv, eps)
        torch.testing.assert_close(z, torch.tensor([[3., 1.]]))
        z.sum().backward()
        torch.testing.assert_close(mu.grad, torch.ones_like(mu))
        torch.testing.assert_close(lv.grad, eps / 2)

    def test_adversarial_gradient_boundaries(self):
        model, disc = VAE(), Discriminator()
        x = torch.rand(4, 784)
        rec, _, _ = model(x, torch.randn(4, 2))
        prior = model.decoder(torch.randn(4, 2))
        d_loss(disc(x), disc(rec.detach()), disc(prior.detach())).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(any(p.grad is not None for p in disc.parameters()))
        disc.zero_grad(set_to_none=True)
        disc.eval().requires_grad_(False)
        buffers_before = {n: b.clone() for n, b in disc.named_buffers()}
        g_loss(disc(rec), disc(prior)).backward()
        self.assertTrue(all(p.grad is None for p in disc.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertGreater(float(model.mu.weight.grad.abs().sum()), 0.)
        for name, buffer in disc.named_buffers():
            torch.testing.assert_close(buffer, buffers_before[name])

    def test_grid_and_feature_distance(self):
        grid = make_grid().numpy().reshape(21, 21, 2)
        np.testing.assert_allclose(grid[0, :, 0], np.linspace(-3, 3, 21), atol=1e-6)
        np.testing.assert_allclose(grid[:, 0, 1], np.linspace(3, -3, 21), atol=1e-6)
        features = np.random.default_rng(7).normal(size=(100, 4))
        self.assertAlmostEqual(feature_distance(features, features), 0., places=8)
        self.assertAlmostEqual(feature_distance(features, features + 1), 4., places=8)

    def test_standalone_pretrain_and_continuation(self):
        args = SimpleNamespace(pretrain_epochs=1, epochs=1, batch_size=4,
            seed=84, lr=1e-4, adv_weight=20., adv_warmup=10, classifier_epochs=1,
            instance_noise=0.05, run_name="vaegan")
        workspace = vaegan.ROOT / "tmp"
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            folder = Path(directory)
            self.assertTrue(folder.resolve().is_relative_to(workspace.resolve()))
            with patch.object(vaegan, "ART", folder), patch.object(vaegan, "OUT", folder):
                x = torch.rand(8, 784)
                vaegan.pretrain(x, args)
                base = vaegan.load_vae("vae_base.pt", x.device)[0]
                vaegan.train_variant(x, args, True)
                vaegan.train_variant(x, args, False)
                gan = vaegan.load_vae("vaegan.pt", x.device)[0]
                control = vaegan.load_vae("vae_control.pt", x.device)[0]
                self.assertFalse(torch.equal(base.mu.weight, gan.mu.weight))
                self.assertFalse(torch.equal(gan.mu.weight, control.mu.weight))
                vaegan.train_classifier(x, torch.arange(8), args)
                self.assertTrue((folder / "classifier.pt").is_file())


class ArtifactTests(unittest.TestCase):
    @unittest.skipUnless(Path("results/samples.npz").exists(), "Results pending")
    def test_saved_samples_and_hashes(self):
        data = np.load("results/samples.npz")
        self.assertEqual(data["vaegan_grid"].shape, (441, 784))
        self.assertEqual(data["vaegan_five"].shape, (10, 784))
        np.testing.assert_allclose(data["vaegan_five_z"], data["vaegan_five_mu"] +
            np.exp(data["vaegan_five_logvar"] / 2) * data["epsilon"], atol=1e-6)
        self.assertEqual(len(np.unique(data["vaegan_five_z"], axis=0)), 10)
        self.assertTrue(np.all((data["vaegan_grid"] >= 0) & (data["vaegan_grid"] <= 1)))
        hashes = json.loads(Path("results/sha256.json").read_text())
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), expected)

    @unittest.skipUnless(Path("reports/vaegan_mnist_report.md").exists(), "Report pending")
    def test_report_is_complete_and_portable(self):
        report = Path("reports/vaegan_mnist_report.md").read_text(encoding="utf-8")
        source = Path("src/vaegan.py").read_text(encoding="utf-8")
        self.assertEqual(report.split("```python\n")[1].split("```")[0], source)
        self.assertLessEqual(len(source.splitlines()), 800)
        self.assertIsNone(re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", report))
        self.assertNotIn("作业", report)
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
            self.assertTrue((Path("reports") / target).is_file())
        self.assertNotRegex(report, r"\{\{[A-Z_]+\}\}")


if __name__ == "__main__":
    unittest.main()
