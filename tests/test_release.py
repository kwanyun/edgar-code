import json

import pytest
import torch
from PIL import Image

from edgar.data import TripletDataset, image_tensor, inference_tensor, read_manifest


def test_reference_normalization():
    image = Image.new("RGB", (32, 32), (123, 116, 103))
    x = image_tensor(image, reference=True)
    assert x.shape == (3, 1024, 1024)
    assert torch.allclose(
        x[:, 0, 0], torch.tensor([-0.5 / 58.5, -0.5 / 57, -0.5 / 57.5]), atol=1e-5
    )


def test_application_reference_matches_original_script():
    from torchvision import transforms

    pixels = torch.arange(21 * 17 * 3).remainder(256).byte().reshape(21, 17, 3).numpy()
    image = Image.fromarray(pixels)
    original = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )(image)
    actual = inference_tensor(image, reference=True, preprocessing="application")
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not torch.allclose(actual, image_tensor(image, reference=True))


def test_application_render_padding_preserves_aspect_ratio():
    image = Image.new("RGB", (256, 512), (0, 0, 0))
    actual = inference_tensor(image, preprocessing="application")
    assert actual.shape == (3, 512, 512)
    assert torch.all(actual[:, :, :128] == 1)
    assert torch.all(actual[:, :, 128:384] == -1)
    assert torch.all(actual[:, :, 384:] == 1)
    torch.testing.assert_close(
        inference_tensor(image), image_tensor(image), rtol=0, atol=0
    )
    with pytest.raises(ValueError):
        inference_tensor(image, preprocessing="unknown")


def test_packed_triplet_and_deterministic_validation(tmp_path):
    image = Image.new("RGB", (1536, 512))
    image.paste((255, 0, 0), (0, 0, 512, 512))
    image.paste((0, 255, 0), (512, 0, 1024, 512))
    image.paste((0, 0, 255), (1024, 0, 1536, 512))
    image.save(tmp_path / "triplet.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"validation": [{"triplet": "triplet.png"}]}))
    dataset = TripletDataset(manifest, "validation")
    a, b = dataset[0], dataset[0]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert a["input"][:, 0, 0].tolist() == [-1, -1, 1]
    assert a["target"][:, 0, 0].tolist() == [-1, 1, -1]


def test_bad_manifest_fails_early(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": [{"triplet": "missing.png"}]}))
    with pytest.raises(FileNotFoundError):
        read_manifest(manifest, "train")


def test_no_global_diffusers_patch():
    from diffusers.models.attention import BasicTransformerBlock

    before = BasicTransformerBlock.forward
    import edgar.model  # noqa: F401 -- importing must not patch the global class

    assert BasicTransformerBlock.forward is before


def test_incomplete_checkpoint_is_rejected(tmp_path):
    from edgar.model import read_checkpoint

    p = tmp_path / "incomplete.pt"
    torch.save({"rank_vae": 4}, p)
    with pytest.raises(ValueError, match="state_dict_ref_proj"):
        read_checkpoint(p)


def test_poster_has_no_implicit_backbone_downloads():
    from edgar.vendor.poster.PosterV2_8cls import pyramid_trans_expr2

    model = pyramid_trans_expr2()
    assert model.face_landback is not None


def test_poster_load_requires_features_but_ignores_classifier():
    from edgar.losses import load_expression_state

    model = torch.nn.Linear(2, 2)
    state = dict(model.state_dict(), **{"VIT.head.weight": torch.zeros(1)})
    load_expression_state(model, state)
    with pytest.raises(RuntimeError):
        load_expression_state(model, {"weight": state["weight"]})
    with pytest.raises(RuntimeError):
        load_expression_state(model, dict(state, unknown=torch.zeros(1)))


def test_avatar_optimization_and_identity_ssim():
    from edgar.avatar import optimize_avatar, ssim_loss

    class Avatar(torch.nn.Module):
        gaussian_count = 1

        def __init__(self):
            super().__init__()
            self.color = torch.nn.Parameter(torch.full((3, 16, 16), 0.2))

        def render(self, view):
            return self.color

    avatar = Avatar()
    target = torch.ones(3, 16, 16) * 0.8
    assert abs(float(ssim_loss(target, target))) < 1e-6
    losses = optimize_avatar(
        avatar, [0], [target], torch.optim.Adam(avatar.parameters(), lr=0.05), steps=4
    )
    assert losses[-1] < losses[0]
