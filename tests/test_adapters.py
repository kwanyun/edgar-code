import torch

from edgar.adapters.common import canvas
from edgar.adapters.rgbavatar import orbit
from edgar.avatar import optimize_avatar


def test_rgbavatar_zero_orbit_preserves_projection():
    w2c = torch.eye(4)
    w2c[:3, 3] = torch.tensor([0.2, -0.4, 2.0])
    projection = torch.diag(torch.tensor([1.2, 1.4, 1.0, 1.0]))
    view = {
        "cam_view_mat": w2c[None],
        "cam_proj_mat": (projection @ w2c)[None],
        "cam_position": torch.linalg.inv(w2c)[:3, 3][None],
    }
    result = orbit(view, 0)
    for key in view:
        torch.testing.assert_close(result[key], view[key])


def test_native_canvas_keeps_gradient_and_white_padding():
    image = torch.zeros((3, 512, 256), requires_grad=True)
    result = canvas(image)
    assert result.shape == (3, 512, 512)
    assert torch.all(result[:, :, :128] == 1)
    result.sum().backward()
    assert torch.all(image.grad == 1)


def test_native_post_step_hook_and_parameter_invariant():
    class Avatar(torch.nn.Module):
        gaussian_count = 1

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.ones(3, 16, 16) * 0.2)
            self.started = False
            self.updates = 0

        def begin_optimization(self):
            self.started = True

        def render(self, view):
            assert self.started
            return self.value

        def after_step(self):
            self.updates += 1

    avatar = Avatar()
    optimizer = torch.optim.Adam(avatar.parameters(), lr=0.01)
    losses = optimize_avatar(
        avatar, [0], [torch.ones(3, 16, 16) * 0.5], optimizer, steps=3
    )
    assert avatar.updates == 3
    assert losses[-1] < losses[0]


def test_numpy_geometry_copy_preserves_arrays(tmp_path):
    import pickle
    import numpy as np
    from edgar.adapters.numpy_pickle import convert_geometry_copy

    original = {"shape": np.arange(12, dtype=np.float32).reshape(3, 4)}
    # Protocol 0 GLOBAL references can be renamed without changing frame lengths.
    raw = pickle.dumps(original, protocol=0).replace(b"numpy.core", b"numpy._core")
    path = tmp_path / "copied_geometry.pkl"
    path.write_bytes(raw)
    assert convert_geometry_copy(path)
    with path.open("rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(restored["shape"], original["shape"])
    assert not convert_geometry_copy(path)


def test_cap4d_offsets_reject_a_different_driving_sequence():
    from types import SimpleNamespace
    import pytest
    from edgar.adapters.cap4d import restore_training_offsets, training_state_digest

    model = SimpleNamespace(
        static_neck=False,
        flame_param={
            name: torch.zeros(2, 3) for name in ("expr", "eye_rot", "rot", "tra")
        },
        neck_rot_offset=torch.nn.Embedding(2, 3),
    )
    state = {
        "edgar_training_state": {
            "sequence_sha256": training_state_digest(model),
            "neck_rot_offset": torch.full((2, 3), 0.01),
        }
    }
    restore_training_offsets(model, state)
    torch.testing.assert_close(model.neck_rot_offset.weight, torch.full((2, 3), 0.01))
    model.flame_param["expr"][0, 0] = 1
    with pytest.raises(ValueError, match="original tracked sequence"):
        restore_training_offsets(model, state)
