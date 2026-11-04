"""Read trusted NumPy-2-authored geometry pickles in the NumPy 1.x avatar runtime."""

import pickle
import io
from pathlib import Path


class GeometryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


# torch.load accepts a pickle-module interface for native checkpoints that also
# contain NumPy scalar metadata. Tensor storage handling remains PyTorch's own.
Unpickler = GeometryUnpickler


def load(stream, **kwargs):
    return GeometryUnpickler(stream, **kwargs).load()


def loads(data, **kwargs):
    return load(io.BytesIO(data), **kwargs)


def convert_geometry_copy(path):
    """Rewrite an already copied native asset; callers retain the original file."""
    path = Path(path)
    if b"numpy._core" not in path.read_bytes():
        return False
    with path.open("rb") as stream:
        value = GeometryUnpickler(stream, encoding="latin1").load()
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)
    return True
