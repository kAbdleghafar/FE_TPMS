"""tpmsfem -- GPU-accelerated finite elements for TPMS lattice implants."""
from .backend import set_backend, backend_name, on_gpu, device_info  # noqa: F401
from .mesh.mesh import Mesh, read_mesh  # noqa: F401
from .fem.materials import Material  # noqa: F401
from .fem.model import SolidModel  # noqa: F401

__version__ = "0.1.0"
