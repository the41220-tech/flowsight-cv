"""FlowSight AI — input-agnostic crowd-safety perception/physics toolkit.

Submodules are imported lazily; importing `flowsight` does NOT pull torch /
transformers so the geometry/physics/sim stack runs on a CPU-only box (e.g. the
research sandbox) while the model wrappers run on Colab GPU.
"""
__version__ = "0.0.1"
