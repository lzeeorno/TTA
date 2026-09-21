"""
ATLAS: Adaptive Test-time Learning Across Scenarios.
"""

from .classification import ATLAS, create_atlas

__all__ = [
    "ATLAS",
    "create_atlas",
    "ATLASSegmentationAdapter",
    "ATLASInstance",
]


def __getattr__(name):
    if name == "ATLASSegmentationAdapter":
        from .segmentation import ATLASSegmentationAdapter

        return ATLASSegmentationAdapter
    if name == "ATLASInstance":
        from .vlm_instance import ATLASInstance

        return ATLASInstance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
