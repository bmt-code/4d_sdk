"""
4D SDK for stereo 4D imaging.
"""

__version__ = "0.0.1"
__author__ = "Guilherme Soares Silvestre"
__credits__ = "Brazilian Medical Technologies (BMT)"

from .stereo_4d import (
    CustomLogger,
    Stereo4DCameraHandler,
    Stereo4DCameraInfo,
    Stereo4DFrame,
)

# The package's public surface. Named explicitly because these are re-exports and would
# otherwise read as unused imports.
__all__ = [
    "CustomLogger",
    "Stereo4DCameraHandler",
    "Stereo4DCameraInfo",
    "Stereo4DFrame",
]
