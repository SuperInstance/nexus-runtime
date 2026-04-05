"""NEXUS Skill System Tests — pytest path configuration."""
import sys
import os

# Add jetson directory to sys.path so module imports work
_jetson_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _jetson_root not in sys.path:
    sys.path.insert(0, _jetson_root)
