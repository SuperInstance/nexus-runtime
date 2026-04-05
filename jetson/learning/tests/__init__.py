"""NEXUS Learning tests - pytest path configuration."""
import sys
import os

_jetson_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _jetson_root not in sys.path:
    sys.path.insert(0, _jetson_root)
