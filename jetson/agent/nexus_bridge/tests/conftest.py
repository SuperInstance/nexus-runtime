"""NEXUS git-agent bridge tests — pytest path configuration."""
import sys
import os

# Add jetson/agent to sys.path so nexus_bridge can be imported
_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
