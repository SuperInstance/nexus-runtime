"""NEXUS Runtime CLI entry point.

Allows users to run ``python -m jetson`` to verify installation
and inspect basic system information.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

def main() -> None:
    logger = logging.getLogger("jetson")
    logger.info("NEXUS Runtime v0.1.0")

    # Verify key submodules can be imported
    modules = [
        "jetson.performance.resource_monitor",
        "jetson.swarm.flocking",
        "jetson.navigation.pilot",
        "jetson.fleet_coordination.fleet_manager",
        "jetson.security.authentication",
        "jetson.decision_engine.multi_objective",
    ]

    loaded = 0
    for mod_name in modules:
        try:
            __import__(mod_name)
            loaded += 1
        except Exception as exc:
            logger.error("Failed to import %s: %s", mod_name, exc)

    logger.info("Core modules loaded: %d/%d", loaded, len(modules))

    if loaded == len(modules):
        logger.info("NEXUS Runtime is ready.")
    else:
        logger.warning("Some modules failed to load — check errors above.")

    return 0 if loaded == len(modules) else 1


if __name__ == "__main__":
    sys.exit(main())
