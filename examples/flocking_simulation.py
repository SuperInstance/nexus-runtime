"""Flocking Simulation — 10-agent Reynolds flock in 2D.

Demonstrates the NEXUS swarm flocking module using classic Reynolds rules
(separation, alignment, cohesion) with 10 marine agents.

Run:
    cd /tmp/nexus-runtime && python examples/flocking_simulation.py
"""

import sys
import os
import random
import math
import io

# Ensure jetson package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress noisy separation force warnings from flocking module
from jetson.swarm.flocking import (
    Agent,
    FlockSimulation,
    FlockingParams,
    Obstacle,
)


def main() -> None:
    print("=" * 60)
    print("  NEXUS Flocking Simulation — 10 Marine Agents")
    print("=" * 60)

    random.seed(42)

    # ── Create 10 agents with random initial positions ──────────────
    agents = []
    for i in range(10):
        agent = Agent(
            agent_id=f"AUV-{i:02d}",
            x=random.uniform(20, 80),
            y=random.uniform(20, 80),
            vx=random.uniform(-1, 1),
            vy=random.uniform(-1, 1),
        )
        agents.append(agent)

    # ── Add an obstacle ─────────────────────────────────────────────
    obstacles = [Obstacle(x=50, y=50, radius=8)]

    # ── Configure flocking parameters ───────────────────────────────
    params = FlockingParams(
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        max_speed=4.0,
        max_force=2.5,
        perception_radius=40.0,
        separation_radius=12.0,
        obstacle_avoidance_radius=15.0,
        obstacle_avoidance_weight=3.0,
    )

    # ── Create simulation ───────────────────────────────────────────
    sim = FlockSimulation(agents=agents, obstacles=obstacles, params=params)

    print(f"\nAgents: {len(agents)}  |  Obstacles: {len(obstacles)}  |  Steps: 50")
    print(f"Params: sep={params.separation_weight}  ali={params.alignment_weight}  "
          f"coh={params.cohesion_weight}  speed={params.max_speed}")

    # ── Print initial positions ─────────────────────────────────────
    def print_state(label, agents):
        print(f"\n{label}")
        print(f"{'Agent':>8s}  {'X':>7s}  {'Y':>7s}  {'VX':>7s}  {'VY':>7s}  {'Speed':>6s}")
        print("-" * 50)
        for a in agents:
            print(f"{a.agent_id:>8s}  {a.x:7.1f}  {a.y:7.1f}  "
                  f"{a.vx:7.2f}  {a.vy:7.2f}  {a.speed():6.2f}")

    print_state("Initial positions:", agents)

    # ── Run 50 simulation steps, capturing snapshots ────────────────
    snapshots = {}
    snapshots[0] = [(a.agent_id, a.x, a.y, a.vx, a.vy, a.speed()) for a in agents]

    # Redirect stderr to suppress flocking warnings during simulation
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    for step in range(1, 51):
        sim.step(dt=1.0)
        if step in (10, 25, 50):
            snapshots[step] = [
                (a.agent_id, a.x, a.y, a.vx, a.vy, a.speed()) for a in agents
            ]

    sys.stderr = old_stderr

    # ── Print positions at key intervals ────────────────────────────
    for step_idx in [10, 25, 50]:
        data = snapshots[step_idx]
        print_state(f"After step {step_idx}:", [(type('A', (), {
            'agent_id': d[0], 'x': d[1], 'y': d[2], 'vx': d[3], 'vy': d[4], 'speed': lambda: d[5]
        })) for d in data])

    # ── Compute flock statistics ────────────────────────────────────
    print("\n" + "-" * 50)
    print("  Flock Statistics")
    print("-" * 50)

    final = snapshots[50]
    xs = [d[1] for d in final]
    ys = [d[2] for d in final]
    speeds = [d[5] for d in final]

    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    spread = math.sqrt(
        sum((x - centroid_x) ** 2 + (y - centroid_y) ** 2 for x, y in zip(xs, ys))
        / len(xs)
    )

    print(f"  Centroid:      ({centroid_x:.1f}, {centroid_y:.1f})")
    print(f"  Flock spread:  {spread:.1f} (RMS distance from centroid)")
    print(f"  Avg speed:     {sum(speeds) / len(speeds):.2f}")
    print(f"  Min speed:     {min(speeds):.2f}")
    print(f"  Max speed:     {max(speeds):.2f}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
