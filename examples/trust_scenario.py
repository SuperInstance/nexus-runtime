"""Trust Scenario — INCREMENTS trust evolution between 5 AUV agents.

Demonstrates the NEXUS trust engine with a realistic multi-agent marine
scenario: five autonomous underwater vehicles cooperate on a survey mission.
Trust scores evolve based on interaction quality (success/failure, latency).

Run:
    cd /tmp/nexus-runtime && python examples/trust_scenario.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nexus.trust.engine import TrustEngine, CapabilityProfile

AGENT_NAMES = ["AUV-Alpha", "AUV-Bravo", "AUV-Charlie", "AUV-Delta", "AUV-Echo"]


def main() -> None:
    print("=" * 65)
    print("  NEXUS INCREMENTS Trust Scenario — 5 AUV Agents")
    print("=" * 65)

    # ── Create trust engine ──────────────────────────────────────────
    engine = TrustEngine()

    # ── Register 5 agents with different capability profiles ─────────
    profiles = {
        "AUV-Alpha":   CapabilityProfile(navigation=0.95, sensing=0.80, speed=0.90),
        "AUV-Bravo":   CapabilityProfile(navigation=0.70, sensing=0.95, speed=0.60),
        "AUV-Charlie": CapabilityProfile(navigation=0.85, sensing=0.85, speed=0.75),
        "AUV-Delta":   CapabilityProfile(navigation=0.60, sensing=0.70, speed=0.80),
        "AUV-Echo":    CapabilityProfile(navigation=0.90, sensing=0.75, speed=0.85),
    }

    for name, caps in profiles.items():
        engine.register_agent(name, capabilities=caps)
    print(f"\nRegistered {engine.agent_count} agents:")
    for agent in engine.list_agents():
        print(f"  - {agent.name}: nav={agent.capabilities.navigation:.2f}  "
              f"sense={agent.capabilities.sensing:.2f}  "
              f"speed={agent.capabilities.speed:.2f}")

    # ── Simulate cooperative survey mission ─────────────────────────
    interactions = [
        # (trustor,        trustee,         success, latency_ms)
        ("AUV-Alpha",   "AUV-Bravo",   True,  45),
        ("AUV-Alpha",   "AUV-Charlie", True,  30),
        ("AUV-Bravo",   "AUV-Alpha",   True,  50),
        ("AUV-Bravo",   "AUV-Delta",   False, 200),
        ("AUV-Charlie", "AUV-Echo",    True,  35),
        ("AUV-Charlie", "AUV-Alpha",   True,  25),
        ("AUV-Delta",   "AUV-Bravo",   True,  60),
        ("AUV-Delta",   "AUV-Echo",    False, 300),
        ("AUV-Echo",    "AUV-Alpha",   True,  40),
        ("AUV-Echo",    "AUV-Charlie", True,  55),
        # Round 2 — Delta starts improving
        ("AUV-Alpha",   "AUV-Bravo",   True,  42),
        ("AUV-Bravo",   "AUV-Delta",   True,  90),   # improved!
        ("AUV-Bravo",   "AUV-Delta",   True,  80),   # consistent
        ("AUV-Delta",   "AUV-Echo",    True,  70),   # improved
        ("AUV-Charlie", "AUV-Delta",   True,  65),   # new pairing
        ("AUV-Echo",    "AUV-Alpha",   True,  38),
        ("AUV-Alpha",   "AUV-Echo",    True,  28),
        ("AUV-Bravo",   "AUV-Charlie", True,  48),
        # Round 3 — Bravo has a failure streak
        ("AUV-Alpha",   "AUV-Bravo",   False, 250),
        ("AUV-Alpha",   "AUV-Bravo",   False, 280),
    ]

    print(f"\nSimulating {len(interactions)} interactions...\n")

    for i, (trustor, trustee, success, latency) in enumerate(interactions, 1):
        score = engine.record_interaction(
            trustor, trustee, success=success, latency_ms=latency
        )
        status = "OK" if success else "FAIL"
        print(f"  [{i:2d}] {trustor:12s} -> {trustee:12s}  "
              f"{status:4s}  latency={latency:4.0f}ms  "
              f"trust={score:.3f}")

    # ── Show final trust matrix ─────────────────────────────────────
    print("\n" + "-" * 65)
    print("  Final Trust Matrix (trustor -> trustee)")
    print("-" * 65)

    header = f"{'':>14s}"
    for name in AGENT_NAMES:
        short = name.split("-")[1][:6]
        header += f"  {short:>8s}"
    print(header)
    print("-" * len(header))

    for trustor in AGENT_NAMES:
        row = f"{trustor:>14s}"
        for trustee in AGENT_NAMES:
            if trustor == trustee:
                row += f"  {'---':>8s}"
            else:
                score = engine.get_trust(trustor, trustee)
                row += f"  {score:>8.3f}"
        print(row)

    # ── Show trust profiles with most-trusted analysis ──────────────
    print("\n" + "-" * 65)
    print("  Most-Trusted Agent (per trustor)")
    print("-" * 65)
    for trustor in AGENT_NAMES:
        result = engine.get_most_trusted(trustor)
        if result:
            best_id, best_score = result
            print(f"  {trustor:14s} trusts {best_id:14s} most (score={best_score:.3f})")

    # ── Show Delta's recovery ───────────────────────────────────────
    print("\n" + "-" * 65)
    print("  Delta's Trust Recovery")
    print("-" * 65)
    for trustor in AGENT_NAMES:
        if trustor == "AUV-Delta":
            continue
        profile = engine.get_trust_profile(trustor, "AUV-Delta")
        if profile:
            print(f"  {trustor:14s} -> AUV-Delta:  score={profile.composite_score:.3f}  "
                  f"interactions={profile.interaction_count}  "
                  f"history_ema={profile.dimensions.history:.3f}  "
                  f"consistency={profile.dimensions.consistency:.3f}")

    print(f"\nTotal trust profiles tracked: {engine.profile_count}")
    print("Done.")


if __name__ == "__main__":
    main()
