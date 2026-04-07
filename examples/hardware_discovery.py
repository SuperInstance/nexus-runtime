"""Hardware Discovery — Explore the NEXUS hardware catalog.

Lists all supported platforms, boards, and architectural details from the
NEXUS hardware configuration package. No hardware required.

Run:
    cd /tmp/nexus-runtime && python examples/hardware_discovery.py
"""

import sys
import os

# Ensure hardware package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware import (
    list_platforms,
    list_boards,
    list_all_boards,
    get_platform_display,
    total_board_count,
)


def main() -> None:
    print("=" * 60)
    print("  NEXUS Hardware Catalog Discovery")
    print("=" * 60)

    # ── Summary ─────────────────────────────────────────────────────
    platforms = list_platforms()
    total = total_board_count()

    print(f"\n  Total platforms: {len(platforms)}")
    print(f"  Total boards:    {total}")
    print()

    # ── Per-platform details ────────────────────────────────────────
    print("-" * 60)
    print(f"  {'Platform':<18s} {'Display Name':<32s} {'Boards':>6s}")
    print("-" * 60)

    for platform in platforms:
        boards = list_boards(platform)
        display = get_platform_display(platform)
        print(f"  {platform:<18s} {display:<32s} {len(boards):>6d}")

    # ── Full board listing ──────────────────────────────────────────
    all_boards = list_all_boards()

    print("\n" + "=" * 60)
    print("  Full Board Listing by Platform")
    print("=" * 60)

    for platform, boards in all_boards.items():
        display = get_platform_display(platform)
        print(f"\n  [{platform}] {display}")
        for board in boards:
            print(f"    - {board}")

    # ── Platform family categories ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  Platform Categories")
    print("=" * 60)

    mcu_platforms = ["arduino", "esp32", "esp8266", "stm32", "nrf52", "teensy", "imx_rt", "rp2040"]
    gpu_platforms = ["jetson_nano"]
    sbc_platforms = ["raspberry_pi", "beaglebone"]

    mcu_boards = sum(len(list_boards(p)) for p in mcu_platforms if p in all_boards)
    gpu_boards = sum(len(list_boards(p)) for p in gpu_platforms if p in all_boards)
    sbc_boards = sum(len(list_boards(p)) for p in sbc_platforms if p in all_boards)

    print(f"\n  Microcontrollers:  {mcu_boards:>3d} boards  ({', '.join(mcu_platforms[:4])}, ...)")
    print(f"  Edge GPUs:         {gpu_boards:>3d} boards  ({', '.join(gpu_platforms)})")
    print(f"  Single-Board PCs:  {sbc_boards:>3d} boards  ({', '.join(sbc_platforms)})")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
