#!/usr/bin/env python3
"""
CLI tool for logging power measurements from the JT-TC66C USB power meter.

Connects via BLE, polls voltage/current/power at a configurable interval,
and writes timestamped rows to a CSV file.

Usage
-----
    # Auto-discover TC66C and log for 120 seconds at 1 Hz:
    python edge/power_log.py -o results/power/run01.csv -d 120

    # Specify MAC address, 0.5 Hz sample rate:
    python edge/power_log.py --address AA:BB:CC:DD:EE:FF -i 2.0 -d 300

    # Scan only (find the device address):
    python edge/power_log.py --scan
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path for editable-install-free usage
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.tc66c import (
    TC66CReading,
    collect,
    scan_for_tc66c,
)


def _print_reading(r: TC66CReading) -> None:
    """Live console output: one-line summary per sample."""
    print(
        f"  {r.voltage_V:7.3f} V  "
        f"{r.current_A:8.5f} A  "
        f"{r.power_W:7.3f} W  "
        f"{r.temp_C:5.1f} °C",
        flush=True,
    )


async def _scan(verbose: bool = False) -> None:
    """Scan for TC66C.  With --verbose, dump all visible BLE devices."""
    from bleak import BleakScanner

    timeout = 15.0 if verbose else 10.0
    print(f"Active-scanning BLE devices ({timeout:.0f} s) …")

    if verbose:
        devices = await BleakScanner.discover(
            timeout=timeout, return_adv=True, scanning_mode="active",
        )
        print(f"\n{'Address':20s}  {'RSSI':>5s}  {'Name':<30s}  Service UUIDs")
        print("─" * 90)
        for addr, (dev, adv) in sorted(
            devices.items(), key=lambda x: x[1][1].rssi, reverse=True,
        ):
            name = dev.name or adv.local_name or "(unnamed)"
            uuids = ", ".join(adv.service_uuids or []) or "—"
            print(f"{addr:20s}  {adv.rssi:5d}  {name:<30s}  {uuids}")
        print(f"\nTotal: {len(devices)} devices")

    try:
        addr = await scan_for_tc66c(timeout=timeout)
        print(f"\nFound TC66C at: {addr}")
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    # Resolve device address
    if args.address is None:
        print("No address given — scanning …")
        args.address = await scan_for_tc66c(timeout=10.0)
        print(f"Auto-detected TC66C at: {args.address}")

    csv_path = Path(args.output) if args.output else None
    print(
        f"Logging {args.address} → {csv_path or '(stdout only)'}  "
        f"[{args.duration}s @ {1/args.interval:.1f} Hz]"
    )
    print(f"{'Voltage':>10}  {'Current':>10}  {'Power':>10}  {'Temp':>8}")
    print("  " + "─" * 46)

    readings = await collect(
        address=args.address,
        duration_s=args.duration,
        interval_s=args.interval,
        csv_path=csv_path,
        on_reading=_print_reading,
    )

    print(f"\nCollected {len(readings)} samples.")
    if csv_path:
        print(f"Saved to {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Log power from JT-TC66C via BLE"
    )
    ap.add_argument(
        "--scan", action="store_true",
        help="scan for TC66C and print its address, then exit",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="with --scan, list all visible BLE devices for debugging",
    )
    ap.add_argument(
        "-a", "--address",
        help="BLE MAC address (Linux/Windows) or UUID (macOS). "
             "If omitted, auto-scan is performed.",
    )
    ap.add_argument(
        "-o", "--output",
        help="CSV output path (created if needed)",
    )
    ap.add_argument(
        "-d", "--duration", type=float, default=60.0,
        help="collection duration in seconds (default: 60)",
    )
    ap.add_argument(
        "-i", "--interval", type=float, default=1.0,
        help="seconds between polls (default: 1.0)",
    )
    args = ap.parse_args()

    if args.scan:
        asyncio.run(_scan(verbose=args.verbose))
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
