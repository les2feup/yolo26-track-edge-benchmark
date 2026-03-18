#!/usr/bin/env python3
"""
Host-side power profiling with remote inference via SSH.

Decouples TC66C power sampling from on-device inference by running them
on separate machines:
  - HOST (this script): polls TC66C at a fixed 1 Hz via serial (or BLE fallback)
  - DUT  (Arduino Uno Q via SSH): runs NCNN inference for a timed window

This eliminates the CPU-starvation problem where heavy inference on the
DUT throttles the BLE power-logging thread, causing under-sampling that
scales with model size (e.g. yolo26m gets 5–7 s intervals instead of ~1.35 s).

Topology
--------
    ┌────────────┐  USB serial/BLE  ┌─────────┐   USB-C power   ┌─────────┐
    │  HOST      │ ◄──────────────  │  TC66C  │ ◄────────────── │   DUT   │
    │  (this     │                  │  meter  │                  │ Arduino │
    │   script)  │                  └─────────┘                  │  Uno Q  │
    │            │ ─── SSH ──────────────────────────────────────►│         │
    └────────────┘                                               └─────────┘

Usage
-----
    # Default: all conditions × 5 reps, serial transport
    python edge/custom/remote_power_profile.py

    # Single condition for testing
    python edge/custom/remote_power_profile.py --models yolo26n --resolutions 640 --reps 1

    # Force BLE transport (for connectivity testing)
    python edge/custom/remote_power_profile.py --transport ble --reps 1

    # Dry-run: print what would be executed without running
    python edge/custom/remote_power_profile.py --dry-run
"""

from __future__ import annotations

import argparse
import glob as _glob
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports (host-side only: tc66c + device_profile)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmark.tc66c import (
    TC66CReading,
    collect,
    collect_serial,
    scan_for_tc66c,
    summarise_readings,
    trim_warmup,
    summarise_across_runs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SSH target for the Arduino Uno Q
SSH_HOST = "arduino@arduino-uno-q"
REMOTE_PROJECT = "~/Developer/yolo26-track-edge-benchmark"
REMOTE_CONDA_SH = "/home/arduino/miniforge3/etc/profile.d/conda.sh"
REMOTE_ENV_ACTIVATE = (
    f". {REMOTE_CONDA_SH} && conda activate yolo-edge && cd {REMOTE_PROJECT}"
)

# Measurement protocol (matches notebook 04)
WARMUP_S = 120.0
MEASURE_S = 180.0
TOTAL_REP_S = WARMUP_S + MEASURE_S
INTERVAL_S = 1.0
TRIM_S = 5.0
N_REPS = 5

# Workload sequence (shortest MOT17 static-camera sequence, 525 frames)
POWER_SEQ = "MOT17-09"
SEQ_SUFFIX = "SDP"

# Model conditions (mirrors arduino_uno_q.yaml)
ALL_MODELS = ["yolo26n", "yolo26s", "yolo26m"]
ALL_RESOLUTIONS = [640, 576]

# Output paths (local host)
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
POWER_DIR = RESULTS_ROOT / "power" / "arduino_uno_q"


# ---------------------------------------------------------------------------
# TC66C transport detection
# ---------------------------------------------------------------------------

def detect_transport() -> tuple[str, str | None, str | None]:
    """Auto-detect TC66C: USB serial first, BLE fallback.

    Returns (transport, serial_port, ble_address) where exactly one of
    serial_port / ble_address is non-None.
    """
    # 1. USB serial (preferred — deterministic timing, no radio jitter)
    for port in sorted(_glob.glob("/dev/ttyACM*")):
        try:
            import serial as pyserial
            ser = pyserial.Serial(port, baudrate=115200, timeout=2)
            time.sleep(0.2)
            ser.reset_input_buffer()
            ser.write(b"getva\r\n")
            probe = ser.read(192)
            ser.close()
            if len(probe) == 192:
                return "serial", port, None
        except Exception:
            continue

    # 2. BLE fallback (useful for connectivity testing)
    import asyncio
    try:
        addr = asyncio.run(scan_for_tc66c(timeout=10.0))
        return "ble", None, addr
    except Exception as exc:
        raise RuntimeError(
            "TC66C not found on any transport.\n"
            f"  Serial: no valid device on /dev/ttyACM*\n"
            f"  BLE:    scan failed ({exc})\n"
            "  Check that TC66C is powered and connected."
        ) from exc


def force_transport(transport: str) -> tuple[str, str | None, str | None]:
    """Force a specific transport type."""
    if transport == "serial":
        for port in sorted(_glob.glob("/dev/ttyACM*")):
            try:
                import serial as pyserial
                ser = pyserial.Serial(port, baudrate=115200, timeout=2)
                time.sleep(0.2)
                ser.reset_input_buffer()
                ser.write(b"getva\r\n")
                probe = ser.read(192)
                ser.close()
                if len(probe) == 192:
                    return "serial", port, None
            except Exception:
                continue
        raise RuntimeError("TC66C not found on USB serial (/dev/ttyACM*)")

    elif transport == "ble":
        import asyncio
        addr = asyncio.run(scan_for_tc66c(timeout=10.0))
        return "ble", None, addr

    raise ValueError(f"Unknown transport: {transport!r}")


# ---------------------------------------------------------------------------
# Local power collection (runs in a thread, completely decoupled from SSH)
# ---------------------------------------------------------------------------

def run_power_collection(
    transport: str,
    serial_port: str | None,
    ble_address: str | None,
    duration_s: float,
    interval_s: float,
    csv_path: Path,
    stop_event: threading.Event | None = None,
) -> list[TC66CReading]:
    """Collect TC66C readings until stop_event or duration expires.

    Runs synchronously — intended to be called from a dedicated thread.
    """
    if transport == "serial":
        return collect_serial(
            serial_port,
            duration_s=duration_s,
            interval_s=interval_s,
            csv_path=csv_path,
            stop_event=stop_event,
        )
    else:
        import asyncio
        assert ble_address is not None, "BLE transport requires an address"
        # BLE collect() expects asyncio.Event; bridge from threading.Event
        async_stop = None
        if stop_event is not None:
            async_stop = asyncio.Event()

            async def _bridge():
                while not stop_event.is_set():
                    await asyncio.sleep(0.5)
                async_stop.set()  # type: ignore[union-attr]

        async def _run():
            if async_stop is not None:
                asyncio.ensure_future(_bridge())
            return await collect(
                ble_address,
                duration_s=duration_s,
                interval_s=interval_s,
                csv_path=csv_path,
                stop_event=async_stop,
            )

        return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Remote inference via SSH
# ---------------------------------------------------------------------------

REMOTE_WORKER = f"{REMOTE_PROJECT}/edge/custom/dut_inference_worker.py"
REMOTE_PROFILE = f"{REMOTE_PROJECT}/edge/profiles/arduino_uno_q.yaml"


def sync_worker_to_dut() -> bool:
    """Rsync the edge/custom/ directory to the DUT so the worker script is current."""
    local_custom = Path(__file__).resolve().parent
    remote_dest = f"{SSH_HOST}:{REMOTE_PROJECT}/edge/custom/"

    result = subprocess.run(
        ["rsync", "-az", "--delete",
         f"{local_custom}/", remote_dest],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def build_ssh_inference_cmd(model_variant: str, imgsz: int, duration_s: float) -> list[str]:
    """SSH command that invokes dut_inference_worker.py on the DUT."""
    ncnn_model = f"{model_variant}_{imgsz}_ncnn_model"

    remote_cmd = (
        f"{REMOTE_ENV_ACTIVATE} && "
        f"python3 {REMOTE_WORKER} "
        f"--profile {REMOTE_PROFILE} "
        f"--model {ncnn_model} "
        f"--imgsz {imgsz} "
        f"--duration {duration_s}"
    )

    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        SSH_HOST, remote_cmd,
    ]


def run_remote_inference(
    model_variant: str,
    imgsz: int,
    duration_s: float,
    verbose: bool = True,
) -> subprocess.CompletedProcess:
    """SSH into the DUT and run timed inference. Blocks until complete."""
    cmd = build_ssh_inference_cmd(model_variant, imgsz, duration_s)

    if verbose:
        print(f"    [SSH] {model_variant}_{imgsz} ({duration_s:.0f}s budget)")

    # Generous timeout: inference budget + 120s for SSH overhead + model load
    timeout = duration_s + 120

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if verbose and result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"    [DUT] {line}")

    if result.returncode != 0:
        print(f"    [DUT] ERROR (rc={result.returncode}): {result.stderr.strip()}")

    return result


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def power_csv_path(model_variant: str, imgsz: int, rep: int) -> Path:
    return POWER_DIR / f"{model_variant}_{imgsz}_rep{rep:02d}.csv"


def idle_csv_path() -> Path:
    return POWER_DIR / "idle.csv"


def existing_reps(model_variant: str, imgsz: int) -> list[Path]:
    pattern = f"{model_variant}_{imgsz}_rep*.csv"
    return sorted(POWER_DIR.glob(pattern))


# ---------------------------------------------------------------------------
# SSH connectivity pre-check
# ---------------------------------------------------------------------------

def check_ssh() -> bool:
    """Verify SSH connectivity to the DUT."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             SSH_HOST, "echo ok"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_remote_env() -> bool:
    """Verify the worker script and benchmark package are importable on the DUT."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             SSH_HOST,
             f"{REMOTE_ENV_ACTIVATE} && python3 {REMOTE_WORKER} --help"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_idle_baseline(
    transport: str,
    serial_port: str | None,
    ble_address: str | None,
    force: bool = False,
):
    """Collect 30 s idle baseline (no inference running on DUT)."""
    csv_path = idle_csv_path()

    if csv_path.exists() and not force:
        print(f"  Idle baseline exists: {csv_path.name} (use --force-idle to recollect)")
        return

    print("  Collecting idle baseline (30 s) — ensure DUT is idle ...")
    POWER_DIR.mkdir(parents=True, exist_ok=True)

    readings = run_power_collection(
        transport, serial_port, ble_address,
        duration_s=30.0, interval_s=INTERVAL_S,
        csv_path=csv_path, stop_event=None,
    )
    print(f"  Idle baseline: {len(readings)} samples → {csv_path.name}")


def run_one_condition(
    model_variant: str,
    imgsz: int,
    rep: int,
    transport: str,
    serial_port: str | None,
    ble_address: str | None,
    dry_run: bool = False,
) -> bool:
    """Run a single (model, resolution, rep) measurement.

    Returns True on success, False on failure.
    """
    csv_path = power_csv_path(model_variant, imgsz, rep)

    if dry_run:
        print(f"    [DRY] rep {rep:02d}: would collect → {csv_path.name}")
        return True

    print(f"  rep {rep:02d}/{N_REPS - 1}: "
          f"{WARMUP_S:.0f}s warm-up + {MEASURE_S:.0f}s measure ...")

    POWER_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: start local power collection in a background thread ──
    stop_power = threading.Event()
    power_readings: list[TC66CReading] = []
    power_error: list[Exception] = []

    def _power_worker():
        try:
            result = run_power_collection(
                transport, serial_port, ble_address,
                duration_s=TOTAL_REP_S + 30,  # margin; stop_power ends it
                interval_s=INTERVAL_S,
                csv_path=csv_path,
                stop_event=stop_power,
            )
            power_readings.extend(result)
        except Exception as exc:
            power_error.append(exc)

    power_thread = threading.Thread(target=_power_worker, daemon=True)
    power_thread.start()

    # Brief pause so the first power sample lands before inference starts
    time.sleep(2.0)

    # ── Phase 2: run inference remotely via SSH (blocks until done) ────
    try:
        result = run_remote_inference(model_variant, imgsz, TOTAL_REP_S)
        if result.returncode != 0:
            print(f"    WARNING: SSH inference returned rc={result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"    ERROR: SSH inference timed out")
    except Exception as exc:
        print(f"    ERROR: SSH inference failed: {exc}")

    # ── Phase 3: stop power collection ────────────────────────────────
    stop_power.set()
    power_thread.join(timeout=15)

    if power_error:
        print(f"    ERROR: power collection failed: {power_error[0]}")
        return False

    n_samples = len(power_readings)
    duration = (power_readings[-1].timestamp - power_readings[0].timestamp
                if n_samples > 1 else 0.0)
    interval = duration / (n_samples - 1) if n_samples > 1 else 0.0

    print(f"    {n_samples} samples, {duration:.0f}s, "
          f"interval={interval:.2f}s → {csv_path.name}")

    # Sanity check: flag if sampling is still degraded
    if interval > 2.0:
        print(f"    WARNING: sample interval {interval:.2f}s > 2.0s "
              f"(expected ~1.35s) — possible transport issue")

    return True


def main():
    ap = argparse.ArgumentParser(
        description="Host-side power profiling with remote SSH inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help=f"Model variants to profile (default: {ALL_MODELS})",
    )
    ap.add_argument(
        "--resolutions", nargs="+", type=int, default=ALL_RESOLUTIONS,
        help=f"Input resolutions (default: {ALL_RESOLUTIONS})",
    )
    ap.add_argument(
        "--reps", type=int, default=N_REPS,
        help=f"Repetitions per condition (default: {N_REPS})",
    )
    ap.add_argument(
        "--transport", choices=["auto", "serial", "ble"], default="auto",
        help="TC66C transport: auto-detect (default), force serial, or force BLE",
    )
    ap.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip conditions where all rep CSVs already exist (default: True)",
    )
    ap.add_argument(
        "--no-skip-existing", action="store_false", dest="skip_existing",
        help="Re-run all conditions even if CSVs exist",
    )
    ap.add_argument(
        "--force-idle", action="store_true",
        help="Recollect idle baseline even if it exists",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print planned actions without executing",
    )

    args = ap.parse_args()
    n_reps = args.reps

    # ── Pre-flight checks ─────────────────────────────────────────────
    print("=" * 60)
    print("Remote Power Profiling — Arduino Uno Q")
    print("=" * 60)

    # 1. TC66C transport
    print("\n[1/4] Detecting TC66C ...")
    if args.transport == "auto":
        transport, serial_port, ble_address = detect_transport()
    else:
        transport, serial_port, ble_address = force_transport(args.transport)
    addr_str = serial_port if transport == "serial" else ble_address
    print(f"  Transport: {transport.upper()} ({addr_str})")

    # 2. SSH connectivity + rsync worker script
    print("\n[2/4] Checking SSH to DUT ...")
    if args.dry_run:
        print(f"  [DRY] Would SSH to {SSH_HOST}")
    else:
        if not check_ssh():
            print(f"  FATAL: cannot reach {SSH_HOST} via SSH")
            print(f"  Verify: ssh {SSH_HOST} echo ok")
            sys.exit(1)
        print(f"  SSH OK: {SSH_HOST}")

        # 3. Sync worker script to DUT
        print("\n[3/4] Syncing worker script to DUT ...")
        if not sync_worker_to_dut():
            print(f"  FATAL: rsync to {SSH_HOST} failed")
            sys.exit(1)
        print(f"  Synced edge/custom/ → {SSH_HOST}:{REMOTE_PROJECT}/edge/custom/")

        # 4. Remote environment
        print("\n[4/4] Checking remote Python environment ...")
        if not check_remote_env():
            print(f"  FATAL: worker script not runnable on {SSH_HOST}")
            print(f"  Verify: ssh {SSH_HOST} "
                  f"'{REMOTE_ENV_ACTIVATE} && python3 {REMOTE_WORKER} --help'")
            sys.exit(1)
        print(f"  Remote environment OK")

    # ── Build condition list ──────────────────────────────────────────
    conditions = [
        (model, imgsz)
        for model in args.models
        for imgsz in args.resolutions
    ]

    total_reps = len(conditions) * args.reps
    total_time_h = total_reps * TOTAL_REP_S / 3600
    print(f"\n  Conditions : {len(conditions)}")
    print(f"  Reps each  : {args.reps}")
    print(f"  Total runs : {total_reps}")
    print(f"  Est. time  : {total_time_h:.1f} h "
          f"({WARMUP_S:.0f}s warm-up + {MEASURE_S:.0f}s measure per rep)")

    # ── Idle baseline ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Idle baseline")
    if args.dry_run:
        print("  [DRY] Would collect 30 s idle baseline")
    else:
        run_idle_baseline(transport, serial_port, ble_address,
                          force=args.force_idle)

    # ── Main loop ─────────────────────────────────────────────────────
    completed = 0
    failed = 0

    for model, imgsz in conditions:
        print(f"\n{'═' * 60}")
        print(f"  {model} @ {imgsz}px")
        print(f"{'═' * 60}")

        # Check existing reps
        done = existing_reps(model, imgsz)
        if args.skip_existing and len(done) >= args.reps:
            print(f"  SKIP: {len(done)}/{args.reps} reps exist")
            completed += args.reps
            continue

        for rep in range(args.reps):
            csv_path = power_csv_path(model, imgsz, rep)

            if args.skip_existing and csv_path.exists():
                print(f"  rep {rep:02d}: skip (exists)")
                completed += 1
                continue

            ok = run_one_condition(
                model, imgsz, rep,
                transport, serial_port, ble_address,
                dry_run=args.dry_run,
            )

            if ok:
                completed += 1
            else:
                failed += 1

        n_done = len(existing_reps(model, imgsz))
        print(f"  {model} {imgsz}px: {n_done}/{args.reps} reps complete")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DONE — {completed} completed, {failed} failed")
    if failed > 0:
        print(f"  Re-run with --no-skip-existing to retry failed conditions")
    print(f"  Power CSVs: {POWER_DIR}/")
    print(f"{'=' * 60}")

    # ── Aggregate power and generate table1_power CSV ─────────────
    if args.dry_run:
        return

    generate_power_table(conditions, args.reps)


# ---------------------------------------------------------------------------
# Post-collection: aggregate power data and merge with Table I
# ---------------------------------------------------------------------------

PROFILING_DIR = RESULTS_ROOT / "profiling"
RESULT_TAG = "arduino_uno_q"


def _model_stem(model_path: str) -> str:
    """Bare model name: yolo26n_640_ncnn_model → yolo26n, yolo26n → yolo26n."""
    stem = model_path.rsplit(".", 1)[0]
    for tag in ("_ncnn_model", "_hq", "_lq", "_qnn"):
        if stem.endswith(tag):
            stem = stem[: -len(tag)]
            break
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[-1].isdigit() else stem


def generate_power_table(conditions: list[tuple[str, int]], n_reps: int):
    """Aggregate per-rep power CSVs and merge with Table I profiling data.

    Mirrors the logic from notebook 04 cells 7–8.
    """
    import numpy as np
    import pandas as pd

    print(f"\n{'─' * 60}")
    print("Generating power table")

    # ── Load idle baseline ────────────────────────────────────────
    idle_csv = idle_csv_path()
    if not idle_csv.exists():
        print("  WARNING: no idle baseline — delta_W will be NaN")
        idle_mean = float("nan")
    else:
        idle_stats = summarise_readings(pd.read_csv(idle_csv), trim_s=TRIM_S)
        idle_mean = idle_stats["mean_W"]
        print(f"  Idle baseline: {idle_mean:.3f} W")

    # ── Per-condition aggregation ─────────────────────────────────
    power_rows = []

    for model, imgsz in conditions:
        rep_files = sorted(POWER_DIR.glob(f"{model}_{imgsz}_rep*.csv"))

        if not rep_files:
            print(f"  WARNING: no power data for {model} {imgsz}px")
            continue

        run_summaries = []
        for csv_path in rep_files:
            df = pd.read_csv(csv_path)
            df_meas = trim_warmup(df, warmup_s=WARMUP_S)
            stats = summarise_readings(df_meas, trim_s=TRIM_S)
            run_summaries.append(stats)

        agg = summarise_across_runs(run_summaries)

        power_rows.append({
            "stem":     model,
            "imgsz":    imgsz,
            "mean_W":   agg["mean_W"],
            "ci95_W":   agg["ci95_W"],
            "std_W":    agg["std_W"],
            "median_W": agg["median_W"],
            "iqr_W":    agg["iqr_W"],
            "peak_W":   agg["peak_W"],
            "delta_W":  round(agg["mean_W"] - idle_mean, 4),
            "n_runs":   agg["n_runs"],
        })

    power_df = pd.DataFrame(power_rows)
    print(f"\n  Power profile ({len(power_rows)} conditions, {n_reps} reps):\n")
    print(power_df[["model", "imgsz", "mean_W", "ci95_W", "median_W", "delta_W", "n_runs"]]
          .to_string(index=False))

    # ── Merge with Table I ────────────────────────────────────────
    table1_path = PROFILING_DIR / f"table1_profiling_{RESULT_TAG}.csv"

    if not table1_path.exists():
        print(f"\n  WARNING: {table1_path.name} not found — saving power-only table")
        out_path = PROFILING_DIR / f"table1_power_{RESULT_TAG}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        power_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        return

    table1 = pd.read_csv(table1_path)
    table1["stem"] = table1["model"].apply(_model_stem)

    # Aggregate timing/MOT across sequences (power is per-condition, not per-seq)
    timing_agg = (
        table1.groupby(["stem", "imgsz"])
        .agg(fps=("fps", "mean"), mota=("mota", "mean"), idf1=("idf1", "mean"))
        .reset_index()
        .round({"fps": 1, "mota": 3, "idf1": 3})
    )

    merged = timing_agg.merge(power_df, on=["stem", "imgsz"], how="left")
    merged["fps_per_W"] = (merged["fps"] / merged["mean_W"]).round(2)

    out_path = PROFILING_DIR / f"table1_power_{RESULT_TAG}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"\n  Merged table saved: {out_path}\n")
    print(merged[["stem", "imgsz", "fps", "mota", "mean_W", "ci95_W", "delta_W", "fps_per_W", "n_runs"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
