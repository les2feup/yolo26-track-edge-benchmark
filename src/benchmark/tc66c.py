"""
JT-TC66C USB power meter — BLE data acquisition module.

Protocol: 192-byte AES-ECB encrypted payload, split into 3×64-byte blocks
(pac1, pac2, pac3).  All multi-byte fields are little-endian uint32.

Reference: sigrok RDTech TC66C wiki, Ralim/TC66C, skgsergio/tc66c-toolkit.
"""

from __future__ import annotations

import asyncio
import csv
import struct
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from Crypto.Cipher import AES

# ── BLE identifiers ─────────────────────────────────────────────────────────
# Primary service present on all known TC66C revisions
SERVICE_FFE0 = "0000ffe0-0000-1000-8000-00805f9b34fb"

# Characteristic layout varies by hardware revision:
#   Rev A (Dialog Semi): ffe1 = read/write/notify combo, ffe2 = write-only
#   Rev B (Ralim docs):  ffe9 = write, ffe4 = notify  (under service ffe5)
# We auto-detect at connect time — see _resolve_chars().
_KNOWN_NOTIFY = ("0000ffe1-0000-1000-8000-00805f9b34fb",
                 "0000ffe4-0000-1000-8000-00805f9b34fb")
_KNOWN_WRITE  = ("0000ffe1-0000-1000-8000-00805f9b34fb",
                 "0000ffe2-0000-1000-8000-00805f9b34fb",
                 "0000ffe9-0000-1000-8000-00805f9b34fb")

# Service UUIDs used for scan matching
SERVICE_FFE5 = "0000ffe5-0000-1000-8000-00805f9b34fb"
SERVICE_NUS = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
_TC66C_SERVICES = {SERVICE_FFE0, SERVICE_FFE5, SERVICE_NUS}

# Known broadcast name prefix
_TC66C_NAME = "TC66C"

# ── AES-256-ECB static key (extracted from manufacturer Android app) ─────────
# Two keys exist in the wild:
#   16-byte "HelloAliens!areu" — older TC66C revisions (Ralim docs)
#   32-byte key below          — current TC66C v1.17+ (skgsergio/tc66c-toolkit)
# The 32-byte key is AES-256 and works on all known hardware revisions.
_AES_KEY = bytes([
    0x58, 0x21, 0xFA, 0x56, 0x01, 0xB2, 0xF0, 0x26,
    0x87, 0xFF, 0x12, 0x04, 0x62, 0x2A, 0x4F, 0xB0,
    0x86, 0xF4, 0x02, 0x60, 0x81, 0x6F, 0x9A, 0x0B,
    0xA7, 0xF1, 0x06, 0x61, 0x9A, 0xB8, 0x72, 0x88,
])
_CIPHER = AES.new(_AES_KEY, AES.MODE_ECB)

# ── Protocol constants ──────────────────────────────────────────────────────
BLOCK_SIZE = 64
TOTAL_PAYLOAD = 3 * BLOCK_SIZE  # 192 bytes
CMD_GETVA = b"bgetva\r\n"

# ── Expected block headers after decryption ─────────────────────────────────
_HEADERS = (b"pac1", b"pac2", b"pac3")


# ── Measurement dataclass ───────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TC66CReading:
    """Single measurement snapshot from the TC66C."""

    timestamp: float       # time.time() epoch seconds
    voltage_V: float       # Volts
    current_A: float       # Amperes
    power_W: float         # Watts
    resistance_ohm: float  # Ohms
    temp_C: float          # Celsius (signed)
    dp_V: float            # D+ line voltage
    dm_V: float            # D- line voltage
    energy0_mWh: int       # Group 0 accumulated energy
    energy1_mWh: int       # Group 1 accumulated energy

    @property
    def csv_header(self) -> list[str]:
        return [f.name for f in fields(self)]

    @property
    def csv_row(self) -> list[str]:
        return [str(getattr(self, f.name)) for f in fields(self)]


# ── Packet decryption ───────────────────────────────────────────────────────
def decrypt_payload(raw: bytes) -> bytes:
    """Decrypt a 192-byte AES-ECB payload from the TC66C.

    Each 64-byte block is decrypted independently (ECB mode processes
    16-byte sub-blocks).  Returns the concatenated plaintext.
    """
    if len(raw) != TOTAL_PAYLOAD:
        raise ValueError(f"expected {TOTAL_PAYLOAD} bytes, got {len(raw)}")
    return _CIPHER.decrypt(raw)


def verify_headers(plain: bytes) -> bool:
    """Check that decrypted blocks start with pac1/pac2/pac3."""
    for i, hdr in enumerate(_HEADERS):
        if plain[i * BLOCK_SIZE : i * BLOCK_SIZE + 4] != hdr:
            return False
    return True


# ── Packet parsing ──────────────────────────────────────────────────────────
def parse_reading(plain: bytes, ts: float | None = None) -> TC66CReading:
    """Extract measurements from a decrypted 192-byte payload.

    Byte offsets follow the sigrok RDTech TC66C specification:
    PAC1 (block 0): voltage @48, current @52, power @56
    PAC2 (block 1): resistance @4, grp0 cap @8, grp0 energy @12,
                     grp1 cap @16, grp1 energy @20, temp_sign @24,
                     temp @28, D+ @32, D- @36
    """
    if ts is None:
        ts = time.time()

    # PAC1 fields — absolute offsets in the 192-byte buffer
    voltage_raw, current_raw, power_raw = struct.unpack_from("<III", plain, 48)

    # PAC2 fields — block 1 starts at offset 64
    b2 = 64
    (
        resistance_raw,
        _cap0, energy0,
        _cap1, energy1,
        temp_sign, temp_raw,
        dp_raw, dm_raw,
    ) = struct.unpack_from("<IIIIIIIII", plain, b2 + 4)

    temp_c = float(temp_raw)
    if temp_sign == 1:
        temp_c = -temp_c

    return TC66CReading(
        timestamp=ts,
        voltage_V=voltage_raw * 1e-4,
        current_A=current_raw * 1e-5,
        power_W=power_raw * 1e-4,
        resistance_ohm=resistance_raw * 1e-2,
        temp_C=temp_c,
        dp_V=dp_raw * 1e-2,
        dm_V=dm_raw * 1e-2,
        energy0_mWh=energy0,
        energy1_mWh=energy1,
    )


# ── BLE acquisition loop ────────────────────────────────────────────────────
async def scan_for_tc66c(timeout: float = 10.0) -> str:
    """Active-scan BLE and return the address of the first TC66C found.

    Matches by broadcast name (TC66C) or by advertised service UUIDs
    (FFE0/FFE5/Nordic UART).  Uses active scanning so the scan-response
    packet (which carries the device name) is requested from peripherals.
    """
    from bleak import BleakScanner

    # Active scan requests scan-response packets that carry the device name
    devices = await BleakScanner.discover(
        timeout=timeout,
        return_adv=True,
        scanning_mode="active",
    )
    for addr, (dev, adv) in devices.items():
        name = dev.name or adv.local_name or ""
        adv_uuids = set(adv.service_uuids or [])

        # Match by name
        if _TC66C_NAME in name.upper():
            return addr

        # Match by advertised service UUID
        if adv_uuids & _TC66C_SERVICES:
            return addr

    raise RuntimeError(
        "TC66C not found.  Checklist:\n"
        "  1. TC66C BT menu is ON (Menu 6)\n"
        "  2. No phone/app is currently connected to it\n"
        "  3. On Linux, run with sudo (raw HCI requires CAP_NET_ADMIN)\n"
        "  4. Device is within BLE range (~10 m)"
    )


def _resolve_chars(client) -> tuple[str, str]:
    """Auto-detect write and notify characteristic UUIDs from the live GATT table.

    Returns (char_write_uuid, char_notify_uuid).  Raises RuntimeError if
    no matching characteristics are found.
    """
    char_notify = None
    char_write = None

    for svc in client.services:
        for char in svc.characteristics:
            uuid = char.uuid.lower()
            if uuid in _KNOWN_NOTIFY and "notify" in char.properties:
                char_notify = uuid
            if uuid in _KNOWN_WRITE and ("write" in char.properties
                                          or "write-without-response" in char.properties):
                char_write = uuid

    if char_notify is None or char_write is None:
        raise RuntimeError(
            f"Could not resolve TC66C characteristics.  "
            f"Found notify={char_notify}, write={char_write}"
        )
    return char_write, char_notify


async def collect(
    address: str,
    duration_s: float = 60.0,
    interval_s: float = 1.0,
    csv_path: Path | None = None,
    on_reading: Callable[[TC66CReading], None] | None = None,
    stop_event: asyncio.Event | None = None,
) -> list[TC66CReading]:
    """Connect to TC66C and poll measurements for *duration_s* seconds.

    Parameters
    ----------
    address : str
        BLE MAC address (Linux/Windows) or UUID (macOS).
    duration_s : float
        Total collection window.  Ignored when *stop_event* is provided.
    interval_s : float
        Seconds between successive poll commands.
    csv_path : Path, optional
        If given, rows are appended in real time.
    on_reading : callable, optional
        Callback invoked with each parsed reading (e.g. for live display).
    stop_event : asyncio.Event, optional
        External stop signal; overrides *duration_s*.

    Returns
    -------
    list[TC66CReading]
        All collected readings.
    """
    from bleak import BleakClient, BleakError

    readings: list[TC66CReading] = []
    buf = bytearray()

    # Notification callback accumulates 64-byte BLE chunks into 192-byte payloads
    ready = asyncio.Event()

    def _on_notify(_sender, data: bytearray):
        buf.extend(data)
        if len(buf) >= TOTAL_PAYLOAD:
            ready.set()

    csv_file = None
    writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)
        # Write header only if file is empty
        if csv_path.stat().st_size == 0:
            writer.writerow(TC66CReading.__dataclass_fields__.keys())

    MAX_RECONNECTS = 5
    reconnects = 0
    t_start = time.monotonic()

    try:
        while reconnects <= MAX_RECONNECTS:
            try:
                async with BleakClient(address) as client:
                    char_write, char_notify = _resolve_chars(client)
                    await client.start_notify(char_notify, _on_notify)
                    if reconnects > 0:
                        print(f"[tc66c] reconnected ({reconnects}/{MAX_RECONNECTS})")

                    while True:
                        if stop_event is not None and stop_event.is_set():
                            break
                        if stop_event is None and (time.monotonic() - t_start) >= duration_s:
                            break

                        buf.clear()
                        ready.clear()
                        await client.write_gatt_char(char_write, CMD_GETVA)

                        try:
                            await asyncio.wait_for(ready.wait(), timeout=interval_s + 2.0)
                        except asyncio.TimeoutError:
                            continue

                        ts = time.time()
                        raw = bytes(buf[:TOTAL_PAYLOAD])
                        plain = decrypt_payload(raw)

                        if not verify_headers(plain):
                            continue

                        reading = parse_reading(plain, ts=ts)
                        readings.append(reading)

                        if writer is not None:
                            writer.writerow(reading.csv_row)
                            csv_file.flush()

                        if on_reading is not None:
                            on_reading(reading)

                        await asyncio.sleep(interval_s)

                    # Clean exit from the polling loop — done
                    try:
                        await client.stop_notify(char_notify)
                    except (EOFError, OSError, Exception):
                        pass
                    break  # success, exit reconnect loop

            except (EOFError, OSError, BleakError) as exc:
                # BLE link dropped — reconnect if budget remains
                reconnects += 1
                if reconnects > MAX_RECONNECTS:
                    print(f"[tc66c] BLE connection lost, max reconnects exceeded: {exc}")
                    break
                print(f"[tc66c] BLE dropped ({exc.__class__.__name__}), "
                      f"reconnecting in 3 s ... ({reconnects}/{MAX_RECONNECTS})")
                await asyncio.sleep(3.0)
    finally:
        if csv_file is not None:
            csv_file.close()

    return readings


# ── Post-processing ──────────────────────────────────────────────────────────
def summarise_readings(df, trim_s: float = 5.0) -> dict:
    """Compute power statistics from a TC66C CSV DataFrame.

    Trims the first and last *trim_s* seconds to discard transients
    (device spin-up / inference wind-down).  Falls back to the full
    recording when the window is too short for trimming.

    Parameters
    ----------
    df : DataFrame
        Must contain ``timestamp`` (epoch-seconds float) and ``power_W``.
    trim_s : float
        Seconds to discard from each end of the recording.

    Returns
    -------
    dict with keys: mean_W, std_W, peak_W, min_W, n_samples
    """
    import pandas as pd

    ts = pd.to_datetime(df["timestamp"], unit="s")
    t0 = ts.iloc[0] + pd.Timedelta(seconds=trim_s)
    t1 = ts.iloc[-1] - pd.Timedelta(seconds=trim_s)
    trimmed = df.loc[(ts >= t0) & (ts <= t1), "power_W"]

    if trimmed.empty:
        trimmed = df["power_W"]

    return {
        "mean_W":    float(trimmed.mean()),
        "std_W":     float(trimmed.std(ddof=1)),
        "peak_W":    float(trimmed.max()),
        "min_W":     float(trimmed.min()),
        "n_samples": int(len(trimmed)),
    }
