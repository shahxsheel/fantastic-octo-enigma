"""
BLE GATT peripheral for the ADA driver monitoring system.

Advertises service A1B2C3D4-E5F6-7890-ABCD-1234567890AB and exposes
three characteristics that match BluetoothManager.swift on the iOS side:

  0001  realtime   read + notify   BLERealtimeData  (compact telemetry)
  0002  settings   write           BLESettingsData  (feature toggles)
  0003  buzzer     write           buzzer command

BLE is the offline fallback transport. The Pi starts advertising only
after Supabase has been unreachable for BLE_FALLBACK_SEC seconds
(default 30). When Supabase recovers the Pi stops advertising.

Requires: bless  (pip install bless)
BlueZ must be running on the Pi: sudo systemctl enable --now bluetooth
"""

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

_BLESS_IMPORT_ERROR = ""
try:
    from bless import (
        BlessServer,
        GATTAttributePermissions,
        GATTCharacteristicProperties,
    )

    _BLESS_AVAILABLE = True
except Exception as _bless_err:
    _BLESS_AVAILABLE = False
    _BLESS_IMPORT_ERROR = str(_bless_err)

# ── UUIDs — must match BluetoothManager.swift ──────────────────────────────
SERVICE_UUID  = "A1B2C3D4-E5F6-7890-ABCD-1234567890AB"
REALTIME_UUID = "A1B2C3D4-E5F6-7890-ABCD-123456780001"
SETTINGS_UUID = "A1B2C3D4-E5F6-7890-ABCD-123456780002"
BUZZER_UUID   = "A1B2C3D4-E5F6-7890-ABCD-123456780003"


def _enc(obj: dict) -> bytearray:
    return bytearray(json.dumps(obj, separators=(",", ":")).encode())


class BluetoothPeripheral:
    """
    Runs a BLE GATT server in a dedicated daemon thread.

    Parameters
    ----------
    vehicle_id      : value of VEHICLE_ID env var (e.g. "VEHICLE-001")
    shared_results  : the _shared_results dict from run_single_usb.py
    lock            : the _lock that protects _shared_results
    on_settings     : called with parsed dict when iOS writes settings char
    on_buzzer       : called with parsed dict when iOS writes buzzer char
    """

    def __init__(
        self,
        vehicle_id: str,
        shared_results: dict,
        lock: threading.Lock,
        on_settings: Optional[Callable[[dict], None]] = None,
        on_buzzer: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.vehicle_id = vehicle_id
        self._shared = shared_results
        self._lock = lock
        self._on_settings = on_settings
        self._on_buzzer = on_buzzer

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None

        self.enabled = _BLESS_AVAILABLE
        self.reason = "ok" if _BLESS_AVAILABLE else _BLESS_IMPORT_ERROR
        self._notify_max_bytes = max(64, int(os.environ.get("BLE_NOTIFY_MAX_BYTES", "180")))
        self._notify_debug = os.environ.get("BLE_NOTIFY_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return  # Already running
        # Reset async state for potential re-start after stop()
        self._loop = None
        self._stop_event = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ble-peripheral"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None

    # ── Thread entry ────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[BLE] Fatal: {e}", flush=True)
            self.enabled = False
            self.reason = str(e)
        finally:
            self._loop.close()

    # ── GATT server ─────────────────────────────────────────────────────────

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        self._log_startup_diagnostics()

        server = BlessServer(name=f"ADA-{self.vehicle_id}", loop=self._loop)
        server.read_request_func = self._read_request
        server.write_request_func = self._write_request

        await server.add_new_service(SERVICE_UUID)

        rn = (
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.notify
        )
        rd = GATTAttributePermissions.readable
        wr = GATTAttributePermissions.writeable

        await server.add_new_characteristic(
            SERVICE_UUID, REALTIME_UUID, rn, _enc(self._realtime_payload()), rd
        )
        await server.add_new_characteristic(
            SERVICE_UUID, SETTINGS_UUID,
            GATTCharacteristicProperties.write, None, wr
        )
        await server.add_new_characteristic(
            SERVICE_UUID, BUZZER_UUID,
            GATTCharacteristicProperties.write, None, wr
        )

        await server.start()
        print(
            f"[BLE] Advertising as ADA-{self.vehicle_id} "
            f"(notify_max={self._notify_max_bytes}B, mode=connectable)",
            flush=True,
        )

        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)

            # Push realtime every 0.5 s
            self._safe_notify(
                server,
                REALTIME_UUID,
                self._realtime_payload(),
                "realtime",
            )

        await server.stop()
        print("[BLE] Stopped.", flush=True)

    # ── GATT callbacks ──────────────────────────────────────────────────────

    def _read_request(self, char, **kwargs):
        return char.value

    def _write_request(self, char, value: bytearray, **kwargs) -> None:
        try:
            data = json.loads(bytes(value).decode("utf-8"))
        except Exception:
            return
        uuid_upper = str(char.uuid).upper()
        if SETTINGS_UUID in uuid_upper and self._on_settings:
            self._on_settings(data)
        elif BUZZER_UUID in uuid_upper and self._on_buzzer:
            self._on_buzzer(data)

    # ── Payload builders ────────────────────────────────────────────────────

    def _realtime_payload(self) -> dict:
        with self._lock:
            return {
                "spd": int(self._shared.get("speed_mph", 0)),
                "hdg": int(self._shared.get("heading_degrees", 0)),
                "lat": float(self._shared.get("latitude", 0.0)),
                "lng": float(self._shared.get("longitude", 0.0)),
                "dir": str(self._shared.get("compass_direction", "N")),
                "ds":  str(self._shared.get("driver_status", "alert")),
                "ph":  bool(self._shared.get("is_phone_detected", False)),
                "dr":  bool(self._shared.get("is_drinking_detected", False)),
                "ix":  int(self._shared.get("intoxication_score", 0)),
                "sp":  bool(self._shared.get("is_speeding", False)),
                "sat": int(self._shared.get("satellites", 0) or 0),
            }

    def _safe_notify(self, server: "BlessServer", uuid: str, payload: dict, label: str) -> None:
        """Best-effort notify with guardrails so a bad packet never destabilizes BLE."""
        char = server.get_characteristic(uuid)
        if char is None:
            if self._notify_debug:
                print(f"[BLE][debug] skip notify {label}: characteristic missing", flush=True)
            return

        try:
            encoded = _enc(payload)
        except Exception as e:
            print(f"[BLE] notify {label} encode failed: {e}", flush=True)
            return

        size = len(encoded)
        if self._notify_debug:
            print(f"[BLE][debug] notify {label}: {size}B", flush=True)

        if size > self._notify_max_bytes:
            print(
                f"[BLE] skip notify {label}: {size}B exceeds BLE_NOTIFY_MAX_BYTES={self._notify_max_bytes}",
                flush=True,
            )
            return

        try:
            char.value = encoded
            server.update_value(SERVICE_UUID, uuid)
        except Exception as e:
            print(f"[BLE] notify {label} failed ({size}B): {e}", flush=True)

    def _log_startup_diagnostics(self) -> None:
        adapter = os.environ.get("BLE_ADAPTER", "hci0")
        print(f"[BLE][diag] adapter={adapter} expected_connectable=true", flush=True)

        bluetoothctl = shutil.which("bluetoothctl")
        if bluetoothctl:
            try:
                proc = subprocess.run(
                    [bluetoothctl, "show"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if proc.returncode == 0:
                    powered = self._extract_show_value(proc.stdout, "Powered:")
                    discoverable = self._extract_show_value(proc.stdout, "Discoverable:")
                    pairable = self._extract_show_value(proc.stdout, "Pairable:")
                    print(
                        "[BLE][diag] bluetoothctl"
                        f" powered={powered or 'unknown'}"
                        f" discoverable={discoverable or 'unknown'}"
                        f" pairable={pairable or 'unknown'}",
                        flush=True,
                    )
                else:
                    print(f"[BLE][diag] bluetoothctl show failed rc={proc.returncode}", flush=True)
            except Exception as e:
                print(f"[BLE][diag] bluetoothctl show error={e}", flush=True)
        else:
            print("[BLE][diag] bluetoothctl unavailable", flush=True)

        hciconfig = shutil.which("hciconfig")
        if hciconfig:
            try:
                proc = subprocess.run(
                    [hciconfig, adapter],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if proc.returncode == 0:
                    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
                    adapter_state = lines[1] if len(lines) > 1 else (lines[0] if lines else "unknown")
                    print(f"[BLE][diag] {adapter} state={adapter_state}", flush=True)
                else:
                    print(
                        f"[BLE][diag] hciconfig {adapter} failed rc={proc.returncode}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[BLE][diag] hciconfig error={e}", flush=True)
        else:
            print("[BLE][diag] hciconfig unavailable", flush=True)

    @staticmethod
    def _extract_show_value(output: str, prefix: str) -> Optional[str]:
        for line in output.splitlines():
            text = line.strip()
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return None
