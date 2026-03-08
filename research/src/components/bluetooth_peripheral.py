"""
BLE GATT peripheral for the ADA driver monitoring system.

Advertises service A1B2C3D4-E5F6-7890-ABCD-1234567890AB and exposes
five characteristics that match BluetoothManager.swift on the iOS side:

  0001  realtime   read + notify   BLERealtimeData  (compact telemetry)
  0002  settings   write           BLESettingsData  (feature toggles)
  0003  buzzer     write           buzzer command
  0004  trip       read + notify   BLETripData      (session stats)
  0005  relay      notify          BLERelayData     (full Supabase records)

Requires: bless  (pip install bless)
BlueZ must be running on the Pi: sudo systemctl enable --now bluetooth
"""

import asyncio
import json
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
TRIP_UUID     = "A1B2C3D4-E5F6-7890-ABCD-123456780004"
RELAY_UUID    = "A1B2C3D4-E5F6-7890-ABCD-123456780005"

_READ_NOTIFY = None  # filled in _serve() once bless is confirmed available


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

        # Trip-level accumulators (reset on each start())
        self._trip_start = time.time()
        self._trip_id = f"{vehicle_id}-{int(self._trip_start)}"
        self._max_spd = 0
        self._spd_sum = 0
        self._spd_samples = 0
        self._spd_events = 0
        self._ph_events = 0
        self._drw_events = 0
        self._ix_max = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ble-peripheral"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3.0)

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
        await server.add_new_characteristic(
            SERVICE_UUID, TRIP_UUID, rn, _enc(self._trip_payload()), rd
        )
        await server.add_new_characteristic(
            SERVICE_UUID, RELAY_UUID,
            GATTCharacteristicProperties.notify,
            _enc(self._relay_payload()), rd
        )

        await server.start()
        print(f"[BLE] Advertising as ADA-{self.vehicle_id}", flush=True)

        tick = 0
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)
            self._update_trip_stats()

            # Push realtime every 0.5 s
            char = server.get_characteristic(REALTIME_UUID)
            if char is not None:
                char.value = _enc(self._realtime_payload())
                server.update_value(SERVICE_UUID, REALTIME_UUID)

            # Push trip + relay every 2 s
            if tick % 4 == 0:
                char = server.get_characteristic(TRIP_UUID)
                if char is not None:
                    char.value = _enc(self._trip_payload())
                    server.update_value(SERVICE_UUID, TRIP_UUID)

                char = server.get_characteristic(RELAY_UUID)
                if char is not None:
                    char.value = _enc(self._relay_payload())
                    server.update_value(SERVICE_UUID, RELAY_UUID)

            tick += 1

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

    def _trip_payload(self) -> dict:
        elapsed = int(time.time() - self._trip_start)
        avg = (self._spd_sum / self._spd_samples) if self._spd_samples else 0.0
        return {
            "tid":    self._trip_id,
            "dur":    elapsed,
            "mx_spd": self._max_spd,
            "avg_spd": round(avg, 1),
            "spd_ev": self._spd_events,
            "drw_ev": self._drw_events,
            "ph_ev":  self._ph_events,
            "ix_max": self._ix_max,
        }

    def _relay_payload(self) -> dict:
        with self._lock:
            rt = {
                "vehicle_id":        self.vehicle_id,
                "speed_mph":         int(self._shared.get("speed_mph", 0)),
                "speed_limit_mph":   int(self._shared.get("speed_limit") or 0),
                "heading_degrees":   int(self._shared.get("heading_degrees", 0)),
                "compass_direction": str(self._shared.get("compass_direction", "N")),
                "is_speeding":       bool(self._shared.get("is_speeding", False)),
                "is_moving":         bool(self._shared.get("speed_mph", 0) > 0),
                "driver_status":     str(self._shared.get("driver_status", "alert")),
                "intoxication_score": int(self._shared.get("intoxication_score", 0)),
                "satellites":        int(self._shared.get("satellites", 0) or 0),
                "is_phone_detected":    bool(self._shared.get("is_phone_detected", False)),
                "is_drinking_detected": bool(self._shared.get("is_drinking_detected", False)),
                "latitude":  float(self._shared.get("latitude", 0.0)),
                "longitude": float(self._shared.get("longitude", 0.0)),
            }
        return {"rt": rt, "trip": None}

    def _update_trip_stats(self) -> None:
        with self._lock:
            spd  = int(self._shared.get("speed_mph", 0))
            ph   = bool(self._shared.get("is_phone_detected", False))
            dr   = bool(self._shared.get("is_drinking_detected", False))
            ix   = int(self._shared.get("intoxication_score", 0))
            sp   = bool(self._shared.get("is_speeding", False))

        self._max_spd = max(self._max_spd, spd)
        self._spd_sum += spd
        self._spd_samples += 1
        self._ix_max = max(self._ix_max, ix)
        if sp:  self._spd_events += 1
        if ph:  self._ph_events  += 1
        if dr:  self._drw_events += 1
