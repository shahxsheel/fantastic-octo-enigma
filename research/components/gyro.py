import os
import struct
import threading
import time
import math

# cSpell:disable

DEFAULT_PORT = "/dev/ttyACM0"
BAUD = 115200
PRINT_INTERVAL = 0.2


class GyroReader:
    """Reads acc/gyro from serial; exposes thread-safe get_latest()."""

    def __init__(self, port=None, baud=BAUD, print_interval=PRINT_INTERVAL):
        self.port = port or os.environ.get("GYRO_SERIAL_PORT", DEFAULT_PORT)
        self.baud = baud
        self.print_interval = print_interval
        self._lock = threading.Lock()
        self._ser = None
        self._thread = None
        self._running = False
        # Latest values (None until first packet)
        self._last_acc_mag = None
        self._last_acc_delta = None
        self._last_gyro_mag = None
        self._last_gyrox = None
        self._last_gyroy = None
        self._last_gyroz = None
        self._last_ts = None
        self._measurement_count = 0
        self._prev_acc_mag = None
        self._last_print = time.time()
        self._print_from_loop = False

    def start(self, print_from_loop=False):
        """Open serial and start the read loop in a daemon thread."""
        import serial
        self._ser = serial.Serial(self.port, self.baud, timeout=0)
        self._running = True
        self._print_from_loop = print_from_loop
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the read loop and close serial."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def get_latest(self):
        """
        Return latest reading or None if no data yet.
        Returns a dict: acc_mag, acc_delta, gyro_mag, gyrox, gyroy, gyroz, timestamp.
        """
        with self._lock:
            if self._last_ts is None:
                return None
            return {
                "acc_mag": self._last_acc_mag,
                "acc_delta": self._last_acc_delta,
                "gyro_mag": self._last_gyro_mag,
                "gyrox": self._last_gyrox,
                "gyroy": self._last_gyroy,
                "gyroz": self._last_gyroz,
                "timestamp": self._last_ts,
            }

    def _read_loop(self):
        buffer = bytearray()
        while self._running and self._ser and self._ser.is_open:
            try:
                buffer += self._ser.read(self._ser.in_waiting or 1)
            except Exception:
                break
            while len(buffer) >= 15 and self._running:
                if buffer[0] != 0xAA or buffer[1] != 0x55:
                    buffer.pop(0)
                    continue
                payload = buffer[2:14]
                checksum = buffer[14]
                calc_checksum = 0
                for b in payload:
                    calc_checksum ^= b
                if calc_checksum != checksum:
                    buffer.pop(0)
                    continue
                accx_i, accy_i, accz_i, gyrox_i, gyroy_i, gyroz_i = struct.unpack(
                    "<hhhhhh", payload
                )
                accx = accx_i / 1000
                accy = accy_i / 1000
                accz = accz_i / 1000
                gyrox = gyrox_i / 100
                gyroy = gyroy_i / 100
                gyroz = gyroz_i / 100
                acc_mag = math.sqrt(accx**2 + accy**2 + accz**2)
                if self._prev_acc_mag is None:
                    acc_delta = 0.0
                else:
                    acc_delta = acc_mag - self._prev_acc_mag
                self._prev_acc_mag = acc_mag
                gyro_mag = math.sqrt(gyrox**2 + gyroy**2 + gyroz**2)
                self._measurement_count += 1
                buffer = buffer[15:]

                with self._lock:
                    self._last_acc_mag = acc_mag
                    self._last_acc_delta = acc_delta
                    self._last_gyro_mag = gyro_mag
                    self._last_gyrox = gyrox
                    self._last_gyroy = gyroy
                    self._last_gyroz = gyroz
                    self._last_ts = time.time()

                now = time.time()
                if self._print_from_loop and now - self._last_print >= self.print_interval:
                    if self._measurement_count > 0:
                        print(
                            f"Acceleration magnitude: {acc_mag:.3f} g | Δg: {acc_delta:.3f} g | "
                            f"Gyro: {gyro_mag:.2f} deg/s | M: {self._measurement_count}"
                        )
                    self._measurement_count = 0
                    self._last_print = now
            time.sleep(0.001)


if __name__ == "__main__":
    import serial  # noqa: F401 - used by GyroReader.start()
    reader = GyroReader()
    reader.start(print_from_loop=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
