# ls -l /dev/serial0
# should show: /dev/serial0 -> ttyS0 or ttyAMA0

# Make sure bluetooth isn't hogging it (Pi 4):
# sudo nano /boot/firmware/config.txt
# Add:
# dtoverlay=disable-bt

# sudo systemctl disable hciuart
# sudo reboot

import serial
import pynmea2
import threading
import time


class GPSReader:
    """GPS module reader for Raspberry Pi with fallback to fake coordinates"""

    # Apple Park coordinates (fallback)
    APPLE_PARK_LAT = 37.3349
    APPLE_PARK_LON = -122.0090

    def __init__(self, port='/dev/serial0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.thread = None
        self.use_fake_data = False

        # GPS data
        self._data = {
            'satellites': 0,
            'speed_mph': 0.0,
            'heading': 0.0,
            'latitude': self.APPLE_PARK_LAT,
            'longitude': self.APPLE_PARK_LON,
            'has_fix': False,
        }
        self._lock = threading.Lock()

    def _compass_direction(self, degrees):
        """Convert degrees to compass direction (e.g., 342NW)"""
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((degrees + 22.5) % 360 // 45)
        return f"{int(degrees)}{dirs[idx]}"

    def start(self):
        """Start reading GPS data in background thread"""
        try:
            self.serial = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
            self.use_fake_data = False
            print(f"GPS connected on {self.port}")
        except Exception as e:
            print(f"GPS unavailable ({e}), using fake Apple Park coordinates")
            self.use_fake_data = True
            self._set_fake_data()

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the GPS reader"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.serial:
            self.serial.close()

    def _set_fake_data(self):
        """Set fake Apple Park data with simulated movement"""
        with self._lock:
            self._data['latitude'] = self.APPLE_PARK_LAT
            self._data['longitude'] = self.APPLE_PARK_LON
            self._data['satellites'] = 0
            self._data['speed_mph'] = 0.0
            self._data['heading'] = 0.0
            self._data['has_fix'] = False

    def _read_loop(self):
        """Background thread to read GPS data"""
        while self.running:
            if self.use_fake_data:
                time.sleep(0.5)
                continue

            try:
                line = self.serial.readline().decode('ascii', errors='replace').strip()
                if not line:
                    continue

                if line.startswith('$GPGGA'):
                    msg = pynmea2.parse(line)
                    with self._lock:
                        self._data['satellites'] = int(msg.num_sats or 0)
                        if msg.latitude and msg.longitude:
                            self._data['latitude'] = msg.latitude
                            self._data['longitude'] = msg.longitude
                            self._data['has_fix'] = True
                        else:
                            self._data['has_fix'] = False

                elif line.startswith('$GPRMC'):
                    msg = pynmea2.parse(line)
                    with self._lock:
                        # Convert knots to mph (1 knot = 1.151 mph)
                        self._data['speed_mph'] = (msg.spd_over_grnd or 0) * 1.151
                        self._data['heading'] = msg.true_course or 0

            except Exception:
                pass

    @property
    def satellites(self):
        """Number of satellites in view"""
        with self._lock:
            return self._data['satellites']

    @property
    def speed_mph(self):
        """Current speed in mph"""
        with self._lock:
            return self._data['speed_mph']

    @property
    def heading(self):
        """Current heading in degrees (0-360)"""
        with self._lock:
            return self._data['heading']

    @property
    def direction(self):
        """Compass direction string (e.g., '342NW')"""
        with self._lock:
            return self._compass_direction(self._data['heading'])

    @property
    def compass_direction(self):
        """Just the compass letters (e.g., 'NW')"""
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        with self._lock:
            idx = int((self._data['heading'] + 22.5) % 360 // 45)
            return dirs[idx]

    @property
    def latitude(self):
        """Current latitude"""
        with self._lock:
            return self._data['latitude']

    @property
    def longitude(self):
        """Current longitude"""
        with self._lock:
            return self._data['longitude']

    @property
    def has_fix(self):
        """Whether GPS has a valid fix"""
        with self._lock:
            return self._data['has_fix']

    @property
    def is_fake(self):
        """Whether using fake data (no GPS available)"""
        return self.use_fake_data

    def get_all(self):
        """Get all GPS data as a dictionary"""
        with self._lock:
            return {
                'satellites': self._data['satellites'],
                'speed_mph': self._data['speed_mph'],
                'heading': self._data['heading'],
                'direction': self._compass_direction(self._data['heading']),
                'compass_direction': self.compass_direction,
                'latitude': self._data['latitude'],
                'longitude': self._data['longitude'],
                'has_fix': self._data['has_fix'],
                'is_fake': self.use_fake_data,
            }


# For standalone testing
if __name__ == "__main__":
    gps = GPSReader()
    gps.start()

    try:
        while True:
            data = gps.get_all()
            print(f"Sats: {data['satellites']} | "
                  f"Speed: {data['speed_mph']:.1f} mph | "
                  f"Heading: {data['direction']} | "
                  f"Lat: {data['latitude']:.6f} | "
                  f"Lon: {data['longitude']:.6f} | "
                  f"Fix: {data['has_fix']} | "
                  f"Fake: {data['is_fake']}")
            time.sleep(1)
    except KeyboardInterrupt:
        gps.stop()
        print("\nGPS stopped")
