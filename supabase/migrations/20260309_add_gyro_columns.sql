-- Add IMU/gyroscope columns to vehicle_realtime.
-- gyro_x/y/z: rotation rates in deg/s (from Pi gyro reader)
-- acc_mag: accelerometer magnitude in g (useful for hard-braking detection)
-- All nullable so existing rows and old Pi firmware remain compatible.

ALTER TABLE public.vehicle_realtime
  ADD COLUMN IF NOT EXISTS gyro_x   double precision,
  ADD COLUMN IF NOT EXISTS gyro_y   double precision,
  ADD COLUMN IF NOT EXISTS gyro_z   double precision,
  ADD COLUMN IF NOT EXISTS acc_mag  double precision;
