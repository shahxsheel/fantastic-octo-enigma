-- Add drowsiness flag for Pi telemetry compatibility.
-- Non-destructive and safe to run multiple times.

alter table if exists public.vehicle_realtime
  add column if not exists is_drowsy boolean default false;

-- Ensure PostgREST sees the new column immediately.
notify pgrst, 'reload schema';
