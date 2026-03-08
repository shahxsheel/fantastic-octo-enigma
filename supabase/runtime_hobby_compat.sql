-- ================================================================
-- Hobby Runtime Compatibility Baseline (non-destructive)
-- Ensures required runtime contracts for iOS + Pi BLE relay paths:
--   public.vehicles
--   public.vehicle_realtime
--   public.vehicle_trips
--
-- This script intentionally avoids destructive cleanup of legacy tables.
-- ================================================================

create extension if not exists pgcrypto;

create table if not exists public.vehicles (
  id text primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  name text,
  description text,
  owner_id uuid references auth.users(id) on delete set null,
  enable_yolo boolean not null default true,
  enable_microphone boolean not null default true,
  enable_camera boolean not null default true,
  enable_dashcam boolean not null default false
);

alter table if exists public.vehicles add column if not exists created_at timestamptz not null default now();
alter table if exists public.vehicles add column if not exists updated_at timestamptz not null default now();
alter table if exists public.vehicles add column if not exists name text;
alter table if exists public.vehicles add column if not exists description text;
alter table if exists public.vehicles add column if not exists owner_id uuid references auth.users(id) on delete set null;
alter table if exists public.vehicles add column if not exists enable_yolo boolean not null default true;
alter table if exists public.vehicles add column if not exists enable_microphone boolean not null default true;
alter table if exists public.vehicles add column if not exists enable_camera boolean not null default true;
alter table if exists public.vehicles add column if not exists enable_dashcam boolean not null default false;

create table if not exists public.vehicle_realtime (
  vehicle_id text primary key references public.vehicles(id) on delete cascade,
  updated_at timestamptz not null default now(),
  latitude double precision,
  longitude double precision,
  speed_mph integer not null default 0,
  speed_limit_mph integer not null default 0,
  heading_degrees integer not null default 0,
  compass_direction text not null default 'N',
  is_speeding boolean not null default false,
  is_moving boolean not null default false,
  driver_status text not null default 'unknown',
  intoxication_score integer not null default 0,
  is_drowsy boolean default false,
  satellites integer,
  is_phone_detected boolean default false,
  is_drinking_detected boolean default false,
  buzzer_active boolean default false,
  buzzer_type text default 'alert',
  buzzer_updated_at timestamptz
);

alter table if exists public.vehicle_realtime add column if not exists updated_at timestamptz not null default now();
alter table if exists public.vehicle_realtime add column if not exists latitude double precision;
alter table if exists public.vehicle_realtime add column if not exists longitude double precision;
alter table if exists public.vehicle_realtime add column if not exists speed_mph integer not null default 0;
alter table if exists public.vehicle_realtime add column if not exists speed_limit_mph integer not null default 0;
alter table if exists public.vehicle_realtime add column if not exists heading_degrees integer not null default 0;
alter table if exists public.vehicle_realtime add column if not exists compass_direction text not null default 'N';
alter table if exists public.vehicle_realtime add column if not exists is_speeding boolean not null default false;
alter table if exists public.vehicle_realtime add column if not exists is_moving boolean not null default false;
alter table if exists public.vehicle_realtime add column if not exists driver_status text not null default 'unknown';
alter table if exists public.vehicle_realtime add column if not exists intoxication_score integer not null default 0;
alter table if exists public.vehicle_realtime add column if not exists is_drowsy boolean default false;
alter table if exists public.vehicle_realtime add column if not exists satellites integer;
alter table if exists public.vehicle_realtime add column if not exists is_phone_detected boolean default false;
alter table if exists public.vehicle_realtime add column if not exists is_drinking_detected boolean default false;
alter table if exists public.vehicle_realtime add column if not exists buzzer_active boolean default false;
alter table if exists public.vehicle_realtime add column if not exists buzzer_type text default 'alert';
alter table if exists public.vehicle_realtime add column if not exists buzzer_updated_at timestamptz;

create or replace function public.update_updated_at_column()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists update_vehicle_realtime_updated_at on public.vehicle_realtime;
create trigger update_vehicle_realtime_updated_at
before update on public.vehicle_realtime
for each row
execute function public.update_updated_at_column();

create table if not exists public.vehicle_trips (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  vehicle_id text not null references public.vehicles(id) on delete cascade,
  session_id uuid not null default gen_random_uuid(),
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  status text not null default 'ok',
  max_speed_mph integer not null default 0,
  avg_speed_mph double precision not null default 0,
  max_intoxication_score integer not null default 0,
  speeding_event_count integer not null default 0,
  speed_sample_count integer not null default 0,
  speed_sample_sum integer not null default 0,
  phone_distraction_event_count integer not null default 0,
  drinking_event_count integer not null default 0,
  route_waypoints jsonb default '[]'::jsonb,
  crash_detected boolean default false,
  crash_severity text
);

alter table if exists public.vehicle_trips add column if not exists created_at timestamptz not null default now();
alter table if exists public.vehicle_trips add column if not exists session_id uuid not null default gen_random_uuid();
alter table if exists public.vehicle_trips add column if not exists started_at timestamptz not null default now();
alter table if exists public.vehicle_trips add column if not exists ended_at timestamptz;
alter table if exists public.vehicle_trips add column if not exists status text not null default 'ok';
alter table if exists public.vehicle_trips add column if not exists max_speed_mph integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists avg_speed_mph double precision not null default 0;
alter table if exists public.vehicle_trips add column if not exists max_intoxication_score integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists speeding_event_count integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists speed_sample_count integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists speed_sample_sum integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists phone_distraction_event_count integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists drinking_event_count integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists route_waypoints jsonb default '[]'::jsonb;
alter table if exists public.vehicle_trips add column if not exists crash_detected boolean default false;
alter table if exists public.vehicle_trips add column if not exists crash_severity text;

alter table public.vehicles enable row level security;
alter table public.vehicle_realtime enable row level security;
alter table public.vehicle_trips enable row level security;

grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on public.vehicles to anon, authenticated;
grant select, insert, update on public.vehicle_realtime to anon, authenticated;
grant select, insert, update on public.vehicle_trips to anon, authenticated;

drop policy if exists "hobby_runtime_vehicles_select_public" on public.vehicles;
create policy "hobby_runtime_vehicles_select_public"
  on public.vehicles
  for select to anon, authenticated
  using (true);

drop policy if exists "hobby_runtime_vehicles_insert_public" on public.vehicles;
create policy "hobby_runtime_vehicles_insert_public"
  on public.vehicles
  for insert to anon, authenticated
  with check (true);

drop policy if exists "hobby_runtime_vehicles_update_public" on public.vehicles;
create policy "hobby_runtime_vehicles_update_public"
  on public.vehicles
  for update to anon, authenticated
  using (true)
  with check (true);

drop policy if exists "hobby_runtime_vehicles_delete_public" on public.vehicles;
create policy "hobby_runtime_vehicles_delete_public"
  on public.vehicles
  for delete to anon, authenticated
  using (true);

drop policy if exists "hobby_runtime_vehicle_realtime_select_public" on public.vehicle_realtime;
create policy "hobby_runtime_vehicle_realtime_select_public"
  on public.vehicle_realtime
  for select to anon, authenticated
  using (true);

drop policy if exists "hobby_runtime_vehicle_realtime_insert_public" on public.vehicle_realtime;
create policy "hobby_runtime_vehicle_realtime_insert_public"
  on public.vehicle_realtime
  for insert to anon, authenticated
  with check (true);

drop policy if exists "hobby_runtime_vehicle_realtime_update_public" on public.vehicle_realtime;
create policy "hobby_runtime_vehicle_realtime_update_public"
  on public.vehicle_realtime
  for update to anon, authenticated
  using (true)
  with check (true);

drop policy if exists "hobby_runtime_vehicle_trips_select_public" on public.vehicle_trips;
create policy "hobby_runtime_vehicle_trips_select_public"
  on public.vehicle_trips
  for select to anon, authenticated
  using (true);

drop policy if exists "hobby_runtime_vehicle_trips_insert_public" on public.vehicle_trips;
create policy "hobby_runtime_vehicle_trips_insert_public"
  on public.vehicle_trips
  for insert to anon, authenticated
  with check (true);

drop policy if exists "hobby_runtime_vehicle_trips_update_public" on public.vehicle_trips;
create policy "hobby_runtime_vehicle_trips_update_public"
  on public.vehicle_trips
  for update to anon, authenticated
  using (true)
  with check (true);

-- Ensure PostgREST refreshes schema cache after compatibility updates.
notify pgrst, 'reload schema';
