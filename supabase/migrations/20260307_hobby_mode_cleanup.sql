-- Hobby-mode destructive cleanup migration
-- Removes face/shazam/live-stream schema and opens public hobby policies.

create extension if not exists pgcrypto;

-- Drop legacy functions by name (all overloads).
do $$
declare
  fn record;
begin
  for fn in
    select p.oid::regprocedure as signature
    from pg_proc p
    join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public'
      and p.proname in (
        'propagate_driver_profile_to_cluster',
        'find_or_create_face_cluster',
        'get_unidentified_face_clusters',
        'join_vehicle_by_invite_code',
        'generate_invite_code',
        'user_has_vehicle_access',
        'get_vehicle_access_users',
        'remove_vehicle_access'
      )
  loop
    execute format('drop function if exists %s cascade', fn.signature);
  end loop;
end;
$$;

drop trigger if exists trigger_propagate_driver_profile on public.face_detections;

drop table if exists public.face_detections cascade;
drop table if exists public.driver_profiles cascade;
drop table if exists public.music_detections cascade;

alter table if exists public.vehicles drop column if exists invite_code;
alter table if exists public.vehicles drop column if exists enable_stream;
alter table if exists public.vehicles drop column if exists enable_shazam;

alter table if exists public.vehicle_realtime drop column if exists current_song_title;
alter table if exists public.vehicle_realtime drop column if exists current_song_artist;
alter table if exists public.vehicle_realtime drop column if exists current_song_detected_at;

-- Drop deprecated storage policies and buckets.
drop policy if exists "storage_service_role_all" on storage.objects;
drop policy if exists "storage_select_with_vehicle_access" on storage.objects;
drop policy if exists "storage_insert_with_auth" on storage.objects;
drop policy if exists "storage_update_with_auth" on storage.objects;
drop policy if exists "storage_delete_with_auth" on storage.objects;
drop policy if exists "live_frames_select_with_vehicle_access" on storage.objects;
drop policy if exists "live_frames_upload_service_role" on storage.objects;
drop policy if exists "live_frames_upload_with_auth" on storage.objects;
drop policy if exists "live_frames_read_service_role" on storage.objects;
drop policy if exists "live_frames_delete_service_role" on storage.objects;

delete from storage.objects where bucket_id in ('face-snapshots', 'live-frames');
delete from storage.buckets where id in ('face-snapshots', 'live-frames');

-- Keep user avatars bucket available.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'user-avatars',
  'user-avatars',
  true,
  10485760,
  array['image/jpeg', 'image/png', 'image/heic', 'image/webp']
)
on conflict (id) do update
set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- Compatibility hardening for runtime tables.
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
  satellites integer,
  is_phone_detected boolean default false,
  is_drinking_detected boolean default false,
  buzzer_active boolean default false,
  buzzer_type text default 'alert',
  buzzer_updated_at timestamptz
);

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

create table if not exists public.user_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  display_name text,
  avatar_path text,
  notification_preferences jsonb not null default
    '{"collision": true, "driver_drowsiness": true, "speed_limit": true, "drunk_driving": true, "fsd": true}'::jsonb,
  notifications_enabled boolean not null default true,
  push_token text
);

create table if not exists public.vehicle_access (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  user_id uuid references auth.users(id) on delete cascade,
  vehicle_id text not null references public.vehicles(id) on delete cascade,
  access_level text not null default 'viewer'
);

alter table if exists public.vehicle_access alter column user_id drop not null;

alter table if exists public.vehicles add column if not exists enable_yolo boolean not null default true;
alter table if exists public.vehicles add column if not exists enable_microphone boolean not null default true;
alter table if exists public.vehicles add column if not exists enable_camera boolean not null default true;
alter table if exists public.vehicles add column if not exists enable_dashcam boolean not null default false;

alter table if exists public.vehicle_realtime add column if not exists is_phone_detected boolean default false;
alter table if exists public.vehicle_realtime add column if not exists is_drinking_detected boolean default false;
alter table if exists public.vehicle_realtime add column if not exists buzzer_active boolean default false;
alter table if exists public.vehicle_realtime add column if not exists buzzer_type text default 'alert';
alter table if exists public.vehicle_realtime add column if not exists buzzer_updated_at timestamptz;

alter table if exists public.vehicle_trips add column if not exists phone_distraction_event_count integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists drinking_event_count integer not null default 0;
alter table if exists public.vehicle_trips add column if not exists route_waypoints jsonb default '[]'::jsonb;
alter table if exists public.vehicle_trips add column if not exists crash_detected boolean default false;
alter table if exists public.vehicle_trips add column if not exists crash_severity text;

-- Public hobby-mode RLS/policies.
alter table public.vehicles enable row level security;
alter table public.vehicle_realtime enable row level security;
alter table public.vehicle_trips enable row level security;
alter table public.user_profiles enable row level security;
alter table public.vehicle_access enable row level security;
alter table storage.objects enable row level security;

grant usage on schema public to anon, authenticated, service_role;
grant select, insert, update, delete on public.vehicles to anon, authenticated;
grant select, insert, update on public.vehicle_realtime to anon, authenticated;
grant select, insert, update on public.vehicle_trips to anon, authenticated;
grant select, insert, update, delete on public.user_profiles to authenticated;
grant select on public.user_profiles to service_role;
grant all on public.vehicle_access to service_role;

drop policy if exists "hobby_vehicles_select_public" on public.vehicles;
create policy "hobby_vehicles_select_public" on public.vehicles for select to anon, authenticated using (true);

drop policy if exists "hobby_vehicles_insert_public" on public.vehicles;
create policy "hobby_vehicles_insert_public" on public.vehicles for insert to anon, authenticated with check (true);

drop policy if exists "hobby_vehicles_update_public" on public.vehicles;
create policy "hobby_vehicles_update_public" on public.vehicles for update to anon, authenticated using (true) with check (true);

drop policy if exists "hobby_vehicles_delete_public" on public.vehicles;
create policy "hobby_vehicles_delete_public" on public.vehicles for delete to anon, authenticated using (true);

drop policy if exists "hobby_vehicle_realtime_select_public" on public.vehicle_realtime;
create policy "hobby_vehicle_realtime_select_public" on public.vehicle_realtime for select to anon, authenticated using (true);

drop policy if exists "hobby_vehicle_realtime_insert_public" on public.vehicle_realtime;
create policy "hobby_vehicle_realtime_insert_public" on public.vehicle_realtime for insert to anon, authenticated with check (true);

drop policy if exists "hobby_vehicle_realtime_update_public" on public.vehicle_realtime;
create policy "hobby_vehicle_realtime_update_public" on public.vehicle_realtime for update to anon, authenticated using (true) with check (true);

drop policy if exists "hobby_vehicle_trips_select_public" on public.vehicle_trips;
create policy "hobby_vehicle_trips_select_public" on public.vehicle_trips for select to anon, authenticated using (true);

drop policy if exists "hobby_vehicle_trips_insert_public" on public.vehicle_trips;
create policy "hobby_vehicle_trips_insert_public" on public.vehicle_trips for insert to anon, authenticated with check (true);

drop policy if exists "hobby_vehicle_trips_update_public" on public.vehicle_trips;
create policy "hobby_vehicle_trips_update_public" on public.vehicle_trips for update to anon, authenticated using (true) with check (true);

drop policy if exists "hobby_user_profiles_select_own" on public.user_profiles;
create policy "hobby_user_profiles_select_own" on public.user_profiles for select to authenticated using (auth.uid() = user_id);

drop policy if exists "hobby_user_profiles_insert_own" on public.user_profiles;
create policy "hobby_user_profiles_insert_own" on public.user_profiles for insert to authenticated with check (auth.uid() = user_id);

drop policy if exists "hobby_user_profiles_update_own" on public.user_profiles;
create policy "hobby_user_profiles_update_own" on public.user_profiles for update to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "hobby_user_profiles_delete_own" on public.user_profiles;
create policy "hobby_user_profiles_delete_own" on public.user_profiles for delete to authenticated using (auth.uid() = user_id);

drop policy if exists "hobby_vehicle_access_service_role_all" on public.vehicle_access;
create policy "hobby_vehicle_access_service_role_all" on public.vehicle_access for all to service_role using (true) with check (true);

drop policy if exists "hobby_user_avatars_select_public" on storage.objects;
create policy "hobby_user_avatars_select_public" on storage.objects for select to anon, authenticated using (bucket_id = 'user-avatars');

drop policy if exists "hobby_user_avatars_insert_own" on storage.objects;
create policy "hobby_user_avatars_insert_own" on storage.objects
for insert to authenticated
with check (bucket_id = 'user-avatars' and (storage.foldername(name))[1] = auth.uid()::text);

drop policy if exists "hobby_user_avatars_update_own" on storage.objects;
create policy "hobby_user_avatars_update_own" on storage.objects
for update to authenticated
using (bucket_id = 'user-avatars' and (storage.foldername(name))[1] = auth.uid()::text)
with check (bucket_id = 'user-avatars' and (storage.foldername(name))[1] = auth.uid()::text);

drop policy if exists "hobby_user_avatars_delete_own" on storage.objects;
create policy "hobby_user_avatars_delete_own" on storage.objects
for delete to authenticated
using (bucket_id = 'user-avatars' and (storage.foldername(name))[1] = auth.uid()::text);

-- Buzzer RPCs without vehicle_access requirements.
drop function if exists public.activate_vehicle_buzzer(text, text);
create or replace function public.activate_vehicle_buzzer(
  p_vehicle_id text,
  p_buzzer_type text default 'alert'
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_type text;
begin
  if not exists (select 1 from public.vehicles where id = p_vehicle_id) then
    return jsonb_build_object('success', false, 'error', 'Vehicle not found');
  end if;

  v_type := coalesce(nullif(trim(p_buzzer_type), ''), 'alert');

  insert into public.vehicle_realtime (vehicle_id, buzzer_active, buzzer_type, buzzer_updated_at, updated_at)
  values (p_vehicle_id, true, v_type, now(), now())
  on conflict (vehicle_id) do update
  set buzzer_active = true,
      buzzer_type = v_type,
      buzzer_updated_at = now(),
      updated_at = now();

  return jsonb_build_object('success', true, 'vehicle_id', p_vehicle_id, 'buzzer_active', true, 'buzzer_type', v_type);
end;
$$;

drop function if exists public.deactivate_vehicle_buzzer(text);
create or replace function public.deactivate_vehicle_buzzer(
  p_vehicle_id text
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
begin
  if not exists (select 1 from public.vehicles where id = p_vehicle_id) then
    return jsonb_build_object('success', false, 'error', 'Vehicle not found');
  end if;

  insert into public.vehicle_realtime (vehicle_id, buzzer_active, buzzer_type, buzzer_updated_at, updated_at)
  values (p_vehicle_id, false, 'alert', now(), now())
  on conflict (vehicle_id) do update
  set buzzer_active = false,
      buzzer_updated_at = now(),
      updated_at = now();

  return jsonb_build_object('success', true, 'vehicle_id', p_vehicle_id, 'buzzer_active', false);
end;
$$;

grant execute on function public.activate_vehicle_buzzer(text, text) to anon, authenticated, service_role;
grant execute on function public.deactivate_vehicle_buzzer(text) to anon, authenticated, service_role;

-- Realtime publication guards.
do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    if not exists (
      select 1
      from pg_publication_tables
      where pubname = 'supabase_realtime'
        and schemaname = 'public'
        and tablename = 'vehicle_realtime'
    ) then
      execute 'alter publication supabase_realtime add table public.vehicle_realtime';
    end if;

    if not exists (
      select 1
      from pg_publication_tables
      where pubname = 'supabase_realtime'
        and schemaname = 'public'
        and tablename = 'vehicle_trips'
    ) then
      execute 'alter publication supabase_realtime add table public.vehicle_trips';
    end if;
  end if;
end;
$$;

