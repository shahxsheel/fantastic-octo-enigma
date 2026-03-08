# AGENTS.md

Guidance for coding agents working in this repository.

## Project State

ADA is currently in **hobby mode**:

- Keep: `GPS`, `Gyro`, `Buzzer`, `YOLO` inference, Supabase telemetry, BLE transport.
- Removed: Shazam/music detection, face recognition/storage, live camera streaming.
- iOS auth is optional; app runs in public mode without Apple Sign-In.

## Repository Map

| Component | Language | Path |
|---|---|---|
| iOS app | Swift/SwiftUI | `iOS/` |
| Pi runtime | Python | `research/` |
| Supabase backend | SQL + Deno/TS | `supabase/` |
| Firmware | C | `firmware/` |
| Utility scripts | Node/Bun | `scripts/` |

## Active Architecture

```
iOS App (SwiftUI)
  <-> BLE (CoreBluetooth)
  <-> Supabase Realtime/PostgREST
Raspberry Pi (Python, YOLO + sensors)
Infineon PSoC firmware (separate track)
```

## Behavior Contracts

1. No face/shazam/stream feature paths should be reintroduced.
2. Join flow is internet make/model selection (NHTSA) that renames an existing vehicle.
3. Supabase writes are public hobby-mode compatible for required runtime tables.
4. Buzzer RPCs do not depend on `vehicle_access`.

## Raspberry Pi (`research/`)

### Setup and Run

```bash
cd research
./setup.sh
./run.sh
HEADLESS=1 ./run.sh
```

### Runtime Notes

- Entry point: `research/src/run_single_usb.py`
- Inference: YOLO-only object detection plus local head-direction estimate (`LEFT/CENTER/RIGHT`)
- Startup logs include colored Supabase status (`ACTIVE` or `INACTIVE`)
- Headless mode must not render UI (`imshow`/window creation disabled)

### Required Python Version

- `research/setup.sh` enforces `.venv` with **Python 3.12.8**.

### Key Env Vars

- `SUPABASE_URL`, `SUPABASE_KEY`
- `VEHICLE_ID`, `VEHICLE_NAME`
- `HEADLESS`
- `USE_FAKE_GPS`, `USE_FAKE_GYRO`
- `ENABLE_HEAD_DIRECTION`, `HEAD_DIRECTION_EVERY_N`

## iOS (`iOS/`)

- Main app entry: `iOS/InfineonProject/InfineonProjectApp.swift`
- Join vehicle flow: `iOS/InfineonProject/Views/Vehicle/JoinVehicleView.swift`
- Supabase client: `iOS/InfineonProject/Services/SupabaseService.swift`
- BLE manager: `iOS/InfineonProject/Services/BluetoothManager.swift`

## Supabase (`supabase/`)

- Migration: `supabase/migrations/20260307_hobby_mode_cleanup.sql`
- Idempotent setup: `supabase/setup.sql`
- Notifications function: `supabase/functions/send-notifications/index.ts`

Primary runtime tables:

- `vehicles`
- `vehicle_realtime`
- `vehicle_trips`
- `vehicle_access` (deprecated; retained for compatibility)

## Agent Guardrails

1. Keep changes scoped and reversible.
2. Do not commit secrets (`.env`, API keys, private keys).
3. Prefer idempotent SQL and backward-compatible migrations.
4. Validate touched areas (`python -m py_compile`, `bash -n`, `xcodebuild` compile-only when relevant).
5. Update docs when contracts or runtime behavior change.
