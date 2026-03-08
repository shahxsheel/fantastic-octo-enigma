# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**ADA (Autonomous Driver Awareness)** is currently running in a simplified hobby-mode configuration focused on reliable end-to-end telemetry and alerts.

## Current Feature Scope

### Kept

- GPS telemetry
- Gyro/accelerometer telemetry
- Buzzer control (local and Supabase RPC path)
- YOLO inference (person/phone/drinking object detection)
- Optional local head-direction estimate (`LEFT`, `CENTER`, `RIGHT`)

### Removed

- Shazam/music detection
- Face recognition and face storage
- Live camera streaming from Pi to app
- Invite-code/QR join and Apple Sign-In dependency

## System Architecture

```
iOS App (SwiftUI)
  <-> BLE (CoreBluetooth)
  <-> Supabase Realtime/PostgREST
Raspberry Pi Runtime (Python, YOLO + sensors)
Infineon PSoC Firmware (separate track)
```

| Component | Language | Root Path |
|---|---|---|
| Firmware | C + ModusToolbox | `firmware/` |
| Pi runtime | Python 3 | `research/` |
| iOS app | Swift + SwiftUI | `iOS/` |
| Cloud backend | PostgreSQL + Deno/TypeScript | `supabase/` |
| Utility scripts | Node.js/Bun | `scripts/` |

## Raspberry Pi Runtime (`research/`)

### Setup and Run

```bash
cd research
./setup.sh        # one-time: system libs, .venv (python3.12.8), model downloads
./run.sh
HEADLESS=1 ./run.sh   # SSH / no-display mode

# Tests
python -m pytest tests/ -v
# or
python -m unittest discover tests -v
```

### Architecture

Four threads + main thread in `src/run_single_usb.py`:

1. **Camera thread** — double-buffer USB producer (writes `_back`, swaps `_front` under lock; zero allocation per frame)
2. **Inference thread** — YOLO every 10 frames (reuses last results between frames); head direction every N frames; updates `_shared_results`
3. **Telemetry thread** — drains bounded queue (`maxsize=50`) to Supabase asynchronously
4. **Speed limit thread** — polls OSM Overpass API every 10 s
5. **Main thread** — reads `_shared_results`, CLI stats at 0.5 s, GUI at 30 FPS, fires buzzer

Hardware components (`src/components/`) fall back to mock/simulated mode when GPIO/serial is unavailable (safe for non-Pi dev).

### Runtime Notes

- Main runtime: `research/src/run_single_usb.py`
- Inference thread uses YOLO only for object detection
- Head direction is estimated locally from camera frames (no identity recognition)
- Startup prints colored Supabase status (`ACTIVE`/`INACTIVE`)
- In `HEADLESS=1`, no UI rendering path is executed
- Model files (`face_landmarker.task`, `yolov8n_ncnn_model/`) are downloaded by `setup.sh` and required at runtime

### Python Version Contract

- `research/setup.sh` enforces `.venv` created with `python3.12` version `3.12.8`.
- If an existing `.venv` is on another version, setup recreates it.

### Key Env Variables

| Variable | Default | Purpose |
|---|---|---|
| `HEADLESS` | auto (`1` when no display) | Disable GUI rendering |
| `USE_FAKE_GPS` | `0` | Use synthetic GPS values |
| `USE_FAKE_GYRO` | `0` | Use synthetic gyro values |
| `ENABLE_HEAD_DIRECTION` | `1` | Enable local left/center/right estimate |
| `HEAD_DIRECTION_EVERY_N` | `4` | Compute cadence for head-direction |
| `SUPABASE_URL` / `SUPABASE_KEY` | placeholders | Optional cloud telemetry |
| `VEHICLE_ID` | `VEHICLE-001` in `.env` | Vehicle identifier |

## iOS App (`iOS/`)

### Current App Behavior

- App launch is public-mode friendly (auth optional for local/dev usage).
- Join Vehicle screen uses NHTSA make/model internet catalog.
- Join action renames an existing connected vehicle in Supabase.

### Setup

Copy `iOS/InfineonProject/Utilities/Constants.template.swift` → `Constants.swift` (same folder, gitignored) and fill in your Supabase URL and anon key. The app will not compile without `Constants.swift`.

### Key Files

- `iOS/InfineonProject/InfineonProjectApp.swift` — app entry point, SwiftData container, RootView routing
- `iOS/InfineonProject/Services/SupabaseService.swift` — `@Observable @MainActor` singleton (`supabase`); all Supabase reads/writes, Realtime subscription, SwiftData cache
- `iOS/InfineonProject/Services/BluetoothManager.swift` — CoreBluetooth BLE connection to Pi
- `iOS/InfineonProject/Models/CachedModels.swift` — SwiftData models (`CachedVehicle`, `CachedVehicleRealtime`)
- `iOS/InfineonProject/Views/V2LaunchUI/V2MainView.swift` — main tab container
- `iOS/InfineonProject/Views/Vehicle/JoinVehicleView.swift` — NHTSA catalog + vehicle rename

### Build Commands

```bash
# Compile check (no signing required)
xcodebuild -project iOS/InfineonProject.xcodeproj \
  -scheme InfineonProject \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build

# Run unit tests
xcodebuild test \
  -project iOS/InfineonProject.xcodeproj \
  -scheme InfineonProject \
  -destination 'platform=iOS Simulator,name=iPhone 16'
```

### Swift / SwiftUI Conventions

These rules apply to all iOS code (from `iOS/CLAUDE.md`):

- Target iOS 26.0+, Swift 6.2+, strict concurrency. All `@Observable` classes must be `@MainActor`.
- No `ObservableObject` / `@Published` / `DispatchQueue` — use `@Observable` + `async`/`await`.
- No `NavigationView` → use `NavigationStack` + `navigationDestination(for:)`.
- Use `foregroundStyle()` not `foregroundColor()`, `clipShape(.rect(cornerRadius:))` not `.cornerRadius()`, `Tab` API not `tabItem()`.
- No UIKit colors or `UIScreen.main.bounds` in SwiftUI.
- Break views into separate `View` structs (not computed properties).
- No third-party frameworks without asking first.

## Supabase Backend (`supabase/`)

### Source-of-Truth SQL

- Cleanup migration: `supabase/migrations/20260307_hobby_mode_cleanup.sql`
- Idempotent setup: `supabase/setup.sql`

### Current Data Model

Runtime tables in use:

- `vehicles`
- `vehicle_realtime`
- `vehicle_trips`

Compatibility table:

- `vehicle_access` (deprecated; retained for backward compatibility only)

### Removed Objects

- Tables: `face_detections`, `driver_profiles`, `music_detections`
- Columns: `vehicles.enable_stream`, `vehicles.enable_shazam`, `vehicle_realtime.current_song_*`
- Buckets: `face-snapshots`, `live-frames`

### Functions

- `activate_vehicle_buzzer(text, text)` and `deactivate_vehicle_buzzer(text)` are available without `vehicle_access` checks.
- `supabase/functions/send-notifications/index.ts` keeps speeding notification path.

## Firmware (`firmware/`)

Firmware remains separate from current Pi hobby-mode runtime changes. Use ModusToolbox for build/programming workflows.

## Agent Rules

1. Keep edits scoped to the request.
2. Do not reintroduce removed face/music/stream features unless explicitly requested.
3. Do not commit secrets (`.env`, API keys, signing keys).
4. Prefer idempotent SQL and explicit migration changes.
5. Run relevant checks for touched areas before finishing.
