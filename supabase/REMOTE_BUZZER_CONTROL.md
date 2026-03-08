# Remote Buzzer Control

This feature allows you to remotely control the vehicle's buzzer from the iOS app.

## Database Schema

The `vehicle_realtime` table includes these buzzer control fields:
- `buzzer_active` (BOOLEAN): Whether the buzzer should be playing
- `buzzer_type` (TEXT): Type of buzzer pattern ('alert', 'emergency', 'warning')
- `buzzer_updated_at` (TIMESTAMPTZ): Last time buzzer state was changed

## Usage from iOS App

### Method 1: Using Helper Functions (Recommended)

```swift
// Activate buzzer
let response = try await supabase.rpc(
    fn: "activate_vehicle_buzzer",
    params: [
        "p_vehicle_id": vehicleId,
        "p_buzzer_type": "emergency"  // or "alert" or "warning"
    ]
).execute()

// Deactivate buzzer
let response = try await supabase.rpc(
    fn: "deactivate_vehicle_buzzer",
    params: ["p_vehicle_id": vehicleId]
).execute()
```

### Method 2: Direct Update

```swift
// Activate buzzer
try await supabase
    .from("vehicle_realtime")
    .update([
        "buzzer_active": true,
        "buzzer_type": "emergency",
        "buzzer_updated_at": Date.now
    ])
    .eq("vehicle_id", vehicleId)
    .execute()

// Deactivate buzzer
try await supabase
    .from("vehicle_realtime")
    .update([
        "buzzer_active": false,
        "buzzer_updated_at": Date.now
    ])
    .eq("vehicle_id", vehicleId)
    .execute()
```

## Buzzer Patterns

The Raspberry Pi will play different patterns based on `buzzer_type`:

- **`alert`**: 800Hz, 0.5s on / 0.5s off (moderate urgency)
- **`emergency`**: 1200Hz, 0.3s on / 0.3s off (high urgency, faster)
- **`warning`**: 600Hz, 0.7s on / 0.7s off (low urgency, slower)

## How It Works

1. iOS app updates `buzzer_active` in the `vehicle_realtime` table
2. Supabase Realtime broadcasts the change to all subscribers
3. Raspberry Pi (running `main.py`) receives the realtime update
4. `BuzzerController` starts/stops continuous buzzer playback in a background thread
5. Buzzer plays continuously until `buzzer_active` is set to `false`

## Security

- Row Level Security (RLS) policies ensure users can only control buzzer for vehicles they have access to
- Helper functions (`activate_vehicle_buzzer`, `deactivate_vehicle_buzzer`) include built-in access checks
- Only authenticated users with vehicle access can modify buzzer state

## Testing Locally

You can test the buzzer control using the Supabase SQL Editor:

```sql
-- Activate buzzer for a vehicle (as service role)
UPDATE vehicle_realtime
SET buzzer_active = true,
    buzzer_type = 'emergency',
    buzzer_updated_at = NOW()
WHERE vehicle_id = 'YOUR_VEHICLE_ID';

-- Deactivate buzzer
UPDATE vehicle_realtime
SET buzzer_active = false,
    buzzer_updated_at = NOW()
WHERE vehicle_id = 'YOUR_VEHICLE_ID';
```

## Troubleshooting

If the buzzer doesn't respond:
1. Check that `main.py` is running on the Raspberry Pi
2. Verify realtime subscription is active (check console logs for "Subscribed to remote buzzer commands")
3. Ensure the vehicle_id matches between `.env` file and database
4. Check GPIO pin 18 is correctly connected to the buzzer
5. If using simulated mode, buzzer will only print messages instead of playing sound
