# iOS App

## [Join TestFlight](https://testflight.apple.com/join/TaDdSd1f)

## About

Download the companion app for the on the go status updated on your vehicle!

## Technologies Used

- Swift
- SwiftUI
- AaronUI
- Supabase

## Supabase Runtime Config

The app now reads Supabase config with this precedence:

1. Process environment: `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`
2. Info.plist keys (injected from build settings): `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`
3. Legacy fallback in `iOS/Constants.swift` (placeholder by default)

Set build settings on the app target:

- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`

If config is missing/placeholder, cloud operations are disabled at runtime and BLE/local features continue.
