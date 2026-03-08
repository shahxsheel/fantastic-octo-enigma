# Winter 2026 Infineon Special Project

## About

We're making driving safe for everyone.

## Project Structure

We use a monorepo structure, so all project components can be in one repo.

## [Firmware](firmware/README.md)

**Folder:** [`firmware`](firmware/README.md)

- *Built with:* C, Infineon ModusToolbox

The actual low-level C code that runs on the Infineon AI board.

## [iOS Project](iOS/)

**Folder:** [`iOS`](iOS/)

- *Built with:* Swift, SwiftUI, Supabase

The companion iOS app allows users to keep track of driver profiles, previous infractions, and other features.

## [Research](research/)

**Folder:** [`iOS`](research/README.md)

- *Built with:* Python

All the AI/ML code running on the Raspberry Pi.

## [Supabase Database](supabase/)

**Folder:** [`supabase`](supabase/)

- *Built with:* Supabase, PostgreSQL

This folder contains the Supabase PostgreSQL table and bucket setup, along with Supabase Edge functions and future migrations.

## [Helper Scripts](scripts/README.md)

**Folder:** [`scripts`](scripts/README.md)

- *Built with:* JavaScript, Bun

This folder contains helpful functions for the project.

## © Copyright 2026 Aaron Ma. All rights reserved.
