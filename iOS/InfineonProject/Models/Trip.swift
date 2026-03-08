//
//  Trip.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/12/26.
//

import MapKit
import SwiftUI

struct TripTelemetryPoint: Identifiable {
  let id: Int
  let timestamp: Date
  let speedMph: Int
  let riskScore: Int
}

/// A wrapper struct for VehicleTrip that provides convenient computed properties for UI display
struct Trip: Identifiable {
  let vehicleTrip: VehicleTrip

  var id: UUID { vehicleTrip.id }
  var timeStarted: Date { vehicleTrip.startedAt }
  var timeEnded: Date? { vehicleTrip.endedAt }
  var status: VehicleTrip.TripStatus { vehicleTrip.tripStatus }

  // Trip statistics
  var maxSpeedMph: Int { vehicleTrip.maxSpeedMph }
  var avgSpeedMph: Double { vehicleTrip.avgSpeedMph }
  var maxIntoxicationScore: Int { vehicleTrip.maxIntoxicationScore }
  var speedingEventCount: Int { vehicleTrip.speedingEventCount }
  var sessionId: UUID { vehicleTrip.sessionId }
  // GPS route
  var routeCoordinates: [CLLocationCoordinate2D] {
    vehicleTrip.routeWaypoints?.map { $0.coordinate } ?? []
  }
  var telemetryPoints: [TripTelemetryPoint] {
    let waypoints = vehicleTrip.routeWaypoints ?? []
    let sorted = waypoints.sorted { $0.ts < $1.ts }
    return sorted.enumerated().map { idx, waypoint in
      TripTelemetryPoint(
        id: idx,
        timestamp: waypoint.date,
        speedMph: waypoint.spd,
        riskScore: waypoint.ix ?? 0
      )
    }
  }
  // Distraction events
  var phoneDistractionEventCount: Int { vehicleTrip.phoneDistractionEventCount ?? 0 }
  var drinkingEventCount: Int { vehicleTrip.drinkingEventCount ?? 0 }
  // Crash detection
  var crashDetected: Bool { vehicleTrip.crashDetected ?? false }
  var crashSeverity: String? { vehicleTrip.crashSeverity }

  init(vehicleTrip: VehicleTrip) {
    self.vehicleTrip = vehicleTrip
  }

  var tripStatus: String {
    status.displayName
  }

  var tripColor: Color {
    status.color
  }

  var tripIcon: String {
    status.icon
  }

  /// Duration of the trip in seconds, or time since start if still ongoing
  var duration: TimeInterval {
    let endTime = timeEnded ?? Date()
    return endTime.timeIntervalSince(timeStarted)
  }

  /// Formatted duration string
  var formattedDuration: String {
    let formatter = DateComponentsFormatter()
    formatter.allowedUnits = [.hour, .minute, .second]
    formatter.unitsStyle = .abbreviated
    return formatter.string(from: duration) ?? "0s"
  }

  /// Whether the trip is still ongoing (no end time)
  var isOngoing: Bool {
    timeEnded == nil
  }

  /// Smart driving score computed from all trip event data.
  var score: TripScore {
    DrivingScoreCalculator.score(for: self)
  }
}

extension Trip {
  /// Sample trip for previews
  static let sample = Trip(
    vehicleTrip: VehicleTrip(
      id: UUID(),
      createdAt: .now,
      vehicleId: "sample",
      sessionId: UUID(),
      startedAt: .now.addingTimeInterval(-3600),
      endedAt: .now,
      status: "ok",
      maxSpeedMph: 65,
      avgSpeedMph: 45.5,
      maxIntoxicationScore: 0,
      speedingEventCount: 0,
      speedSampleCount: 100,
      speedSampleSum: 4550,
      phoneDistractionEventCount: 0,
      drinkingEventCount: 0,
      routeWaypoints: nil,
      crashDetected: nil,
      crashSeverity: nil
    )
  )

  static let sampleWarning = Trip(
    vehicleTrip: VehicleTrip(
      id: UUID(),
      createdAt: .now,
      vehicleId: "sample",
      sessionId: UUID(),
      startedAt: .now.addingTimeInterval(-1800),
      endedAt: .now,
      status: "warning",
      maxSpeedMph: 80,
      avgSpeedMph: 55.0,
      maxIntoxicationScore: 2,
      speedingEventCount: 3,
      speedSampleCount: 50,
      speedSampleSum: 2750,
      phoneDistractionEventCount: 2,
      drinkingEventCount: 1,
      routeWaypoints: nil,
      crashDetected: nil,
      crashSeverity: nil
    )
  )

  static let sampleDanger = Trip(
    vehicleTrip: VehicleTrip(
      id: UUID(),
      createdAt: .now,
      vehicleId: "sample",
      sessionId: UUID(),
      startedAt: .now.addingTimeInterval(-900),
      endedAt: nil,
      status: "danger",
      maxSpeedMph: 95,
      avgSpeedMph: 70.0,
      maxIntoxicationScore: 5,
      speedingEventCount: 8,
      speedSampleCount: 30,
      speedSampleSum: 2100,
      phoneDistractionEventCount: 5,
      drinkingEventCount: 3,
      routeWaypoints: nil,
      crashDetected: nil,
      crashSeverity: nil
    )
  )
}
