//
//  CachedModels.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/16/26.
//

import Foundation
import SwiftData

/// Local cache of a vehicle for instant loading and offline access.
@Model
final class CachedVehicle {
  @Attribute(.unique) var id: String = ""
  var createdAt: Date = Date()
  var updatedAt: Date = Date()
  var name: String?
  var vehicleDescription: String?
  var ownerId: UUID?

  // Feature toggles
  var enableYolo: Bool = true
  var enableMicrophone: Bool = true
  var enableCamera: Bool = true
  var enableDashcam: Bool = false

  init() {}

  init(from vehicle: Vehicle) {
    self.id = vehicle.id
    self.createdAt = vehicle.createdAt
    self.updatedAt = vehicle.updatedAt
    self.name = vehicle.name
    self.vehicleDescription = vehicle.description
    self.ownerId = vehicle.ownerId
    self.enableYolo = vehicle.enableYolo
    self.enableMicrophone = vehicle.enableMicrophone
    self.enableCamera = vehicle.enableCamera
    self.enableDashcam = vehicle.enableDashcam
  }

  func toVehicle() -> Vehicle {
    Vehicle(
      id: id,
      createdAt: createdAt,
      updatedAt: updatedAt,
      name: name,
      description: vehicleDescription,
      ownerId: ownerId,
      enableYolo: enableYolo,
      enableMicrophone: enableMicrophone,
      enableCamera: enableCamera,
      enableDashcam: enableDashcam
    )
  }

  func update(from vehicle: Vehicle) {
    self.createdAt = vehicle.createdAt
    self.updatedAt = vehicle.updatedAt
    self.name = vehicle.name
    self.vehicleDescription = vehicle.description
    self.ownerId = vehicle.ownerId
    self.enableYolo = vehicle.enableYolo
    self.enableMicrophone = vehicle.enableMicrophone
    self.enableCamera = vehicle.enableCamera
    self.enableDashcam = vehicle.enableDashcam
  }
}

/// Local cache of a vehicle's latest realtime telemetry data.
@Model
final class CachedVehicleRealtime {
  @Attribute(.unique) var vehicleId: String = ""
  var updatedAt: Date = Date()
  var latitude: Double?
  var longitude: Double?
  var speedMph: Int = 0
  var speedLimitMph: Int = 0
  var headingDegrees: Int = 0
  var compassDirection: String = ""
  var isSpeeding: Bool = false
  var isMoving: Bool = false
  var driverStatus: String = ""
  var intoxicationScore: Int = 0
  var satellites: Int?
  var isPhoneDetected: Bool?
  var isDrinkingDetected: Bool?
  var buzzerActive: Bool?
  var buzzerType: String?
  var buzzerUpdatedAt: Date?

  init() {}

  init(from data: VehicleRealtime) {
    self.vehicleId = data.vehicleId
    self.updatedAt = data.updatedAt
    self.latitude = data.latitude
    self.longitude = data.longitude
    self.speedMph = data.speedMph
    self.speedLimitMph = data.speedLimitMph
    self.headingDegrees = data.headingDegrees
    self.compassDirection = data.compassDirection
    self.isSpeeding = data.isSpeeding
    self.isMoving = data.isMoving
    self.driverStatus = data.driverStatus
    self.intoxicationScore = data.intoxicationScore
    self.satellites = data.satellites
    self.isPhoneDetected = data.isPhoneDetected
    self.isDrinkingDetected = data.isDrinkingDetected
    self.buzzerActive = data.buzzerActive
    self.buzzerType = data.buzzerType
    self.buzzerUpdatedAt = data.buzzerUpdatedAt
  }

  func toVehicleRealtime() -> VehicleRealtime {
    VehicleRealtime(
      vehicleId: vehicleId,
      updatedAt: updatedAt,
      latitude: latitude,
      longitude: longitude,
      speedMph: speedMph,
      speedLimitMph: speedLimitMph,
      headingDegrees: headingDegrees,
      compassDirection: compassDirection,
      isSpeeding: isSpeeding,
      isMoving: isMoving,
      driverStatus: driverStatus,
      intoxicationScore: intoxicationScore,
      satellites: satellites,
      isPhoneDetected: isPhoneDetected,
      isDrinkingDetected: isDrinkingDetected,
      buzzerActive: buzzerActive,
      buzzerType: buzzerType,
      buzzerUpdatedAt: buzzerUpdatedAt
    )
  }

  func update(from data: VehicleRealtime) {
    self.updatedAt = data.updatedAt
    self.latitude = data.latitude
    self.longitude = data.longitude
    self.speedMph = data.speedMph
    self.speedLimitMph = data.speedLimitMph
    self.headingDegrees = data.headingDegrees
    self.compassDirection = data.compassDirection
    self.isSpeeding = data.isSpeeding
    self.isMoving = data.isMoving
    self.driverStatus = data.driverStatus
    self.intoxicationScore = data.intoxicationScore
    self.satellites = data.satellites
    self.isPhoneDetected = data.isPhoneDetected
    self.isDrinkingDetected = data.isDrinkingDetected
    self.buzzerActive = data.buzzerActive
    self.buzzerType = data.buzzerType
    self.buzzerUpdatedAt = data.buzzerUpdatedAt
  }
}

/// Locally finalized trip captured from BLE stream while Pi was connected.
@Model
final class CachedLocalTrip {
  @Attribute(.unique) var id: UUID = UUID()
  var createdAt: Date = Date()
  var vehicleId: String = ""
  var sessionId: UUID = UUID()
  var startedAt: Date = Date()
  var endedAt: Date = Date()
  var status: String = "ok"
  var maxSpeedMph: Int = 0
  var avgSpeedMph: Double = 0
  var maxIntoxicationScore: Int = 0
  var speedingEventCount: Int = 0
  var speedSampleCount: Int = 0
  var speedSampleSum: Int = 0
  var phoneDistractionEventCount: Int = 0
  var drinkingEventCount: Int = 0
  var routeWaypointsData: Data = Data()

  init() {}

  init(from payload: LocalTripPersistPayload) {
    id = payload.id
    createdAt = payload.createdAt
    vehicleId = payload.vehicleId
    sessionId = payload.sessionId
    startedAt = payload.startedAt
    endedAt = payload.endedAt
    status = payload.status
    maxSpeedMph = payload.maxSpeedMph
    avgSpeedMph = payload.avgSpeedMph
    maxIntoxicationScore = payload.maxIntoxicationScore
    speedingEventCount = payload.speedingEventCount
    speedSampleCount = payload.speedSampleCount
    speedSampleSum = payload.speedSampleSum
    phoneDistractionEventCount = payload.phoneDistractionEventCount
    drinkingEventCount = payload.drinkingEventCount
    routeWaypointsData = (try? JSONEncoder().encode(payload.routeWaypoints)) ?? Data()
  }

  func toVehicleTrip() -> VehicleTrip {
    let routeWaypoints = (try? JSONDecoder().decode([RouteWaypoint].self, from: routeWaypointsData)) ?? []
    return VehicleTrip(
      id: id,
      createdAt: createdAt,
      vehicleId: vehicleId,
      sessionId: sessionId,
      startedAt: startedAt,
      endedAt: endedAt,
      status: status,
      maxSpeedMph: maxSpeedMph,
      avgSpeedMph: avgSpeedMph,
      maxIntoxicationScore: maxIntoxicationScore,
      speedingEventCount: speedingEventCount,
      speedSampleCount: speedSampleCount,
      speedSampleSum: speedSampleSum,
      phoneDistractionEventCount: phoneDistractionEventCount,
      drinkingEventCount: drinkingEventCount,
      routeWaypoints: routeWaypoints,
      crashDetected: nil,
      crashSeverity: nil
    )
  }
}
