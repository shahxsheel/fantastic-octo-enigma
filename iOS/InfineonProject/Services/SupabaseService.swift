//
//  SupabaseService.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/13/26.
//

import CoreLocation
import CryptoKit
import Supabase
import SwiftData
import SwiftUI

// MARK: - Models

struct Vehicle: Codable, Identifiable {
  let id: String
  let createdAt: Date
  let updatedAt: Date
  let name: String?
  let description: String?
  let ownerId: UUID?

  // Feature toggles
  var enableYolo: Bool
  var enableMicrophone: Bool
  var enableCamera: Bool
  var enableDashcam: Bool

  enum CodingKeys: String, CodingKey {
    case id
    case createdAt = "created_at"
    case updatedAt = "updated_at"
    case name, description
    case ownerId = "owner_id"
    case enableYolo = "enable_yolo"
    case enableMicrophone = "enable_microphone"
    case enableCamera = "enable_camera"
    case enableDashcam = "enable_dashcam"
  }

  init(
    id: String,
    createdAt: Date,
    updatedAt: Date,
    name: String?,
    description: String?,
    ownerId: UUID?,
    enableYolo: Bool = true,
    enableMicrophone: Bool = true,
    enableCamera: Bool = true,
    enableDashcam: Bool = false
  ) {
    self.id = id
    self.createdAt = createdAt
    self.updatedAt = updatedAt
    self.name = name
    self.description = description
    self.ownerId = ownerId
    self.enableYolo = enableYolo
    self.enableMicrophone = enableMicrophone
    self.enableCamera = enableCamera
    self.enableDashcam = enableDashcam
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    id = try container.decode(String.self, forKey: .id)
    createdAt = try container.decode(Date.self, forKey: .createdAt)
    updatedAt = try container.decode(Date.self, forKey: .updatedAt)
    name = try container.decodeIfPresent(String.self, forKey: .name)
    description = try container.decodeIfPresent(String.self, forKey: .description)
    ownerId = try container.decodeIfPresent(UUID.self, forKey: .ownerId)
    enableYolo = try container.decodeIfPresent(Bool.self, forKey: .enableYolo) ?? true
    enableMicrophone = try container.decodeIfPresent(Bool.self, forKey: .enableMicrophone) ?? true
    enableCamera = try container.decodeIfPresent(Bool.self, forKey: .enableCamera) ?? true
    enableDashcam = try container.decodeIfPresent(Bool.self, forKey: .enableDashcam) ?? false
  }
}

struct VehicleRealtime: Codable, Identifiable {
  var id: String { vehicleId }
  let vehicleId: String
  let updatedAt: Date
  let latitude: Double?
  let longitude: Double?
  let speedMph: Int
  let speedLimitMph: Int
  let headingDegrees: Int
  let compassDirection: String
  let isSpeeding: Bool
  let isMoving: Bool
  let driverStatus: String
  let intoxicationScore: Int
  // GPS data
  let satellites: Int?
  // Distraction detection
  let isPhoneDetected: Bool?
  let isDrinkingDetected: Bool?
  // Remote buzzer control
  let buzzerActive: Bool?
  let buzzerType: String?
  let buzzerUpdatedAt: Date?

  enum CodingKeys: String, CodingKey {
    case vehicleId = "vehicle_id"
    case updatedAt = "updated_at"
    case latitude, longitude
    case speedMph = "speed_mph"
    case speedLimitMph = "speed_limit_mph"
    case headingDegrees = "heading_degrees"
    case compassDirection = "compass_direction"
    case isSpeeding = "is_speeding"
    case isMoving = "is_moving"
    case driverStatus = "driver_status"
    case intoxicationScore = "intoxication_score"
    case satellites
    case isPhoneDetected = "is_phone_detected"
    case isDrinkingDetected = "is_drinking_detected"
    case buzzerActive = "buzzer_active"
    case buzzerType = "buzzer_type"
    case buzzerUpdatedAt = "buzzer_updated_at"
  }

  init(
    vehicleId: String,
    updatedAt: Date,
    latitude: Double?,
    longitude: Double?,
    speedMph: Int,
    speedLimitMph: Int,
    headingDegrees: Int,
    compassDirection: String,
    isSpeeding: Bool,
    isMoving: Bool,
    driverStatus: String,
    intoxicationScore: Int,
    satellites: Int?,
    isPhoneDetected: Bool?,
    isDrinkingDetected: Bool?,
    buzzerActive: Bool? = nil,
    buzzerType: String? = nil,
    buzzerUpdatedAt: Date? = nil
  ) {
    self.vehicleId = vehicleId
    self.updatedAt = updatedAt
    self.latitude = latitude
    self.longitude = longitude
    self.speedMph = speedMph
    self.speedLimitMph = speedLimitMph
    self.headingDegrees = headingDegrees
    self.compassDirection = compassDirection
    self.isSpeeding = isSpeeding
    self.isMoving = isMoving
    self.driverStatus = driverStatus
    self.intoxicationScore = intoxicationScore
    self.satellites = satellites
    self.isPhoneDetected = isPhoneDetected
    self.isDrinkingDetected = isDrinkingDetected
    self.buzzerActive = buzzerActive
    self.buzzerType = buzzerType
    self.buzzerUpdatedAt = buzzerUpdatedAt
  }
}

struct RouteWaypoint: Codable {
  let lat: Double
  let lng: Double
  let spd: Int
  let ts: Int
  let ix: Int?

  var coordinate: CLLocationCoordinate2D {
    CLLocationCoordinate2D(latitude: lat, longitude: lng)
  }

  var date: Date {
    Date(timeIntervalSince1970: TimeInterval(ts))
  }
}

struct VehicleTrip: Codable, Identifiable {
  let id: UUID
  let createdAt: Date
  let vehicleId: String
  let sessionId: UUID
  let startedAt: Date
  let endedAt: Date?
  let status: String
  let maxSpeedMph: Int
  let avgSpeedMph: Double
  let maxIntoxicationScore: Int
  let speedingEventCount: Int
  let speedSampleCount: Int
  let speedSampleSum: Int
  // Distraction event counts
  let phoneDistractionEventCount: Int?
  let drinkingEventCount: Int?

  // GPS route waypoints
  let routeWaypoints: [RouteWaypoint]?

  // Crash detection
  let crashDetected: Bool?
  let crashSeverity: String?

  enum CodingKeys: String, CodingKey {
    case id
    case createdAt = "created_at"
    case vehicleId = "vehicle_id"
    case sessionId = "session_id"
    case startedAt = "started_at"
    case endedAt = "ended_at"
    case status
    case maxSpeedMph = "max_speed_mph"
    case avgSpeedMph = "avg_speed_mph"
    case maxIntoxicationScore = "max_intoxication_score"
    case speedingEventCount = "speeding_event_count"
    case speedSampleCount = "speed_sample_count"
    case speedSampleSum = "speed_sample_sum"
    case phoneDistractionEventCount = "phone_distraction_event_count"
    case drinkingEventCount = "drinking_event_count"
    case routeWaypoints = "route_waypoints"
    case crashDetected = "crash_detected"
    case crashSeverity = "crash_severity"
  }

  init(
    id: UUID,
    createdAt: Date,
    vehicleId: String,
    sessionId: UUID,
    startedAt: Date,
    endedAt: Date?,
    status: String,
    maxSpeedMph: Int,
    avgSpeedMph: Double,
    maxIntoxicationScore: Int,
    speedingEventCount: Int,
    speedSampleCount: Int,
    speedSampleSum: Int,
    phoneDistractionEventCount: Int?,
    drinkingEventCount: Int?,
    routeWaypoints: [RouteWaypoint]?,
    crashDetected: Bool?,
    crashSeverity: String?
  ) {
    self.id = id
    self.createdAt = createdAt
    self.vehicleId = vehicleId
    self.sessionId = sessionId
    self.startedAt = startedAt
    self.endedAt = endedAt
    self.status = status
    self.maxSpeedMph = maxSpeedMph
    self.avgSpeedMph = avgSpeedMph
    self.maxIntoxicationScore = maxIntoxicationScore
    self.speedingEventCount = speedingEventCount
    self.speedSampleCount = speedSampleCount
    self.speedSampleSum = speedSampleSum
    self.phoneDistractionEventCount = phoneDistractionEventCount
    self.drinkingEventCount = drinkingEventCount
    self.routeWaypoints = routeWaypoints
    self.crashDetected = crashDetected
    self.crashSeverity = crashSeverity
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(UUID.self, forKey: .id)
    createdAt = try c.decode(Date.self, forKey: .createdAt)
    vehicleId = try c.decode(String.self, forKey: .vehicleId)
    sessionId = try c.decode(UUID.self, forKey: .sessionId)
    startedAt = try c.decode(Date.self, forKey: .startedAt)
    endedAt = try c.decodeIfPresent(Date.self, forKey: .endedAt)
    status = try c.decode(String.self, forKey: .status)
    maxSpeedMph = try c.decode(Int.self, forKey: .maxSpeedMph)
    avgSpeedMph = try c.decode(Double.self, forKey: .avgSpeedMph)
    maxIntoxicationScore = try c.decode(Int.self, forKey: .maxIntoxicationScore)
    speedingEventCount = try c.decode(Int.self, forKey: .speedingEventCount)
    speedSampleCount = try c.decode(Int.self, forKey: .speedSampleCount)
    speedSampleSum = try c.decode(Int.self, forKey: .speedSampleSum)
    phoneDistractionEventCount = try c.decodeIfPresent(Int.self, forKey: .phoneDistractionEventCount)
    drinkingEventCount = try c.decodeIfPresent(Int.self, forKey: .drinkingEventCount)

    // Handle route_waypoints: may be a JSON array or a double-encoded JSON string
    if let waypoints = try? c.decodeIfPresent([RouteWaypoint].self, forKey: .routeWaypoints) {
      routeWaypoints = waypoints
    } else if let waypointsString = try? c.decodeIfPresent(String.self, forKey: .routeWaypoints),
      let data = waypointsString.data(using: .utf8),
      let decoded = try? JSONDecoder().decode([RouteWaypoint].self, from: data)
    {
      routeWaypoints = decoded
    } else {
      routeWaypoints = nil
    }

    crashDetected = try c.decodeIfPresent(Bool.self, forKey: .crashDetected)
    crashSeverity = try c.decodeIfPresent(String.self, forKey: .crashSeverity)
  }

  /// Returns the trip status as an enum for easier handling
  var tripStatus: TripStatus {
    TripStatus(rawValue: status) ?? .ok
  }

  enum TripStatus: String, Codable, CaseIterable {
    case ok
    case warning
    case danger

    var displayName: String {
      switch self {
      case .ok: "OK"
      case .warning: "Warning"
      case .danger: "Danger"
      }
    }

    var color: Color {
      switch self {
      case .ok: .green
      case .warning: .yellow
      case .danger: .red
      }
    }

    var icon: String {
      switch self {
      case .ok: "checkmark"
      case .warning: "exclamationmark.triangle.fill"
      case .danger: "xmark"
      }
    }
  }
}

struct NotificationPreferences: Codable, Equatable {
  var collision: Bool
  var driverDrowsiness: Bool
  var speedLimit: Bool
  var drunkDriving: Bool
  var fsd: Bool

  enum CodingKeys: String, CodingKey {
    case collision
    case driverDrowsiness = "driver_drowsiness"
    case speedLimit = "speed_limit"
    case drunkDriving = "drunk_driving"
    case fsd
  }

  static var allEnabled: NotificationPreferences {
    NotificationPreferences(
      collision: true,
      driverDrowsiness: true,
      speedLimit: true,
      drunkDriving: true,
      fsd: true
    )
  }
}

struct UserProfile: Codable, Identifiable, Equatable {
  var id: UUID { userId }
  let userId: UUID
  let createdAt: Date
  let updatedAt: Date
  let displayName: String?
  let avatarPath: String?
  let notificationPreferences: NotificationPreferences?
  let notificationsEnabled: Bool?
  let pushToken: String?

  enum CodingKeys: String, CodingKey {
    case userId = "user_id"
    case createdAt = "created_at"
    case updatedAt = "updated_at"
    case displayName = "display_name"
    case avatarPath = "avatar_path"
    case notificationPreferences = "notification_preferences"
    case notificationsEnabled = "notifications_enabled"
    case pushToken = "push_token"
  }

  /// Returns true if the profile needs setup (no display name set)
  var needsSetup: Bool {
    displayName == nil || displayName?.isEmpty == true
  }
}

enum PiConnectivityState: String {
  case online
  case inactive
  case offline
}

struct LocalTripPersistPayload {
  let id: UUID
  let createdAt: Date
  let vehicleId: String
  let sessionId: UUID
  let startedAt: Date
  let endedAt: Date
  let status: String
  let maxSpeedMph: Int
  let avgSpeedMph: Double
  let maxIntoxicationScore: Int
  let speedingEventCount: Int
  let speedSampleCount: Int
  let speedSampleSum: Int
  let phoneDistractionEventCount: Int
  let drinkingEventCount: Int
  let routeWaypoints: [RouteWaypoint]
}

// MARK: - SupabaseService

@Observable
class SupabaseService {
  let client: SupabaseClient

  // Auth state
  var isLoggedIn = false
  var isLoading = true
  var isRefreshingSession = false
  var currentUser: User?

  // User profile state
  var userProfile: UserProfile?

  // Push notification token
  var deviceToken: String?

  /// Returns true if the user needs to set up their profile (first-time user or no display name)
  var needsProfileSetup: Bool {
    userProfile?.needsSetup ?? true
  }

  // Vehicle state
  var vehicles: [Vehicle] = []
  var vehicleRealtimeData: [String: VehicleRealtime] = [:]

  // Realtime channel
  private var realtimeChannel: RealtimeChannelV2?

  // BLE relay dedup state
  private var lastRelayedRealtimePacketAt: [String: Date] = [:]
  private var lastRelayedTripSignature: [String: String] = [:]

  // Cache
  private var modelContext: ModelContext?

  init() {
    let configuredURL =
      URL(string: Constants.Supabase.supabaseURL)
      ?? URL(string: "https://placeholder.supabase.co")!

    if URL(string: Constants.Supabase.supabaseURL) == nil {
      print("Invalid SUPABASE_URL in Constants.swift. Using placeholder URL so app can still launch.")
    }

    self.client = SupabaseClient(
      supabaseURL: configuredURL,
      supabaseKey: Constants.Supabase.supabasePublishableKey,
      options: SupabaseClientOptions(
        db: .init(decoder: JSONDecoder.supabaseDecoder),
        auth: .init(emitLocalSessionAsInitialSession: true)  // see https://github.com/supabase/supabase-swift/pull/822
      )
    )

    Task {
      await listenToAuthChanges()
    }
  }

  // MARK: - Cache Methods

  func configureCache(modelContext: ModelContext) {
    self.modelContext = modelContext
  }

  /// Loads cached vehicles and realtime data from SwiftData into the in-memory arrays.
  func loadCachedData() {
    loadCachedVehicles()
    loadCachedVehicleRealtimeData()
  }

  private func loadCachedVehicles() {
    guard let modelContext else { return }
    do {
      let descriptor = FetchDescriptor<CachedVehicle>(
        sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]
      )
      let cached = try modelContext.fetch(descriptor)
      if !cached.isEmpty {
        self.vehicles = cached.map { $0.toVehicle() }
      }
    } catch {
      print("Error loading cached vehicles: \(error)")
    }
  }

  private func saveCachedVehicles(_ vehicles: [Vehicle]) {
    guard let modelContext else { return }
    do {
      let descriptor = FetchDescriptor<CachedVehicle>()
      let existing = try modelContext.fetch(descriptor)
      let existingById = Dictionary(
        uniqueKeysWithValues: existing.map { ($0.id, $0) }
      )

      let currentIds = Set(vehicles.map(\.id))

      // Delete cached vehicles no longer in the list
      for cached in existing where !currentIds.contains(cached.id) {
        modelContext.delete(cached)
      }

      // Insert or update
      for vehicle in vehicles {
        if let cached = existingById[vehicle.id] {
          cached.update(from: vehicle)
        } else {
          modelContext.insert(CachedVehicle(from: vehicle))
        }
      }

      try modelContext.save()
    } catch {
      print("Error saving cached vehicles: \(error)")
    }
  }

  private func loadCachedVehicleRealtimeData() {
    guard let modelContext else { return }
    do {
      let descriptor = FetchDescriptor<CachedVehicleRealtime>()
      let cached = try modelContext.fetch(descriptor)
      for item in cached {
        self.vehicleRealtimeData[item.vehicleId] = item.toVehicleRealtime()
      }
    } catch {
      print("Error loading cached realtime data: \(error)")
    }
  }

  private func saveCachedVehicleRealtime(_ data: VehicleRealtime) {
    guard let modelContext else { return }
    do {
      let vehicleId = data.vehicleId
      var descriptor = FetchDescriptor<CachedVehicleRealtime>(
        predicate: #Predicate { $0.vehicleId == vehicleId }
      )
      descriptor.fetchLimit = 1
      let existing = try modelContext.fetch(descriptor)

      if let cached = existing.first {
        cached.update(from: data)
      } else {
        modelContext.insert(CachedVehicleRealtime(from: data))
      }

      try modelContext.save()
    } catch {
      print("Error saving cached realtime data: \(error)")
    }
  }

  private func deleteAllCachedData() {
    guard let modelContext else { return }
    do {
      try modelContext.delete(model: CachedVehicle.self)
      try modelContext.delete(model: CachedVehicleRealtime.self)
      try modelContext.save()
    } catch {
      print("Error clearing cache: \(error)")
    }
  }

  // MARK: - Auth Methods

  private func listenToAuthChanges() async {
    for await state in client.auth.authStateChanges {
      await MainActor.run {
        switch state.event {
        case .initialSession, .signedIn, .tokenRefreshed:
          if let session = state.session, !session.isExpired {
            self.currentUser = session.user
            self.isLoggedIn = true
            self.isRefreshingSession = false
          } else if state.event == .initialSession {
            self.currentUser = nil
            self.isLoggedIn = false
            self.isRefreshingSession = false
          }
          self.isLoading = false

          if self.isLoggedIn {
            Task { await self.loadUserProfile() }
          } else {
            self.userProfile = nil
          }

          // Hobby/public mode: always load vehicles regardless of auth state.
          Task { await self.loadVehicles() }

        case .signedOut:
          self.currentUser = nil
          self.isLoggedIn = false
          self.isLoading = false
          self.isRefreshingSession = false
          self.userProfile = nil
          self.vehicleRealtimeData = [:]
          self.unsubscribeFromRealtime()
          self.deleteAllCachedData()

          Task { await self.loadVehicles() }

        default:
          break
        }
      }
    }
  }

  func loadOrCreateUser(userId: UUID, email: String, fullName: String? = nil) async {
    await MainActor.run {
      self.isLoggedIn = true
    }
    await loadUserProfile(initialDisplayName: fullName)
    await loadVehicles()
  }

  func signOut() async throws {
    try await client.auth.signOut()
    await MainActor.run {
      self.isLoggedIn = false
      self.currentUser = nil
      self.userProfile = nil
      self.vehicleRealtimeData = [:]
      self.deleteAllCachedData()
    }
    await loadVehicles()
  }

  // MARK: - User Profile Methods

  /// Loads the user's profile, creating one if it doesn't exist
  /// - Parameter initialDisplayName: Optional display name to set when creating a new profile.
  func loadUserProfile(initialDisplayName: String? = nil) async {
    guard let userId = currentUser?.id else { return }

    do {
      // Try to fetch existing profile
      let profiles: [UserProfile] =
        try await client
        .from("user_profiles")
        .select()
        .eq("user_id", value: userId)
        .execute()
        .value

      if let existingProfile = profiles.first {
        // If profile exists but has no display name, and we have one to set, update it
        if existingProfile.displayName == nil || existingProfile.displayName?.isEmpty == true,
          let initialDisplayName, !initialDisplayName.isEmpty
        {
          try? await updateUserProfile(displayName: initialDisplayName)
        } else {
          await MainActor.run {
            self.userProfile = existingProfile
          }
        }
      } else {
        // Create a new profile if one doesn't exist
        var insertData: [String: String] = ["user_id": userId.uuidString]
        if let initialDisplayName, !initialDisplayName.isEmpty {
          insertData["display_name"] = initialDisplayName
        }

        do {
          let newProfile: UserProfile =
            try await client
            .from("user_profiles")
            .insert(insertData)
            .select()
            .single()
            .execute()
            .value

          await MainActor.run {
            self.userProfile = newProfile
          }
        } catch {
          // Handle race condition: profile may have been created by another call
          let existingProfiles: [UserProfile] =
            try await client
            .from("user_profiles")
            .select()
            .eq("user_id", value: userId)
            .execute()
            .value

          if let profile = existingProfiles.first {
            await MainActor.run {
              self.userProfile = profile
            }
          }
        }
      }
    } catch {
      print("Error loading user profile: \(error)")
    }
  }

  /// Updates the user's profile with the provided values
  func updateUserProfile(
    displayName: String? = nil,
    avatarPath: String? = nil,
    notificationPreferences: NotificationPreferences? = nil,
    notificationsEnabled: Bool? = nil,
    pushToken: String? = nil
  ) async throws {
    guard let userId = currentUser?.id else { return }

    var updateData: [String: AnyJSON] = [:]

    if let displayName {
      updateData["display_name"] = .string(displayName)
    }
    if let avatarPath {
      updateData["avatar_path"] = .string(avatarPath)
    }
    if let notificationsEnabled {
      updateData["notifications_enabled"] = .bool(notificationsEnabled)
    }
    if let pushToken {
      updateData["push_token"] = .string(pushToken)
    }
    if let notificationPreferences {
      let prefsData = try JSONEncoder().encode(notificationPreferences)
      let prefsJSON = try JSONDecoder().decode(AnyJSON.self, from: prefsData)
      updateData["notification_preferences"] = prefsJSON
    }

    let profile: UserProfile =
      try await client
      .from("user_profiles")
      .update(updateData)
      .eq("user_id", value: userId)
      .select()
      .single()
      .execute()
      .value

    await MainActor.run {
      self.userProfile = profile
    }
  }

  /// Uploads a user avatar image and returns the storage path
  func uploadUserAvatar(imageData: Data) async throws -> String {
    guard let userId = currentUser?.id else {
      throw NSError(
        domain: "SupabaseService", code: 401,
        userInfo: [NSLocalizedDescriptionKey: "User not authenticated"])
    }

    let fileName = "\(userId.uuidString)/avatar.jpg"

    // Remove existing avatar if it exists (upsert)
    try? await client.storage
      .from("user-avatars")
      .remove(paths: [fileName])

    // Upload new avatar
    try await client.storage
      .from("user-avatars")
      .upload(fileName, data: imageData, options: .init(contentType: "image/jpeg", upsert: true))

    return fileName
  }

  /// Gets the public URL for a user avatar
  func getUserAvatarURL(path: String) -> URL? {
    try? client.storage
      .from("user-avatars")
      .getPublicURL(path: path)
  }

  // MARK: - Vehicle Methods

  func loadVehicles() async {
    do {
      let vehicleList: [Vehicle] =
        try await client
        .from("vehicles")
        .select()
        .order("updated_at", ascending: false)
        .execute()
        .value

      await MainActor.run {
        self.vehicles = vehicleList
        self.saveCachedVehicles(vehicleList)
      }
    } catch {
      print("Error loading public vehicles: \(error)")
    }
  }

  func loadVehicleRealtimeData(vehicleId: String) async {
    // BLE takes priority — skip Supabase fetch when connected
    if bluetooth.isConnected {
      return
    }

    do {
      let realtimeData: VehicleRealtime =
        try await client
        .from("vehicle_realtime")
        .select()
        .eq("vehicle_id", value: vehicleId)
        .single()
        .execute()
        .value

      await MainActor.run {
        self.vehicleRealtimeData[vehicleId] = realtimeData
        self.saveCachedVehicleRealtime(realtimeData)
      }
    } catch {
      print("Error loading vehicle realtime data: \(error)")
    }
  }

  /// Feed BLE realtime data into the shared vehicleRealtimeData dict.
  /// Call this from a timer or observation when BLE data arrives.
  func updateFromBLE(vehicleId: String) {
    guard bluetooth.isConnected, let bleData = bluetooth.latestRealtime else { return }
    let realtime = bleData.toVehicleRealtime(vehicleId: vehicleId)
    self.vehicleRealtimeData[vehicleId] = realtime
    self.saveCachedVehicleRealtime(realtime)
  }

  func piConnectivityState(for vehicleId: String) -> PiConnectivityState {
    if bluetooth.connectedVehicleId == vehicleId {
      // BLE path takes precedence whenever BLE mode is enabled for this selected vehicle.
      if bluetooth.bleEnabled {
        if !bluetooth.isConnected {
          return .offline
        }
        switch bluetooth.piConnectionState {
        case .online: return .online
        case .inactive: return .inactive
        case .offline: return .offline
        }
      }
      // If BLE is no longer enabled but stale BLE data exists, treat as offline.
      if bluetooth.lastDataReceivedAt != nil && !bluetooth.isConnected {
        return .offline
      }
    }

    guard let realtime = vehicleRealtimeData[vehicleId] else { return .offline }
    let age = Date.now.timeIntervalSince(realtime.updatedAt)
    return age <= 10 ? .online : .offline
  }

  // MARK: - BLE Relay (upload Pi data to Supabase on its behalf)

  /// Relay BLE data to Supabase. Called by the relay timer when BLE is connected.
  /// The iOS app builds deterministic relay records from compact BLE caches.
  func relayBLEDataToSupabase(vehicleId: String) async {
    guard bluetooth.isConnected else { return }

    // Realtime upsert from compact BLE packet.
    if let realtime = bluetooth.latestRealtime,
      let packetAt = bluetooth.lastDataReceivedAt
    {
      if lastRelayedRealtimePacketAt[vehicleId] != packetAt {
        let record = Self.realtimeRelayRecord(
          vehicleId: vehicleId,
          data: realtime,
          updatedAt: packetAt
        )
        do {
          try await client
            .from("vehicle_realtime")
            .upsert(record)
            .execute()
          lastRelayedRealtimePacketAt[vehicleId] = packetAt
        } catch {
          print("[BLE Relay] Error relaying realtime: \(error)")
        }
      }
    }

    // Trip upsert from compact BLE trip summary (dedup by payload signature).
    if let trip = bluetooth.latestTrip {
      let signature = Self.tripSignature(vehicleId: vehicleId, data: trip)
      if lastRelayedTripSignature[vehicleId] != signature {
        let record = Self.tripRelayRecord(vehicleId: vehicleId, data: trip)
        do {
          try await client
            .from("vehicle_trips")
            .upsert(record)
            .execute()
          lastRelayedTripSignature[vehicleId] = signature
        } catch {
          print("[BLE Relay] Error relaying trip: \(error)")
        }
      }
    }
  }

  private static let relayTimeFormatter: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f
  }()

  private static func realtimeRelayRecord(
    vehicleId: String,
    data: BLERealtimeData,
    updatedAt: Date
  ) -> [String: AnyJSON] {
    [
      "vehicle_id": .string(vehicleId),
      "updated_at": .string(relayTimeFormatter.string(from: updatedAt)),
      "latitude": .double(data.lat),
      "longitude": .double(data.lng),
      "speed_mph": .double(Double(data.spd)),
      "speed_limit_mph": .double(0),
      "heading_degrees": .double(Double(data.hdg)),
      "compass_direction": .string(data.dir),
      "is_speeding": .bool(data.sp),
      "is_moving": .bool(data.spd > 0),
      "driver_status": .string(data.ds),
      "intoxication_score": .double(Double(data.ix)),
      "satellites": .double(Double(data.sat)),
      "is_phone_detected": .bool(data.ph),
      "is_drinking_detected": .bool(data.dr),
    ]
  }

  private static func tripRelayRecord(vehicleId: String, data: BLETripData) -> [String: AnyJSON] {
    let now = Date.now
    let startedAt = estimatedTripStart(from: data.tid, durationSeconds: data.dur, now: now)
    let tripId = stableUUID(seed: "ble-trip:\(vehicleId):\(data.tid)")
    let sessionId = stableUUID(seed: "ble-session:\(vehicleId):\(data.tid)")
    let sampleCount = max(1, data.dur * 2)
    let sampleSum = Int((data.avg_spd * Double(sampleCount)).rounded())

    return [
      "id": .string(tripId.uuidString.lowercased()),
      "vehicle_id": .string(vehicleId),
      "session_id": .string(sessionId.uuidString.lowercased()),
      "started_at": .string(relayTimeFormatter.string(from: startedAt)),
      "status": .string(tripStatus(from: data)),
      "max_speed_mph": .double(Double(data.mx_spd)),
      "avg_speed_mph": .double(data.avg_spd),
      "max_intoxication_score": .double(Double(data.ix_max)),
      "speeding_event_count": .double(Double(data.spd_ev)),
      "speed_sample_count": .double(Double(sampleCount)),
      "speed_sample_sum": .double(Double(sampleSum)),
      "phone_distraction_event_count": .double(Double(data.ph_ev)),
      "drinking_event_count": .double(Double(data.drw_ev)),
      "ended_at": .null,
    ]
  }

  private static func tripStatus(from data: BLETripData) -> String {
    if data.ix_max >= 4 {
      return "danger"
    }
    if data.ix_max >= 2 || data.spd_ev > 0 || data.ph_ev > 0 || data.drw_ev > 0 {
      return "warning"
    }
    return "ok"
  }

  private static func estimatedTripStart(
    from tripId: String,
    durationSeconds: Int,
    now: Date
  ) -> Date {
    if let unixString = tripId.split(separator: "-").last,
      let unixSeconds = Double(unixString)
    {
      let parsed = Date(timeIntervalSince1970: unixSeconds)
      return parsed > now ? now : parsed
    }
    return now.addingTimeInterval(-Double(max(0, durationSeconds)))
  }

  private static func tripSignature(vehicleId: String, data: BLETripData) -> String {
    "\(vehicleId)|\(data.tid)|\(data.dur)|\(data.mx_spd)|\(data.avg_spd)|\(data.spd_ev)|\(data.drw_ev)|\(data.ph_ev)|\(data.ix_max)"
  }

  private static func stableUUID(seed: String) -> UUID {
    let digest = SHA256.hash(data: Data(seed.utf8))
    let bytes = Array(digest)
    let uuidBytes: uuid_t = (
      bytes[0], bytes[1], bytes[2], bytes[3],
      bytes[4], bytes[5], (bytes[6] & 0x0F) | 0x50, bytes[7],
      (bytes[8] & 0x3F) | 0x80, bytes[9], bytes[10], bytes[11],
      bytes[12], bytes[13], bytes[14], bytes[15]
    )
    return UUID(uuid: uuidBytes)
  }

  struct BuzzerRelayState {
    let active: Bool
    let type: String
  }

  /// Fetch buzzer state from Supabase to relay back to Pi via BLE.
  /// Returns non-nil only when the buzzer state has changed.
  func fetchBuzzerStateForRelay(vehicleId: String) async -> BuzzerRelayState? {
    do {
      struct BuzzerRow: Decodable {
        let buzzerActive: Bool?
        let buzzerType: String?

        enum CodingKeys: String, CodingKey {
          case buzzerActive = "buzzer_active"
          case buzzerType = "buzzer_type"
        }
      }

      let rows: [BuzzerRow] =
        try await client
        .from("vehicle_realtime")
        .select("buzzer_active, buzzer_type")
        .eq("vehicle_id", value: vehicleId)
        .execute()
        .value

      guard let row = rows.first, let active = row.buzzerActive else { return nil }
      let type = row.buzzerType ?? "alert"

      // Only relay if state changed from what we last relayed
      let key = "lastRelayedBuzzer_\(vehicleId)"
      let lastState = UserDefaults.standard.bool(forKey: key)
      if active != lastState {
        UserDefaults.standard.set(active, forKey: key)
        return BuzzerRelayState(active: active, type: type)
      }
      return nil
    } catch {
      print("[BLE Relay] Error fetching buzzer state: \(error)")
      return nil
    }
  }

  // MARK: - Vehicle Update Methods

  /// Updates a vehicle's mutable fields
  func updateVehicle(
    vehicleId: String,
    name: String? = nil,
    description: String? = nil,
    enableYolo: Bool? = nil,
    enableMicrophone: Bool? = nil,
    enableCamera: Bool? = nil,
    enableDashcam: Bool? = nil
  ) async throws {
    var updateData: [String: AnyJSON] = [:]

    if let name {
      updateData["name"] = .string(name)
    }
    if let description {
      updateData["description"] = .string(description)
    }
    if let enableYolo {
      updateData["enable_yolo"] = .bool(enableYolo)
    }
    if let enableMicrophone {
      updateData["enable_microphone"] = .bool(enableMicrophone)
    }
    if let enableCamera {
      updateData["enable_camera"] = .bool(enableCamera)
    }
    if let enableDashcam {
      updateData["enable_dashcam"] = .bool(enableDashcam)
    }

    guard !updateData.isEmpty else { return }

    let updatedVehicles: [Vehicle] =
      try await client
      .from("vehicles")
      .update(updateData)
      .eq("id", value: vehicleId)
      .select()
      .execute()
      .value

    await MainActor.run {
      if let updatedVehicle = updatedVehicles.first,
        let index = self.vehicles.firstIndex(where: { $0.id == vehicleId })
      {
        self.vehicles[index] = updatedVehicle
      }
      self.saveCachedVehicles(self.vehicles)
    }
  }

  // MARK: - Vehicle Trip Methods

  func fetchTrips(for vehicleId: String, limit: Int = 50) async throws -> [VehicleTrip] {
    let trips: [VehicleTrip] =
      try await client
      .from("vehicle_trips")
      .select()
      .eq("vehicle_id", value: vehicleId)
      .order("started_at", ascending: false)
      .limit(limit)
      .execute()
      .value

    return trips
  }

  func fetchTripsForToday(for vehicleId: String) async throws -> [VehicleTrip] {
    let calendar = Calendar.current
    let startOfDay = calendar.startOfDay(for: Date())

    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    let startOfDayString = formatter.string(from: startOfDay)

    let trips: [VehicleTrip] =
      try await client
      .from("vehicle_trips")
      .select()
      .eq("vehicle_id", value: vehicleId)
      .gte("started_at", value: startOfDayString)
      .order("started_at", ascending: false)
      .execute()
      .value

    return trips
  }

  func saveLocalTrip(_ payload: LocalTripPersistPayload) {
    guard let modelContext else { return }
    do {
      modelContext.insert(CachedLocalTrip(from: payload))
      try modelContext.save()
    } catch {
      print("Error saving local trip: \(error)")
    }
  }

  func fetchLocalTrips(for vehicleId: String, limit: Int = 50) -> [VehicleTrip] {
    guard let modelContext else { return [] }
    do {
      var descriptor = FetchDescriptor<CachedLocalTrip>(
        predicate: #Predicate { $0.vehicleId == vehicleId },
        sortBy: [SortDescriptor(\.startedAt, order: .reverse)]
      )
      descriptor.fetchLimit = limit
      return try modelContext.fetch(descriptor).map { $0.toVehicleTrip() }
    } catch {
      print("Error loading local trips: \(error)")
      return []
    }
  }

  func fetchLocalTripsForToday(for vehicleId: String) -> [VehicleTrip] {
    guard let modelContext else { return [] }
    let startOfDay = Calendar.current.startOfDay(for: Date())
    do {
      let descriptor = FetchDescriptor<CachedLocalTrip>(
        predicate: #Predicate {
          $0.vehicleId == vehicleId && $0.startedAt >= startOfDay
        },
        sortBy: [SortDescriptor(\.startedAt, order: .reverse)]
      )
      return try modelContext.fetch(descriptor).map { $0.toVehicleTrip() }
    } catch {
      print("Error loading today's local trips: \(error)")
      return []
    }
  }

  func fetchCombinedTrips(for vehicleId: String, limit: Int = 50) async throws -> [VehicleTrip] {
    let remote = try await fetchTrips(for: vehicleId, limit: limit)
    let local = fetchLocalTrips(for: vehicleId, limit: limit)
    var mergedById: [UUID: VehicleTrip] = [:]
    for trip in remote { mergedById[trip.id] = trip }
    for trip in local { mergedById[trip.id] = trip }
    return mergedById.values.sorted { $0.startedAt > $1.startedAt }
  }

  func fetchCombinedTripsForToday(for vehicleId: String) async throws -> [VehicleTrip] {
    let remote = try await fetchTripsForToday(for: vehicleId)
    let local = fetchLocalTripsForToday(for: vehicleId)
    var mergedById: [UUID: VehicleTrip] = [:]
    for trip in remote { mergedById[trip.id] = trip }
    for trip in local { mergedById[trip.id] = trip }
    return mergedById.values.sorted { $0.startedAt > $1.startedAt }
  }

  // MARK: - Realtime Subscription

  func subscribeToVehicleRealtime(vehicleId: String) async {
    // Unsubscribe from existing channel
    unsubscribeFromRealtime()

    // Load initial realtime data
    await loadVehicleRealtimeData(vehicleId: vehicleId)

    // If BLE is connected, data flows via the BLE polling timer — skip Supabase realtime
    guard !bluetooth.isConnected else { return }

    do {
      let channel = client.realtimeV2.channel("vehicle_realtime_\(vehicleId)")

      let changes = channel.postgresChange(
        AnyAction.self,
        schema: "public",
        table: "vehicle_realtime",
        filter: .eq("vehicle_id", value: vehicleId)
      )

      try await channel.subscribeWithError()

      self.realtimeChannel = channel

      // Listen for changes
      Task {
        for await change in changes {
          await handleRealtimeChange(change)
        }
      }
    } catch {
      print("Realtime subscription failed (BLE may provide data): \(error)")
    }
  }

  private func handleRealtimeChange(_ change: AnyAction) async {
    do {
      switch change {
      case .insert(let action):
        let data = try action.decodeRecord(
          as: VehicleRealtime.self, decoder: JSONDecoder.supabaseDecoder)
        await MainActor.run {
          if self.vehicles.contains(where: { $0.id == data.vehicleId }) {
            self.vehicleRealtimeData[data.vehicleId] = data
            self.saveCachedVehicleRealtime(data)
          }
        }
      case .update(let action):
        let data = try action.decodeRecord(
          as: VehicleRealtime.self, decoder: JSONDecoder.supabaseDecoder)
        await MainActor.run {
          if self.vehicles.contains(where: { $0.id == data.vehicleId }) {
            self.vehicleRealtimeData[data.vehicleId] = data
            self.saveCachedVehicleRealtime(data)
          }
        }
      case .delete:
        break
      }
    } catch {
      print("Error decoding realtime change: \(error)")
    }
  }

  func unsubscribeFromRealtime() {
    Task {
      await realtimeChannel?.unsubscribe()
      realtimeChannel = nil
    }
  }
}

// MARK: - JSON Decoder Extension

extension JSONDecoder {
  static var supabaseDecoder: JSONDecoder {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .custom { decoder in
      let container = try decoder.singleValueContainer()
      let dateString = try container.decode(String.self)

      // Try ISO8601 with fractional seconds
      let formatter = ISO8601DateFormatter()
      formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
      if let date = formatter.date(from: dateString) {
        return date
      }

      // Try without fractional seconds
      formatter.formatOptions = [.withInternetDateTime]
      if let date = formatter.date(from: dateString) {
        return date
      }

      throw DecodingError.dataCorruptedError(
        in: container,
        debugDescription: "Cannot decode date: \(dateString)"
      )
    }
    return decoder
  }
}

// MARK: - Environment Key

struct SupabaseServiceKey: EnvironmentKey {
  static let defaultValue = SupabaseService()
}

extension EnvironmentValues {
  var supabaseService: SupabaseService {
    get { self[SupabaseServiceKey.self] }
    set { self[SupabaseServiceKey.self] = newValue }
  }
}

// Global instance for convenience
let supabase = SupabaseService()
