//
//  SupabaseService.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/13/26.
//

import CoreLocation
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

enum CloudRuntimeState: Equatable {
  case active
  case degraded(String)
  case disabled(String)

  var isOperational: Bool {
    switch self {
    case .disabled:
      return false
    case .active, .degraded:
      return true
    }
  }

  var statusLabel: String {
    switch self {
    case .active:
      return "Active"
    case .degraded:
      return "Degraded"
    case .disabled:
      return "Disabled"
    }
  }

  var detail: String? {
    switch self {
    case .active:
      return nil
    case .degraded(let reason), .disabled(let reason):
      return reason
    }
  }
}

enum RealtimeDataSource: String {
  case ble = "BLE"
  case supabase = "Supabase"
  case cache = "Cache"
  case unknown = "Unknown"
}

struct ConnectivityDebugStatus {
  let cloudState: CloudRuntimeState
  let lastCloudError: String?
  let activeSource: RealtimeDataSource
  let sourceAgeSeconds: TimeInterval?
  let policyLabel: String
  let fallbackReason: String?
}

// MARK: - SupabaseService

@Observable
class SupabaseService {
  let client: SupabaseClient
  let appConfig: AppConfig

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
  var cloudRuntimeState: CloudRuntimeState = .disabled("Not configured")
  var lastCloudError: String?

  // Realtime channel
  private var realtimeChannel: RealtimeChannelV2?
  private var realtimeSourceByVehicle: [String: RealtimeDataSource] = [:]
  private var realtimeUpdatedAtByVehicle: [String: Date] = [:]
  private var latestSupabaseRealtimeByVehicle: [String: VehicleRealtime] = [:]
  private var latestBLERealtimeByVehicle: [String: VehicleRealtime] = [:]
  private var realtimeCloudErrorByVehicle: [String: String] = [:]
  private var realtimeFallbackReasonByVehicle: [String: String] = [:]
  private var lastLoggedPiConnectivityStateByVehicle: [String: PiConnectivityState] = [:]
  private var lastLoggedPiConnectivitySourceByVehicle: [String: String] = [:]
  private var lastLoggedPiConnectivityAtByVehicle: [String: Date] = [:]
  private var lastCloudStaleRecoveryAttemptByVehicle: [String: Date] = [:]
  private var cloudStaleRecoveryInFlightVehicleIds: Set<String> = []

  // Cache
  private var modelContext: ModelContext?
  private static let cloudFirstFreshnessWindow: TimeInterval = 5
  private static let bleFallbackFreshnessWindow: TimeInterval = 5
  private static let piConnectivityFreshnessWindow: TimeInterval = 10
  private static let piConnectivityLogInterval: TimeInterval = 10
  private static let cloudStaleRecoveryInterval: TimeInterval = 5
  private static let connectivityPolicyLabel = "Cloud-first (BLE fallback)"

  init() {
    appConfig = .shared

    let supabaseConfig = appConfig.supabase
    if let reason = supabaseConfig.validationError {
      cloudRuntimeState = .disabled(reason)
      isLoading = false
      isRefreshingSession = false
      print("[Supabase] Disabled: \(reason) (source=\(supabaseConfig.sourceDescription))")
    } else {
      cloudRuntimeState = .active
      print("[Supabase] Configured via \(supabaseConfig.sourceDescription)")
    }

    self.client = SupabaseClient(
      supabaseURL: supabaseConfig.resolvedURL,
      supabaseKey: supabaseConfig.publishableKey ?? "placeholder-key",
      options: SupabaseClientOptions(
        db: .init(decoder: JSONDecoder.supabaseDecoder),
        auth: .init(emitLocalSessionAsInitialSession: true)  // see https://github.com/supabase/supabase-swift/pull/822
      )
    )

    if cloudRuntimeState.isOperational {
      Task {
        await listenToAuthChanges()
      }
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
        let realtime = item.toVehicleRealtime()
        self.vehicleRealtimeData[item.vehicleId] = realtime
        realtimeSourceByVehicle[item.vehicleId] = .cache
        realtimeUpdatedAtByVehicle[item.vehicleId] = realtime.updatedAt
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

  private func cloudAvailable(for operation: String) -> Bool {
    guard cloudRuntimeState.isOperational else {
      if case .disabled(let reason) = cloudRuntimeState {
        print("[Supabase][\(operation)] skipped: \(reason)")
      } else {
        print("[Supabase][\(operation)] skipped: cloud not operational")
      }
      return false
    }
    return true
  }

  private func cloudUnavailableError(_ message: String) -> NSError {
    NSError(domain: "SupabaseService", code: 503, userInfo: [NSLocalizedDescriptionKey: message])
  }

  private func markCloudError(context: String, error: Error) {
    let message = "\(context): \(error.localizedDescription)"
    lastCloudError = message
    cloudRuntimeState = .degraded(message)
    print("[Supabase][error] \(message)")
  }

  private func markCloudHealthy() {
    lastCloudError = nil
    if case .degraded = cloudRuntimeState {
      cloudRuntimeState = .active
    }
  }

  private func markRealtimeCloudError(vehicleId: String, context: String, error: Error) {
    realtimeCloudErrorByVehicle[vehicleId] = "\(context): \(error.localizedDescription)"
    recomputeRealtimeSelection(vehicleId: vehicleId)
  }

  private func markRealtimeCloudHealthy(vehicleId: String) {
    realtimeCloudErrorByVehicle.removeValue(forKey: vehicleId)
  }

  private func selectRealtime(vehicleId: String) -> (VehicleRealtime?, RealtimeDataSource, String?) {
    let now = Date.now
    let supabaseRealtime = latestSupabaseRealtimeByVehicle[vehicleId]
    let bleRealtime = latestBLERealtimeByVehicle[vehicleId]
    let supabaseAge = supabaseRealtime.map { now.timeIntervalSince($0.updatedAt) }
    let bleAge = bleRealtime.map { now.timeIntervalSince($0.updatedAt) }
    let supabaseFresh = supabaseAge.map { $0 <= Self.cloudFirstFreshnessWindow } ?? false
    let bleFresh = bleAge.map { $0 <= Self.bleFallbackFreshnessWindow } ?? false
    let cloudErrored = realtimeCloudErrorByVehicle[vehicleId] != nil || !cloudRuntimeState.isOperational

    if let supabaseRealtime, supabaseFresh {
      return (supabaseRealtime, .supabase, nil)
    }

    if let bleRealtime, bleFresh {
      return (bleRealtime, .ble, cloudErrored ? "cloud error" : "cloud stale")
    }

    if cloudErrored {
      return (nil, .unknown, "cloud error")
    }

    if supabaseRealtime != nil {
      // Cloud-first policy: do not keep stale cloud as active source.
      return (nil, .unknown, "cloud stale")
    }

    return (nil, .unknown, nil)
  }

  private func recomputeRealtimeSelection(vehicleId: String) {
    let (selected, selectedSource, fallbackReason) = selectRealtime(vehicleId: vehicleId)

    if let selected {
      vehicleRealtimeData[vehicleId] = selected
      saveCachedVehicleRealtime(selected)
      realtimeSourceByVehicle[vehicleId] = selectedSource
      realtimeUpdatedAtByVehicle[vehicleId] = selected.updatedAt
    } else {
      realtimeSourceByVehicle[vehicleId] = selectedSource
      realtimeUpdatedAtByVehicle.removeValue(forKey: vehicleId)
    }

    if selectedSource != .supabase, let fallbackReason {
      realtimeFallbackReasonByVehicle[vehicleId] = fallbackReason
    } else {
      realtimeFallbackReasonByVehicle.removeValue(forKey: vehicleId)
    }
  }

  private func applyRealtimeData(_ realtime: VehicleRealtime, source: RealtimeDataSource) {
    let vehicleId = realtime.vehicleId
    switch source {
    case .supabase:
      latestSupabaseRealtimeByVehicle[vehicleId] = realtime
      markRealtimeCloudHealthy(vehicleId: vehicleId)
    case .ble:
      latestBLERealtimeByVehicle[vehicleId] = realtime
    case .cache:
      vehicleRealtimeData[vehicleId] = realtime
      realtimeSourceByVehicle[vehicleId] = .cache
      realtimeUpdatedAtByVehicle[vehicleId] = realtime.updatedAt
      saveCachedVehicleRealtime(realtime)
      return
    case .unknown:
      return
    }

    recomputeRealtimeSelection(vehicleId: vehicleId)
  }

  func connectivityDebugStatus(vehicleId: String?) -> ConnectivityDebugStatus {
    let resolvedVehicleId = vehicleId ?? bluetooth.connectedVehicleId
    let source = resolvedVehicleId.flatMap { realtimeSourceByVehicle[$0] } ?? .unknown
    let sourceUpdatedAt = resolvedVehicleId.flatMap { realtimeUpdatedAtByVehicle[$0] }
    let age = sourceUpdatedAt.map { Date.now.timeIntervalSince($0) }
    let fallbackReason = resolvedVehicleId.flatMap { realtimeFallbackReasonByVehicle[$0] }

    return ConnectivityDebugStatus(
      cloudState: cloudRuntimeState,
      lastCloudError: lastCloudError,
      activeSource: source,
      sourceAgeSeconds: age,
      policyLabel: Self.connectivityPolicyLabel,
      fallbackReason: fallbackReason
    )
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
          self.realtimeSourceByVehicle = [:]
          self.realtimeUpdatedAtByVehicle = [:]
          self.latestSupabaseRealtimeByVehicle = [:]
          self.latestBLERealtimeByVehicle = [:]
          self.realtimeCloudErrorByVehicle = [:]
          self.realtimeFallbackReasonByVehicle = [:]
          self.lastLoggedPiConnectivityStateByVehicle = [:]
          self.lastLoggedPiConnectivitySourceByVehicle = [:]
          self.lastLoggedPiConnectivityAtByVehicle = [:]
          self.lastCloudStaleRecoveryAttemptByVehicle = [:]
          self.cloudStaleRecoveryInFlightVehicleIds = []
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
    guard cloudAvailable(for: "loadOrCreateUser") else { return }
    await MainActor.run {
      self.isLoggedIn = true
    }
    await loadUserProfile(initialDisplayName: fullName)
    await loadVehicles()
  }

  func signOut() async throws {
    guard cloudAvailable(for: "signOut") else {
      throw cloudUnavailableError("Supabase is disabled")
    }
    try await client.auth.signOut()
    await MainActor.run {
      self.isLoggedIn = false
      self.currentUser = nil
      self.userProfile = nil
      self.vehicleRealtimeData = [:]
      self.realtimeSourceByVehicle = [:]
      self.realtimeUpdatedAtByVehicle = [:]
      self.latestSupabaseRealtimeByVehicle = [:]
      self.latestBLERealtimeByVehicle = [:]
      self.realtimeCloudErrorByVehicle = [:]
      self.realtimeFallbackReasonByVehicle = [:]
      self.lastLoggedPiConnectivityStateByVehicle = [:]
      self.lastLoggedPiConnectivitySourceByVehicle = [:]
      self.lastLoggedPiConnectivityAtByVehicle = [:]
      self.lastCloudStaleRecoveryAttemptByVehicle = [:]
      self.cloudStaleRecoveryInFlightVehicleIds = []
      self.deleteAllCachedData()
    }
    markCloudHealthy()
    await loadVehicles()
  }

  // MARK: - User Profile Methods

  /// Loads the user's profile, creating one if it doesn't exist
  /// - Parameter initialDisplayName: Optional display name to set when creating a new profile.
  func loadUserProfile(initialDisplayName: String? = nil) async {
    guard cloudAvailable(for: "loadUserProfile") else { return }
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
      markCloudHealthy()
    } catch {
      markCloudError(context: "loadUserProfile", error: error)
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
    guard cloudAvailable(for: "updateUserProfile") else {
      throw cloudUnavailableError("Supabase is disabled")
    }
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
    markCloudHealthy()
  }

  /// Uploads a user avatar image and returns the storage path
  func uploadUserAvatar(imageData: Data) async throws -> String {
    guard cloudAvailable(for: "uploadUserAvatar") else {
      throw cloudUnavailableError("Supabase is disabled")
    }
    guard let userId = currentUser?.id else {
      throw NSError(
        domain: "SupabaseService", code: 401,
        userInfo: [NSLocalizedDescriptionKey: "User not authenticated"])
    }

    let fileName = "\(userId.uuidString)/avatar.jpg"

    // Remove existing avatar if it exists (upsert)
    _ = try? await client.storage
      .from("user-avatars")
      .remove(paths: [fileName])

    // Upload new avatar
    try await client.storage
      .from("user-avatars")
      .upload(fileName, data: imageData, options: .init(contentType: "image/jpeg", upsert: true))

    markCloudHealthy()
    return fileName
  }

  /// Gets the public URL for a user avatar
  func getUserAvatarURL(path: String) -> URL? {
    guard cloudAvailable(for: "getUserAvatarURL") else { return nil }
    return try? client.storage
      .from("user-avatars")
      .getPublicURL(path: path)
  }

  // MARK: - Vehicle Methods

  func loadVehicles() async {
    guard cloudAvailable(for: "loadVehicles") else { return }
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
      markCloudHealthy()
    } catch {
      markCloudError(context: "loadVehicles", error: error)
    }
  }

  func loadVehicleRealtimeData(vehicleId: String) async {
    guard cloudAvailable(for: "loadVehicleRealtimeData") else { return }

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
        self.applyRealtimeData(realtimeData, source: .supabase)
      }
      markCloudHealthy()
    } catch {
      await MainActor.run {
        self.markRealtimeCloudError(vehicleId: vehicleId, context: "loadVehicleRealtimeData", error: error)
      }
      markCloudError(context: "loadVehicleRealtimeData", error: error)
    }
  }

  /// Feed BLE realtime data into the shared vehicleRealtimeData dict.
  /// Call this from a timer or observation when BLE data arrives.
  func updateFromBLE(vehicleId: String) {
    guard bluetooth.isConnected, let bleData = bluetooth.latestRealtime else { return }
    let realtime = bleData.toVehicleRealtime(vehicleId: vehicleId)
    applyRealtimeData(realtime, source: .ble)
  }

  func piConnectivityState(for vehicleId: String) -> PiConnectivityState {
    if bluetooth.connectedVehicleId == vehicleId && bluetooth.bleEnabled {
      if bluetooth.isConnected, bluetooth.lastDataReceivedAt != nil {
        let bleState: PiConnectivityState
        switch bluetooth.piConnectionState {
        case .online: bleState = .online
        case .inactive: bleState = .inactive
        case .offline: bleState = .offline
        }
        if bleState != .offline {
          let bleAge = bluetooth.lastDataReceivedAt.map { Date.now.timeIntervalSince($0) }
          logPiConnectivityDecision(vehicleId: vehicleId, state: bleState, source: "ble", age: bleAge)
          return bleState
        }
      }
      // BLE disconnected or stale: fall through to cloud data if available.
    }

    let cloudRealtime =
      latestSupabaseRealtimeByVehicle[vehicleId]
      ?? {
        if let source = realtimeSourceByVehicle[vehicleId], source == .supabase || source == .cache {
          return vehicleRealtimeData[vehicleId]
        }
        return nil
      }()

    guard let realtime = cloudRealtime else {
      logPiConnectivityDecision(
        vehicleId: vehicleId,
        state: .offline,
        source: "cloud_missing",
        age: nil
      )
      return .offline
    }
    let age = Date.now.timeIntervalSince(realtime.updatedAt)
    let state: PiConnectivityState = age <= Self.piConnectivityFreshnessWindow ? .online : .offline
    logPiConnectivityDecision(vehicleId: vehicleId, state: state, source: "cloud", age: age)
    scheduleCloudStaleRecoveryIfNeeded(vehicleId: vehicleId, age: age)
    return state
  }

  private func logPiConnectivityDecision(
    vehicleId: String,
    state: PiConnectivityState,
    source: String,
    age: TimeInterval?
  ) {
    let now = Date.now
    let previousState = lastLoggedPiConnectivityStateByVehicle[vehicleId]
    let previousSource = lastLoggedPiConnectivitySourceByVehicle[vehicleId]
    let previousLoggedAt = lastLoggedPiConnectivityAtByVehicle[vehicleId]
    let ageText = age.map { String(format: "%.1f", max(0, $0)) } ?? "nil"
    let shouldLog =
      previousState != state
      || previousSource != source
      || previousLoggedAt == nil
      || now.timeIntervalSince(previousLoggedAt ?? .distantPast) >= Self.piConnectivityLogInterval

    guard shouldLog else { return }
    print(
      "[Supabase][pi_status] vehicle=\(vehicleId) state=\(state.rawValue) source=\(source) age_s=\(ageText)")
    lastLoggedPiConnectivityStateByVehicle[vehicleId] = state
    lastLoggedPiConnectivitySourceByVehicle[vehicleId] = source
    lastLoggedPiConnectivityAtByVehicle[vehicleId] = now
  }

  private func scheduleCloudStaleRecoveryIfNeeded(vehicleId: String, age: TimeInterval) {
    guard cloudRuntimeState.isOperational else { return }
    guard age > Self.cloudFirstFreshnessWindow else { return }
    let now = Date.now
    if cloudStaleRecoveryInFlightVehicleIds.contains(vehicleId) {
      return
    }
    if let lastAttempt = lastCloudStaleRecoveryAttemptByVehicle[vehicleId],
      now.timeIntervalSince(lastAttempt) < Self.cloudStaleRecoveryInterval
    {
      return
    }

    lastCloudStaleRecoveryAttemptByVehicle[vehicleId] = now
    cloudStaleRecoveryInFlightVehicleIds.insert(vehicleId)
    print(
      "[Supabase][stale_recovery] vehicle=\(vehicleId) source_age_s=\(String(format: "%.1f", max(0, age)))")

    Task { @MainActor [weak self] in
      guard let self else { return }
      defer { self.cloudStaleRecoveryInFlightVehicleIds.remove(vehicleId) }
      _ = await self.fetchVehicleRealtimeRow(vehicleId: vehicleId)
    }
  }

  func fetchVehicleRealtimeRow(vehicleId: String) async -> VehicleRealtime? {
    guard cloudAvailable(for: "fetchVehicleRealtimeRow") else { return nil }
    do {
      let rows: [VehicleRealtime] =
        try await client
        .from("vehicle_realtime")
        .select()
        .eq("vehicle_id", value: vehicleId)
        .execute()
        .value
      markCloudHealthy()
      if let row = rows.first {
        await MainActor.run {
          self.applyRealtimeData(row, source: .supabase)
        }
      }
      return rows.first
    } catch {
      markCloudError(context: "fetchVehicleRealtimeRow", error: error)
      return nil
    }
  }

  func setVehicleBuzzerState(vehicleId: String, active: Bool, type: String = "alert") async throws {
    guard cloudAvailable(for: "setVehicleBuzzerState") else {
      throw cloudUnavailableError("Supabase is disabled")
    }
    do {
      if active {
        try await client.rpc(
          "activate_vehicle_buzzer",
          params: [
            "p_vehicle_id": vehicleId,
            "p_buzzer_type": type,
          ]
        ).execute()
      } else {
        try await client.rpc(
          "deactivate_vehicle_buzzer",
          params: ["p_vehicle_id": vehicleId]
        ).execute()
      }
      markCloudHealthy()
    } catch {
      markCloudError(context: "setVehicleBuzzerState", error: error)
      throw error
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
    guard cloudAvailable(for: "updateVehicle") else {
      throw cloudUnavailableError("Supabase is disabled")
    }
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
    markCloudHealthy()
  }

  /// Returns true when the app should suggest the user switch to BLE.
  /// Triggered when Supabase data for the vehicle is stale (> cloudFirstFreshnessWindow)
  /// and a BLE connection is not already active.
  func needsBLEConnectionSuggestion(vehicleId: String) -> Bool {
    guard !bluetooth.isConnected else { return false }
    let supabaseRealtime = latestSupabaseRealtimeByVehicle[vehicleId]
    guard let supabaseRealtime else { return false }
    let age = Date.now.timeIntervalSince(supabaseRealtime.updatedAt)
    return age > Self.cloudFirstFreshnessWindow
  }

  // MARK: - Realtime Subscription

  func subscribeToVehicleRealtime(vehicleId: String) async {
    guard cloudAvailable(for: "subscribeToVehicleRealtime") else { return }
    // Unsubscribe from existing channel
    unsubscribeFromRealtime()

    // Load initial realtime data
    await loadVehicleRealtimeData(vehicleId: vehicleId)

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
          await handleRealtimeChange(change, vehicleId: vehicleId)
        }
      }
      markCloudHealthy()
    } catch {
      await MainActor.run {
        self.markRealtimeCloudError(vehicleId: vehicleId, context: "subscribeToVehicleRealtime", error: error)
      }
      markCloudError(context: "subscribeToVehicleRealtime", error: error)
    }
  }

  private func handleRealtimeChange(_ change: AnyAction, vehicleId: String) async {
    do {
      switch change {
      case .insert(let action):
        let data = try action.decodeRecord(
          as: VehicleRealtime.self, decoder: JSONDecoder.supabaseDecoder)
        await MainActor.run {
          if self.vehicles.contains(where: { $0.id == data.vehicleId }) {
            self.applyRealtimeData(data, source: .supabase)
          }
        }
      case .update(let action):
        let data = try action.decodeRecord(
          as: VehicleRealtime.self, decoder: JSONDecoder.supabaseDecoder)
        await MainActor.run {
          if self.vehicles.contains(where: { $0.id == data.vehicleId }) {
            self.applyRealtimeData(data, source: .supabase)
          }
        }
      case .delete:
        break
      }
      markCloudHealthy()
    } catch {
      await MainActor.run {
        self.markRealtimeCloudError(vehicleId: vehicleId, context: "handleRealtimeChange", error: error)
      }
      markCloudError(context: "handleRealtimeChange", error: error)
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
