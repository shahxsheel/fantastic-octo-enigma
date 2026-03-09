//
//  InfineonProjectApp.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/12/26.
//

import SwiftData
import SwiftUI

@main
struct InfineonProjectApp: App {
  @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
  @Environment(\.scenePhase) private var scenePhase
  @AppStorage(
    "lastSelectedVehicleId"
  ) private var lastSelectedVehicleId: String?

  private func handleShortcut(url: URL) {
    deepLinkManager.handleURL(url)
  }

  var sharedModelContainer: ModelContainer = {
    let schema = Schema([
      CachedVehicle.self,
      CachedVehicleRealtime.self,
    ])
    let persistentConfiguration = ModelConfiguration(
      schema: schema,
      isStoredInMemoryOnly: false
    )
    let inMemoryConfiguration = ModelConfiguration(
      schema: schema,
      isStoredInMemoryOnly: true
    )

    do {
      return try ModelContainer(
        for: schema,
        configurations: [persistentConfiguration]
      )
    } catch {
      print("Failed to open persistent SwiftData store: \(error). Falling back to in-memory store.")
      do {
        return try ModelContainer(
          for: schema,
          configurations: [inMemoryConfiguration]
        )
      } catch {
        fatalError("Could not create any ModelContainer: \(error)")
      }
    }
  }()

  func updateShortcutItems() {
    guard let vehicleId = lastSelectedVehicleId,
      let vehicle = supabase.vehicles.first(
        where: {
          $0.id == vehicleId
        })
    else {
      UIApplication.shared.shortcutItems = []
      return
    }

    let vehicleAction = UIApplicationShortcutItem(
      type: "openVehicle",
      localizedTitle: vehicle.name ?? "Vehicle",
      localizedSubtitle: "Open recent vehicle",
      icon: UIApplicationShortcutIcon(systemImageName: "car.fill"),
      userInfo: ["vehicleId": vehicleId as NSString]
    )
    UIApplication.shared.shortcutItems = [vehicleAction]
  }

  var body: some Scene {
    WindowGroup {
      ThemeSwitcher {
        RootView()
          .onOpenURL(perform: handleShortcut)
          .onAppear {
            supabase.configureCache(
              modelContext: sharedModelContainer.mainContext
            )
          }
      }
    }
    .modelContainer(sharedModelContainer)
    .onChange(of: scenePhase) {
      if scenePhase == .background {
        updateShortcutItems()
      }
    }
  }
}

struct RootView: View {
  @State private var showProfileSetup = false

  /// Whether we're still checking if the user has set up their profile
  private var isCheckingProfileStatus: Bool {
    supabase.isLoggedIn && supabase.userProfile == nil
  }

  var body: some View {
    Group {
      if supabase.isLoading || isCheckingProfileStatus {
        ProgressView()
          .controlSize(.extraLarge)
      } else {
        // Auth is optional for local/dev usage. Show main UI even when signed out.
        V2MainView()
      }
    }
    .transition(.blurReplace)
    .animation(.easeInOut(duration: 0.3), value: supabase.isLoggedIn)
    .animation(.easeInOut(duration: 0.3), value: supabase.isLoading)
    .animation(.easeInOut(duration: 0.3), value: isCheckingProfileStatus)
    .fullScreenCover(isPresented: $showProfileSetup) {
      ProfileSetupView {
        showProfileSetup = false
      }
    }
    .sheet(
      isPresented: .init(
        get: { deepLinkManager.showJoinVehicle },
        set: { newValue in
          if !newValue {
            deepLinkManager.resetJoinVehicleState()
          }
        }
      )
    ) {
      JoinVehicleView()
    }
    .onChange(of: supabase.userProfile) { _, userProfile in
      // Show profile setup only after profile is loaded and needs setup
      if supabase.isLoggedIn, let profile = userProfile, profile.needsSetup {
        showProfileSetup = true
      }
    }
    .onChange(of: supabase.userProfile?.displayName) { _, displayName in
      // Dismiss profile setup when profile is completed
      if displayName != nil && !displayName!.isEmpty {
        showProfileSetup = false
      }
    }
  }
}
