//
//  DeepLinkManager.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/21/26.
//

import SwiftUI
import UIKit

/// Manages deep link handling for the app
@MainActor
@Observable
final class DeepLinkManager {
  static let shared = DeepLinkManager()

  /// Whether to show the JoinVehicleView sheet
  var showJoinVehicle = false

  /// Vehicle ID to open directly (from quick action)
  var pendingVehicleId: String?

  private init() {}

  /// Handles incoming URLs from deep links and universal links
  /// - Parameter url: The URL to handle
  func handleURL(_ url: URL) {
    // Handle custom URL scheme: infineon://
    if url.scheme == "infineon" {
      handleInfineonScheme(url)
      return
    }

    // Handle shortcut item types (e.g., com.aaronhma.InfineonProject.add-vehicle)
    if url.absoluteString.contains("add-vehicle") {
      showJoinVehicle = true
      return
    }
  }

  /// Handles Quick Action shortcut items directly
  /// - Parameters:
  ///   - shortcutType: The shortcut item type string
  ///   - userInfo: Optional user info dictionary from the shortcut item
  func handleShortcutItem(_ shortcutType: String, userInfo: [String: NSSecureCoding]? = nil) {
    switch shortcutType {
    case "com.aaronhma.InfineonProject.add-vehicle":
      showJoinVehicle = true

    case "openVehicle":
      if let vehicleId = userInfo?["vehicleId"] as? String {
        pendingVehicleId = vehicleId
      }

    default:
      break
    }
  }

  /// Clears the pending vehicle ID after it has been handled
  func clearPendingVehicle() {
    pendingVehicleId = nil
  }

  private func handleInfineonScheme(_ url: URL) {
    guard let host = url.host else { return }

    switch host {
    case "add-vehicle":
      showJoinVehicle = true

    case "feedback":
      openFeedbackEmail()

    default:
      break
    }
  }

  /// Opens the user's default mail app with pre-filled feedback information
  private func openFeedbackEmail() {
    let appName = "Infineon Project App"
    let appVersion =
      Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
    let buildNumber = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown"
    let deviceModel = deviceModelIdentifier
    let iOSVersion = UIDevice.current.systemVersion

    var components = URLComponents()
    components.scheme = "mailto"
    components.path = "hi@aaronhma.com"
    components.queryItems = [
      URLQueryItem(name: "subject", value: "\(appName) Feedback"),
      URLQueryItem(
        name: "body",
        value: """


          ---
          App: \(appName)
          Version: \(appVersion) (\(buildNumber))
          Device: \(deviceModel)
          iOS: \(iOSVersion)
          """),
    ]

    guard let mailURL = components.url else { return }
    UIApplication.shared.open(mailURL)
  }

  /// Returns the device model identifier (e.g., iPhone18,3)
  private var deviceModelIdentifier: String {
    var systemInfo = utsname()
    uname(&systemInfo)
    let machineMirror = Mirror(reflecting: systemInfo.machine)
    let identifier = machineMirror.children.reduce("") { identifier, element in
      guard let value = element.value as? Int8, value != 0 else { return identifier }
      return identifier + String(UnicodeScalar(UInt8(value)))
    }
    return identifier
  }

  /// Resets the state after the sheet is dismissed
  func resetJoinVehicleState() {
    showJoinVehicle = false
  }
}

/// Global instance for convenience
let deepLinkManager = DeepLinkManager.shared
