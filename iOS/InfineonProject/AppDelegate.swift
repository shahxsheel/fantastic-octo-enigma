//
//  AppDelegate.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/19/26.
//

import UIKit

class AppDelegate: NSObject, UIApplicationDelegate {
  func application(
    _ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession,
    options: UIScene.ConnectionOptions
  ) -> UISceneConfiguration {
    let sceneConfiguration = UISceneConfiguration(
      name: "Infineon Project - App Delegate", sessionRole: connectingSceneSession.role)

    if connectingSceneSession.role == .windowApplication {
      sceneConfiguration.delegateClass = SceneDelegate.self
    }

    return sceneConfiguration
  }

  func application(
    _ application: UIApplication,
    didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
  ) {
    let token = deviceToken.map { String(format: "%02.2hhx", $0) }.joined()
    supabase.deviceToken = token
  }

  func application(
    _ application: UIApplication,
    didFailToRegisterForRemoteNotificationsWithError error: Error
  ) {
    print("Failed to register for remote notifications: \(error)")
  }
}
