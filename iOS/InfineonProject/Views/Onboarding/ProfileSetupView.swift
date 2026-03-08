//
//  ProfileSetupView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/19/26.
//

import PhotosUI
import SwiftUI

struct OnboardingProgressView: View {
  @Binding var current: Int
  var total: Int
  var cornerRadius = CGFloat(32)

  var body: some View {
    GeometryReader {
      let width = $0.size.width

      ZStack(alignment: .trailing) {
        RoundedRectangle(cornerRadius: cornerRadius)
          .fill(.gray)

        LinearGradient(
          colors: [
            .green,
            .orange,
            .indigo,
          ],
          startPoint: .topLeading,
          endPoint: .bottomTrailing
        )
        .mask {
          HStack {
            RoundedRectangle(cornerRadius: cornerRadius)
              .frame(
                width: CGFloat(current) / CGFloat(total) * width
              )

            if current != total {
              Spacer()
            }
          }
        }
        .animation(.easeInOut, value: current)
      }
    }
    .frame(height: 10)
    .padding(.horizontal)
  }
}

struct AirBrowserAIFeature: Identifiable {
  var id = UUID()
  var name: String
  var icon: String
  var description: String
  var isEnabled: Bool
}

struct ProfileSetupView: View {
  var onComplete: () -> Void

  @State private var currentTab = TabOptions.name
  @State private var isSaving = false

  @State private var showingCropViewSheet = false
  @State private var selectedPhotoItem: PhotosPickerItem?
  @State private var selectedImage: UIImage?
  @State private var croppedImage: UIImage?
  @State private var name = ""

  @State private var enabledAirAIFeatures = [
    AirBrowserAIFeature(
      name: "Collision",
      icon: "car.side.rear.and.collision.and.car.side.front",
      description: "Notify when a car crash is detected",
      isEnabled: true
    ),
    AirBrowserAIFeature(
      name: "Driver Drowsiness",
      icon: "eye.half.closed",
      description: "Notify when the driver is drowsy",
      isEnabled: true
    ),
    AirBrowserAIFeature(
      name: "Follow Speed Limit",
      icon: "gauge.with.dots.needle.100percent",
      description:
        "Notify when speed limit is exceeded",
      isEnabled: true
    ),
    AirBrowserAIFeature(
      name: "Drunk Driving",
      icon: "wineglass.fill",
      description: "Notify when alcohol or drink driving is detected",
      isEnabled: true
    ),
    AirBrowserAIFeature(
      name: "FSD",
      icon: "car.side.fill",
      description:
        "Notify when FSD is engaged or disengaged",
      isEnabled: true
    ),
  ]

  private enum TabOptions: Int, CaseIterable {
    case name = 1
    case chooseNotifications = 2
    case allowNotifications = 3
  }

  private var currentStepUncompleted: Bool {
    switch currentTab {
    case .name:
      name.isEmpty
    case .chooseNotifications:
      false
    case .allowNotifications:
      false
    }
  }

  var body: some View {
    if currentTab != .allowNotifications {
      HStack {
        Group {
          if currentTab.rawValue > 1 {
            Button {
              if let prev = TabOptions(
                rawValue: currentTab.rawValue - 1
              ) {
                withAnimation(.bouncy) {
                  currentTab = prev
                }
              }
            } label: {
              Image(systemName: "chevron.left")
            }
            .bold()
            .foregroundStyle(.primary)
          }
        }
        .transition(.blurReplace)

        OnboardingProgressView(
          current: .constant(currentTab.rawValue),
          total: TabOptions.allCases.count
        )

        Text("\(currentTab.rawValue)/\(TabOptions.allCases.count)")
          .foregroundStyle(.secondary)
          .contentTransition(.numericText(value: 0))
      }
      .padding(.horizontal)
    }

    Group {
      switch currentTab {
      case .name:
        nameView()
      case .chooseNotifications:
        chooseNotificationsView()
      case .allowNotifications:
        AllowNotificationsView(
          config: NotificationConfig(
            title: "Stay connected", content: "Get notified when important events happen",
            notificationTitle: "YO WHATS UP", notificationContent: "CLICK ME OR ELSE",
            primaryButtonTitle: "continue", secondaryButtonTitle: "skip for now"),
          fontDesignStyle: .expanded
        ) {
          Image(.benji)
            .resizable()
            .frame(width: 40, height: 40)
            .clipShape(.rect(cornerRadius: 12))
        } onPermissionChange: { isApproved in
          Task {
            await saveProfileAndComplete(notificationsEnabled: isApproved)
          }
        } onPrimaryButtonTap: {
          Task {
            await saveProfileAndComplete(notificationsEnabled: true)
          }
        } onSecondaryButtonTap: {
          Task {
            await saveProfileAndComplete(notificationsEnabled: false)
          }
        }
      }
    }
    .transition(.blurReplace)
    .frame(maxWidth: .infinity)
    .frame(maxHeight: .infinity)
    .overlay(alignment: .bottom) {
      if currentTab != .allowNotifications {
        AaronButtonView(
          text: "continue",
          opacity: currentStepUncompleted ? 0.7 : 1,
          disabled: currentStepUncompleted
        ) {
          if let next = TabOptions(
            rawValue: currentTab.rawValue + 1
          ) {
            withAnimation(.bouncy) {
              hideKeyboard()

              currentTab = next
            }
          }
        }
        .padding(.horizontal)
      }
    }
  }

  @ViewBuilder
  private func nameView() -> some View {
    ScrollView {
      VStack(alignment: .leading) {
        Text("what should we call you?")
          .fontWidth(.expanded)
          .bold()
          .font(.title)
          .multilineTextAlignment(.leading)
          .padding(.horizontal)

        VStack {
          PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
            VStack(spacing: 8) {
              Group {
                if let croppedImage {
                  Image(uiImage: croppedImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 80, height: 80)
                    .clipShape(.circle)
                } else {
                  Circle()
                    .fill(.gray.gradient)
                    .frame(width: 60, height: 60)
                    .overlay {
                      Image(systemName: "plus")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 45, height: 45)
                        .foregroundStyle(.white)
                    }
                }
              }
              .animation(.easeInOut, value: croppedImage != nil)

              Text(croppedImage != nil ? "change photo" : "add photo")
                .contentTransition(.interpolate)
            }
          }
          .padding(.bottom, 40)

          TextField("your name", text: $name)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .multilineTextAlignment(.center)

        Spacer(minLength: 0)
      }
    }
    .scrollDismissesKeyboard(.immediately)
    .onChange(of: selectedPhotoItem) { _, newItem in
      guard let newItem else { return }
      Task {
        if let data = try? await newItem.loadTransferable(type: Data.self),
          let uiImage = UIImage(data: data)
        {
          selectedImage = uiImage
        }
      }
    }
    .onChange(of: selectedImage) { _, newImage in
      if newImage != nil {
        showingCropViewSheet = true
      }
    }
    .sheet(isPresented: $showingCropViewSheet) {
      if let selectedImage {
        CropView(
          crop: .circle,
          image: selectedImage
        ) { resultImage, status in
          if status {
            croppedImage = resultImage
          }
          showingCropViewSheet = false
        }
      }
    }
    .onAppear {
      // Pre-populate name from existing user profile if available
      if let displayName = supabase.userProfile?.displayName, !displayName.isEmpty {
        name = displayName
      }
    }
  }

  @ViewBuilder
  private func chooseNotificationsView() -> some View {
    ScrollView {
      VStack(alignment: .leading) {
        Text("fire...when should we notify you?")
          .fontWidth(.expanded)
          .bold()
          .font(.title)
          .multilineTextAlignment(.leading)
          .padding(.horizontal)

        ForEach($enabledAirAIFeatures) { $i in
          VStack(alignment: .leading) {
            Toggle(isOn: $i.isEnabled) {
              Label {
                Text(i.name)
                  .bold()
              } icon: {
                SettingsBoxView(icon: i.icon, color: .indigo)
              }
            }
            .id($i.id)

            Text(i.description)
              .padding(.top, 5)
              .foregroundStyle(.secondary)
          }
          .padding(.vertical, 5)

          if i.name != "FSD" {
            Divider()
          }
        }
        .padding(.horizontal)

        Spacer(minLength: 0)
      }
      .padding(.bottom, 100)
    }
  }

  /// Saves the user profile and completes the setup
  private func saveProfileAndComplete(notificationsEnabled: Bool) async {
    guard !isSaving else { return }

    await MainActor.run {
      isSaving = true
    }

    // Build notification preferences from the enabled features
    let notificationPreferences = NotificationPreferences(
      collision: enabledAirAIFeatures.first { $0.name == "Collision" }?.isEnabled ?? true,
      driverDrowsiness: enabledAirAIFeatures.first { $0.name == "Driver Drowsiness" }?.isEnabled
        ?? true,
      speedLimit: enabledAirAIFeatures.first { $0.name == "Follow Speed Limit" }?.isEnabled ?? true,
      drunkDriving: enabledAirAIFeatures.first { $0.name == "Drunk Driving" }?.isEnabled ?? true,
      fsd: enabledAirAIFeatures.first { $0.name == "FSD" }?.isEnabled ?? true
    )

    // Register for remote notifications if enabled
    var pushToken: String?
    if notificationsEnabled {
      await UIApplication.shared.registerForRemoteNotifications()
      // Wait briefly for the token to be received
      try? await Task.sleep(for: .milliseconds(500))
      pushToken = supabase.deviceToken
    }

    do {
      // Upload avatar if user selected one
      var avatarPath: String?
      if let croppedImage,
        let imageData = croppedImage.jpegData(compressionQuality: 0.8)
      {
        avatarPath = try await supabase.uploadUserAvatar(imageData: imageData)
      }

      try await supabase.updateUserProfile(
        displayName: name,
        avatarPath: avatarPath,
        notificationPreferences: notificationPreferences,
        notificationsEnabled: notificationsEnabled,
        pushToken: pushToken
      )

      await MainActor.run {
        isSaving = false
        onComplete()
      }
    } catch {
      print("Error saving profile: \(error)")
      await MainActor.run {
        isSaving = false
        // Still complete even on error - the profile will be set up on next launch
        onComplete()
      }
    }
  }
}

#Preview {
  ProfileSetupView {}
}
