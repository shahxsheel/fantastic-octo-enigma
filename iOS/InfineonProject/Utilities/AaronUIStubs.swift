//
//  AaronUIStubs.swift
//  InfineonProject
//
//  Minimal replacements for the private AaronUI package.
//  These allow the project to compile during development.
//

import SwiftUI
import UIKit
import UserNotifications

// MARK: - CloseButton

struct CloseButton: View {
  private let action: () -> Void
  init(_ action: @escaping () -> Void) { self.action = action }
  var body: some View {
    Button(action: action) {
      Image(systemName: "xmark.circle.fill")
        .font(.title2)
        .foregroundStyle(.secondary)
    }
  }
}

// MARK: - SettingsBoxView

struct SettingsBoxView: View {
  let icon: String
  let color: Color
  var body: some View {
    ZStack {
      RoundedRectangle(cornerRadius: 8)
        .fill(color)
        .frame(width: 32, height: 32)
      Image(systemName: icon)
        .foregroundStyle(.white)
        .font(.system(size: 16, weight: .semibold))
    }
  }
}

// MARK: - ProgressRing

struct ProgressRing<S: ShapeStyle>: View {
  let size: CGFloat
  let lineWidth: CGFloat
  @Binding var progress: CGFloat
  let foregroundStyle: S

  var body: some View {
    ZStack {
      Circle()
        .stroke(.gray.opacity(0.25), lineWidth: lineWidth)
      Circle()
        .trim(from: 0, to: min(max(progress, 0), 1))
        .stroke(foregroundStyle, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
        .rotationEffect(.degrees(-90))
    }
    .frame(width: size, height: size)
  }
}

// MARK: - MarqueeView

struct MarqueeView<Content: View>: View {
  let content: Content
  init(@ViewBuilder content: () -> Content) { self.content = content() }
  var body: some View { content }
}

// MARK: - AaronButtonView

struct AaronButtonView: View {
  let text: String
  var opacity: Double = 1.0
  var disabled: Bool = false
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      Text(text.capitalized)
        .frame(maxWidth: .infinity)
        .padding()
        .background(disabled ? Color.secondary : Color.accentColor)
        .foregroundStyle(.white)
        .clipShape(.rect(cornerRadius: 14))
        .opacity(opacity)
    }
    .disabled(disabled)
  }
}

// MARK: - CropShape / CropView

enum CropShape { case circle, square, rectangle }

struct CropView: View {
  let crop: CropShape
  let image: UIImage
  let completion: (UIImage, Bool) -> Void
  @Environment(\.dismiss) private var dismiss

  init(crop: CropShape, image: UIImage, completion: @escaping (UIImage, Bool) -> Void) {
    self.crop = crop
    self.image = image
    self.completion = completion
  }

  var body: some View {
    VStack(spacing: 20) {
      Text("Crop Image")
        .font(.headline)
      Group {
        let img = Image(uiImage: image).resizable().scaledToFit()
        if crop == .circle {
          img.clipShape(.circle)
        } else {
          img
        }
      }
      .frame(maxHeight: 300)
      HStack(spacing: 16) {
        Button("Cancel") {
          completion(image, false)
          dismiss()
        }
        .buttonStyle(.bordered)
        Button("Use Photo") {
          completion(image, true)
          dismiss()
        }
        .buttonStyle(.borderedProminent)
      }
    }
    .padding()
  }
}

// MARK: - NotificationConfig / AllowNotificationsView

struct NotificationConfig {
  let title: String
  let content: String
  let notificationTitle: String
  let notificationContent: String
  let primaryButtonTitle: String
  let secondaryButtonTitle: String
}

struct FontDesignStyle {
  static let expanded = FontDesignStyle()
  static let standard = FontDesignStyle()
  static let condensed = FontDesignStyle()
}

struct AllowNotificationsView<Icon: View>: View {
  let config: NotificationConfig
  var fontDesignStyle: FontDesignStyle = .standard
  let iconContent: Icon
  let onPermissionChange: (Bool) -> Void
  let onPrimaryButtonTap: () -> Void
  let onSecondaryButtonTap: () -> Void

  init(
    config: NotificationConfig,
    fontDesignStyle: FontDesignStyle = .standard,
    @ViewBuilder iconContent: () -> Icon,
    onPermissionChange: @escaping (Bool) -> Void,
    onPrimaryButtonTap: @escaping () -> Void,
    onSecondaryButtonTap: @escaping () -> Void
  ) {
    self.config = config
    self.fontDesignStyle = fontDesignStyle
    self.iconContent = iconContent()
    self.onPermissionChange = onPermissionChange
    self.onPrimaryButtonTap = onPrimaryButtonTap
    self.onSecondaryButtonTap = onSecondaryButtonTap
  }

  var body: some View {
    VStack(spacing: 24) {
      iconContent
      Text(config.title).font(.title2.bold())
      Text(config.content).multilineTextAlignment(.center).foregroundStyle(.secondary)
      Button(config.primaryButtonTitle.capitalized) {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) {
          granted, _ in
          DispatchQueue.main.async { onPermissionChange(granted) }
        }
        onPrimaryButtonTap()
      }
      .buttonStyle(.borderedProminent)
      Button(config.secondaryButtonTitle.capitalized) {
        onSecondaryButtonTap()
      }
      .buttonStyle(.bordered)
    }
    .padding()
  }
}

// MARK: - RingsView

struct RingsView: View {
  let size: CGFloat
  let lineWidth: CGFloat
  @Binding var progressOuter: CGFloat
  @Binding var progressMiddle: CGFloat
  @Binding var progressInner: CGFloat

  var body: some View {
    ZStack {
      ProgressRing(size: size, lineWidth: lineWidth, progress: $progressOuter, foregroundStyle: Color.red)
      ProgressRing(size: size * 0.72, lineWidth: lineWidth, progress: $progressMiddle, foregroundStyle: Color.green)
      ProgressRing(size: size * 0.44, lineWidth: lineWidth, progress: $progressInner, foregroundStyle: Color.blue)
    }
  }
}

// MARK: - FluidZoomTransitionStyle

struct FluidZoomTransitionStyle<S: Shape>: ButtonStyle {
  let id: String
  let namespace: Namespace.ID
  let shape: S
  var applyGlass: Bool = true

  init(id: String, namespace: Namespace.ID, shape: S, applyGlass: Bool = true) {
    self.id = id
    self.namespace = namespace
    self.shape = shape
    self.applyGlass = applyGlass
  }

  func makeBody(configuration: Configuration) -> some View {
    configuration.label
      .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
      .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
  }
}

// MARK: - ConcentricRectangle

enum ConcentricCorners { case concentric }

struct ConcentricRectangle: Shape {
  let corners: ConcentricCorners
  let isUniform: Bool
  func path(in rect: CGRect) -> Path {
    RoundedRectangle(cornerRadius: 20).path(in: rect)
  }
}

// MARK: - Haptics

enum Haptics {
  static func impact(_ style: UIImpactFeedbackGenerator.FeedbackStyle = .medium) {
    UIImpactFeedbackGenerator(style: style).impactOccurred()
  }
  static func notification(_ type: UINotificationFeedbackGenerator.FeedbackType) {
    UINotificationFeedbackGenerator().notificationOccurred(type)
  }
}

// MARK: - hideKeyboard

func hideKeyboard() {
  UIApplication.shared.sendAction(
    #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

// MARK: - BackgroundShadowModifier

struct BackgroundShadowModifier: ViewModifier {
  func body(content: Content) -> some View {
    content.shadow(color: .black.opacity(0.08), radius: 6, y: 3)
  }
}

// MARK: - ButtonSizing

struct ButtonSizing {
  static let flexible = ButtonSizing()
  static let fixed = ButtonSizing()
  static let fit = ButtonSizing()
}

// MARK: - View Extensions

extension View {
  /// Wraps a glass-like effect; falls back to no-op in dev builds.
  func possibleGlassEffect(in shape: some Shape) -> some View { self }

  /// Two-arg variant: style + shape.
  func possibleGlassEffect<S: ShapeStyle>(_ style: S, in shape: some Shape) -> some View { self }

  /// AaronUI's stable matched transition (no-op stub).
  func stableMatchedTransition(id: some Hashable, in namespace: Namespace.ID) -> some View { self }

  /// Shows title in the navigation bar.
  func scrollAwareTitle(_ title: String) -> some View { navigationTitle(title) }

  /// Anchors the large title visibility; no-op stub.
  func titleVisibilityAnchor() -> some View { self }

  /// Disables screenshots; no-op in dev builds.
  func preventScreenshot() -> some View { self }

  /// Controls button sizing; no-op stub.
  func buttonSizing(_ sizing: ButtonSizing) -> some View { self }
}
