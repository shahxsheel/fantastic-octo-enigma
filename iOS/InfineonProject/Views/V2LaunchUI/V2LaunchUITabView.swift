//
//  V2LaunchUITabView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/16/26.
//

import SwiftUI

struct V2Profile: Identifiable {
  var id: String
  var name: String
  var icon: String
  var vehicleId: String
  var vehicle: Vehicle

  var sourceAnchorID: String {
    id + "SOURCE"
  }

  var destinationAnchorID: String {
    id + "DESTINATION"
  }

  var realtimeData: VehicleRealtime? {
    supabase.vehicleRealtimeData[vehicleId]
  }
}

var sampleVehicle = Vehicle(
  id: "BENJI123", createdAt: .now, updatedAt: .now, name: "Benji", description: "Model Y",
  ownerId: UUID())

var mockProfiles: [V2Profile] = [
  .init(
    id: "BENJI123", name: "Benji", icon: "benji", vehicleId: "BENJI123", vehicle: sampleVehicle),
  .init(
    id: "BENJI124", name: "Model Y", icon: "modelY", vehicleId: "BENJI124", vehicle: sampleVehicle),
]

@Observable
class V2AppData {
  var isSplashFinished = false
  var hideMainView = false
  var showProfileView = false
  var tabProfileRect: CGRect = .zero
  var watchingProfile: V2Profile?
  var animateProfile = false
  var fromTabBar = false
}

private struct NoAnimationButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    configuration.label
  }
}

struct V2LaunchUITabView: View {
  @Environment(V2AppData.self) private var appData

  var body: some View {
    HStack(spacing: 0) {
      GeometryReader { proxy in
        let rect = proxy.frame(
          in: .named("MAINVIEW")
        )

        if let profile = appData.watchingProfile, !appData.animateProfile {
          Image(profile.icon)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: 25, height: 25)
            .clipShape(.rect(cornerRadius: 4))
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }

        Color.clear
          .preference(
            key: RectKey.self,
            value: rect
          )
          .onPreferenceChange(RectKey.self) {
            appData.tabProfileRect = $0
          }
      }
      .frame(width: 35, height: 35)

      Text("Switch")
    }
    .padding(.vertical, 5)
    .padding(.horizontal)
    .possibleGlassEffect(in: .capsule)
    .onTapGesture {
      Haptics.impact()

      withAnimation(.bouncy) {
        appData.showProfileView = true
        appData.hideMainView = true
        appData.fromTabBar = true
      }
    }
    .simultaneousGesture(
      LongPressGesture().onEnded { _ in
        Haptics.impact()

        withAnimation(.snappy(duration: 0.1)) {
          appData.showProfileView = true
          appData.hideMainView = true
          appData.fromTabBar = true
        }
      })
  }
}

#Preview {
  @Previewable @State var appData = V2AppData()

  ZStack {
    VStack {
      Spacer(minLength: 0)

      V2LaunchUITabView()
    }
    .coordinateSpace(.named("MAINVIEW"))

    if !appData.isSplashFinished {
      ProgressView()
    }
  }
  .environment(appData)
  .preferredColorScheme(.dark)
}
