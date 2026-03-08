//
//  V2ProfileSelectView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/16/26.
//

import SwiftUI

struct RectAnchorKey: PreferenceKey {
  static var defaultValue: [String: Anchor<CGRect>] = [:]
  static func reduce(
    value: inout [String: Anchor<CGRect>],
    nextValue: () -> [String: Anchor<CGRect>]
  ) {
    value.merge(nextValue()) { $1 }
  }
}

struct RectKey: PreferenceKey {
  static var defaultValue: CGRect = .zero
  static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
    value = nextValue()
  }
}

struct AnimatedPositionModifier: ViewModifier, Animatable {
  var source: CGPoint
  var center: CGPoint
  var destination: CGPoint
  var animateToCenter: Bool
  var animateToMainView: Bool
  var path: Path
  var progress: CGFloat

  var animatableData: CGFloat {
    get { progress }
    set { progress = newValue }
  }

  func body(content: Content) -> some View {
    content
      .position(
        animateToCenter
          ? animateToMainView
            ? (path
              .trimmedPath(
                from: 0,
                to: progress
              ).currentPoint ?? center) : center : source
      )
  }
}

struct V2ProfileSelectView: View {
  @Environment(\.colorScheme) private var colorScheme
  @Environment(V2AppData.self) private var appData

  @AppStorage(
    "lastSelectedVehicleId"
  ) private var lastSelectedVehicleId: String?
  @State private var animateToCenter = false
  @State private var animateToMainView = false
  @State private var progress = CGFloat.zero

  @State private var showingJoinVehicleSheet = false
  @State private var editMode: EditMode = .inactive
  @State private var selectedEditProfile: V2Profile?

  func prefetchStatus() async {
    let vehicleId = appData.watchingProfile!.vehicleId

    // Set BLE vehicle ID so the data polling timer feeds realtime data
    if bluetooth.isConnected {
      bluetooth.connectedVehicleId = vehicleId
    }

    // 1. Subscribe to realtime (will skip Supabase channel if BLE connected)
    await supabase.subscribeToVehicleRealtime(vehicleId: vehicleId)

    // 2. Load initial realtime data (from Supabase cache or BLE)
    await supabase.loadVehicleRealtimeData(vehicleId: vehicleId)
  }

  func animateCard() async {
    withAnimation(.bouncy(duration: 0.35)) {
      animateToCenter = true
    }

    await prefetchStatus()

    withAnimation(
      .snappy(duration: 0.6, extraBounce: 0.1),
      completionCriteria: .removed
    ) {
      animateToMainView = true
      appData.hideMainView = false
      progress = 0.97
    } completion: {
      appData.showProfileView = false
      appData.animateProfile = false
    }
  }

  private var isEditing: Bool {
    editMode == .active
  }

  var body: some View {
    NavigationStack {
      Group {
        if supabase.vehicles.isEmpty {
          JoinVehicleView()
        } else {
          LazyVGrid(
            columns: Array(
              repeating: GridItem(.fixed(100), spacing: 25),
              count: 2
            )
          ) {
            ForEach(
              supabase.vehicles.map {
                V2Profile(
                  id: $0.id,
                  name: $0.name!,
                  icon: "benji",
                  vehicleId: $0.id,
                  vehicle: $0
                )
              }
            ) { profile in
              Button {
                Haptics.impact()

                if isEditing {
                  selectedEditProfile = profile
                } else {
                  lastSelectedVehicleId = profile.vehicleId
                  appData.watchingProfile = profile
                  appData.animateProfile = true
                }
              } label: {
                profileCard(profile)
              }
              .buttonStyle(.plain)
            }

            Button {
              Haptics.impact()
              showingJoinVehicleSheet.toggle()
            } label: {
              VStack(spacing: 8) {
                ZStack {
                  RoundedRectangle(cornerRadius: 10)
                    .stroke(
                      colorScheme == .dark
                        ? .white
                          .opacity(0.8)
                        : .black
                          .opacity(0.8),
                      lineWidth: 0.8
                    )

                  Image(systemName: "plus")
                    .font(.largeTitle)
                    .foregroundStyle(
                      colorScheme == .dark ? .white : .black
                    )
                }
                .frame(width: 100, height: 100)
                .contentShape(.rect)

                Text("Join")
                  .foregroundStyle(Color.primary)
                  .fontWeight(.semibold)
              }
            }
          }
          .padding(15)
          .frame(maxHeight: .infinity)
          .navigationTitle("Select Vehicle")
          .navigationBarTitleDisplayMode(.inline)
          .toolbar {
            if appData.fromTabBar {
              ToolbarItem(placement: .topBarLeading) {
                CloseButton {
                  withAnimation(
                    .snappy(duration: 0.1)
                  ) {
                    appData.showProfileView = false
                    appData.hideMainView = false
                    appData.fromTabBar = false
                  }
                }
              }
            }

            ToolbarItem(placement: .topBarTrailing) {
              EditButton()
            }
          }
          .environment(\.editMode, $editMode)
        }
      }
      .transition(.blurReplace)
    }
    .sheet(isPresented: $showingJoinVehicleSheet) {
      JoinVehicleView()
    }
    .sheet(item: $selectedEditProfile) { profile in
      NavigationStack {
        List {
          Section {
            LabeledContent("ID", value: profile.id)
          }

          Section {
            LabeledContent("Name", value: profile.vehicle.name ?? "Unnamed Vehicle")
            LabeledContent("Description", value: profile.vehicle.description ?? "None")
          }
        }
        .navigationTitle(profile.name)
        .toolbar {
          ToolbarItem(placement: .topBarLeading) {
            CloseButton {
              Haptics.impact()
              selectedEditProfile = nil
            }
          }
        }
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .opacity(animateToCenter ? 0 : 1)
    .background(colorScheme == .dark ? .black : .white)
    .opacity(animateToMainView ? 0 : 1)
    .overlayPreferenceValue(RectAnchorKey.self) { value in
      animationLayerView(value)
    }
  }

  @ViewBuilder
  private func animationLayerView(_ value: [String: Anchor<CGRect>]) -> some View {
    GeometryReader { proxy in
      if let profile = appData.watchingProfile, let sourceAnchor = value[profile.sourceAnchorID],
        appData.animateProfile
      {
        let sRect = proxy[sourceAnchor]
        let screenRect = proxy.frame(in: .global)

        let sourcePosition = CGPoint(x: sRect.midX, y: sRect.midY)
        let centerPosition = CGPoint(
          x: screenRect.width / 2,
          y: (screenRect.height / 2) - 40
        )
        let destinationPosition = CGPoint(
          x: appData.tabProfileRect.midX, y: appData.tabProfileRect.midY)

        let animationPath = Path { path in
          path.move(to: sourcePosition)
          path.addQuadCurve(
            to: destinationPosition,
            control: CGPoint(
              x: centerPosition.x * 2,
              y: centerPosition
                .y - (centerPosition.y / 0.8)))
        }

        //                animationPath.stroke(lineWidth: 2) // TODO: get rid of debug line eventually

        let endPosition =
          animationPath
          .trimmedPath(
            from: 0,
            to: 1
          ).currentPoint ?? destinationPosition
        let currentPosition =
          animationPath
          .trimmedPath(
            from: 0,
            to: 0.97
          ).currentPoint ?? destinationPosition

        let diff = CGSize(
          width: endPosition.x - currentPosition.x, height: endPosition.y - currentPosition.y)

        ZStack {
          Image(profile.icon)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(
              width: animateToMainView ? 25 : sRect.width,
              height: animateToMainView ? 25 : sRect.height
            )
            .clipShape(
              .rect(cornerRadius: animateToMainView ? 4 : 10)
            )
            .animation(
              .snappy(duration: 0.3, extraBounce: 0),
              value: animateToMainView
            )
            //            .opacity(
            //              animateToMainView && appData.activeTab != .account ? 0.6 : 1
            //            )
            .modifier(
              AnimatedPositionModifier(
                source: sourcePosition, center: centerPosition, destination: destinationPosition,
                animateToCenter: animateToCenter, animateToMainView: animateToMainView,
                path: animationPath, progress: progress)
            )
            .offset(animateToMainView ? diff : .zero)

          V2LoadingView()
            .frame(width: 60, height: 60)
            .offset(y: 80)
            .opacity(animateToCenter ? 1 : 0)
            .opacity(animateToMainView ? 0 : 1)
        }
        .transition(.identity)
        .task {
          guard !animateToCenter else { return }

          await animateCard()
        }
      }
    }
  }

  @ViewBuilder
  private func profileCard(_ profile: V2Profile) -> some View {
    VStack(spacing: 8) {
      let status = profile.id == appData.watchingProfile?.id

      GeometryReader { _ in
        Image(profile.icon)
          .resizable()
          .aspectRatio(contentMode: .fill)
          .frame(width: 100, height: 100)
          .clipShape(.rect(cornerRadius: 10))
          .opacity(isEditing ? 0.6 : animateToCenter ? 0 : 1)
          .overlay {
            if isEditing {
              Image(systemName: "pencil")
                .resizable()
                .scaledToFit()
                .frame(width: 50, height: 50)
            }
          }
      }
      .animation(
        status ? .none : .bouncy(duration: 0.35),
        value: animateToCenter
      )
      .frame(width: 100, height: 100)
      .anchorPreference(
        key: RectAnchorKey.self, value: .bounds,
        transform: { anchor in
          return [profile.sourceAnchorID: anchor]
        }
      )

      Text(profile.name)
        .fontWeight(.semibold)
        .lineLimit(1)
    }
  }
}

#Preview {
  V2ProfileSelectView()
    .environment(V2AppData())
    .preferredColorScheme(.dark)
}
