//
//  VehicleView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/12/26.
//

import ActivityKit
import Combine
import CoreLocation
import MapKit
import PostgREST
import Supabase
import SwiftUI

struct VehicleView: View {
  var vehicle: V2Profile

  @Namespace private var namespace

  @State private var showingAlertSheet = false
  @State private var showingTripsSheet = false
  @State private var showingLiveLocationSheet = false
  @State private var showingVehicleSettingsSheet = false
  @State private var showingBluetoothSheet = false
  @State private var showingAccountSheet = false

  @State var currentLiveActivity: Activity<VehicleLiveActivityAttributes>?

  // Location preview data
  @StateObject private var previewLocationManager = UserLocationManager()
  @State private var vehicleStreetName: String?
  @State private var vehicleTravelTime: String?
  @State private var cachedRoute: MKRoute?
  @State private var cachedVehicleCoordinate: CLLocationCoordinate2D?
  @State private var cachedUserCoordinate: CLLocationCoordinate2D?

  // Buzzer preview data
  @State private var cachedBuzzerActive: Bool?
  @State private var cachedBuzzerType: String?

  // Scroll tracking
  @State private var scrollOffset: CGFloat = 0

  private var scrollBlurAmount: CGFloat {
    max(0, min(scrollOffset / 30.0, 10))
  }

  private var scrollDimAmount: Double {
    max(0, min(Double(scrollOffset) / 200.0, 0.5))
  }

  var body: some View {
    NavigationStack {
      ZStack(alignment: .top) {
        VehicleAnimationView(
          isParked: !(vehicle.realtimeData?.isMoving ?? false),
          speed: vehicle.realtimeData?.speedMph ?? 0
        )
        .frame(height: 350)
        .blur(radius: scrollBlurAmount)
        .opacity(1.0 - scrollDimAmount)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .overlay(alignment: .topLeading) {
          VStack(alignment: .leading) {
            Text(vehicle.name)
              .font(.largeTitle)
              .bold()

            if let data = vehicle.realtimeData {
              Text(data.updatedAt, style: .relative)
                .foregroundStyle(.secondary)

              Text(data.isMoving ? "\(data.speedMph) MPH" : "Parked")
                .contentTransition(.numericText(value: 0))
                .foregroundStyle(.secondary)
            }
          }
          .blur(radius: scrollBlurAmount)
          .opacity(1.0 - scrollDimAmount)
          .padding(.horizontal)
        }

        ScrollView {
          VStack(spacing: 0) {
            Color.clear.frame(height: 290)

            VStack(alignment: .leading, spacing: 25) {
              // BLE connected banner
              if bluetooth.isConnected {
                Label {
                  VStack(alignment: .leading) {
                    Text("Bluetooth Connection")
                      .bold()
                    Text("Offline relay mode enabled")
                      .font(.caption)
                  }
                } icon: {
                  SettingsBoxView(icon: "antenna.radiowaves.left.and.right", color: .blue)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.blue.opacity(0.15), in: .rect(cornerRadius: 10))
              }

              // Driver alert
              if let data = vehicle.realtimeData {
                driverAlertSection(data: data)
              }

              Button {
                showingTripsSheet.toggle()
              } label: {
                Label {
                  VStack(alignment: .leading) {
                    Text("Trips")
                  }
                } icon: {
                  SettingsBoxView(
                    icon: "airplane.up.right",
                    color: .indigo
                  )
                }
              }
              .tint(.primary)
              .contentShape(.rect)
              .buttonStyle(
                FluidZoomTransitionStyle(
                  id: "tripsSheet", namespace: namespace, shape: .rect, applyGlass: false))

              // Live Data Section
              if let data = vehicle.realtimeData {
                Group {
                  Button {
                    showingAlertSheet.toggle()
                  } label: {
                    Label {
                      VStack(alignment: .leading) {
                        Text("Alert")
                      }
                    } icon: {
                      SettingsBoxView(
                        icon: "bell.fill",
                        color: .red
                      )
                    }
                  }
                  .tint(.primary)
                  .contentShape(.rect)
                  .buttonStyle(
                    FluidZoomTransitionStyle(
                      id: "alertSheet", namespace: namespace, shape: .rect, applyGlass: false)
                  )
                  .task {
                    await fetchBuzzerStatus()
                  }

                  Button {
                    showingLiveLocationSheet.toggle()
                  } label: {
                    Label {
                      VStack(alignment: .leading) {
                        // Primary: Street name, fallback: "Live Location"
                        if let streetName = vehicleStreetName {
                          Text(streetName)
                        } else if data.latitude != nil && data.longitude != nil {
                          Text("Live Location")
                        } else {
                          Text("Location Unavailable")
                        }

                        // Secondary: Travel time, fallback: coordinates, fallback: "No GPS data"
                        if let travelTime = vehicleTravelTime {
                          Text(travelTime)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        } else if let lat = data.latitude, let lon = data.longitude {
                          Text("\(lat, specifier: "%.4f"), \(lon, specifier: "%.4f")")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        } else {
                          Text("No GPS data")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                      }
                    } icon: {
                      SettingsBoxView(
                        icon: "location.fill",
                        color: .blue
                      )
                    }
                  }
                  .tint(.primary)
                  .contentShape(.rect)
                  .buttonStyle(
                    FluidZoomTransitionStyle(
                      id: "liveLocationSheet", namespace: namespace, shape: .rect, applyGlass: false
                    )
                  )
                  .task {
                    await fetchVehicleLocationPreview(data: data)
                  }
                  .onChange(of: previewLocationManager.userLocation) { _, newLocation in
                    if let userCoord = newLocation,
                      let lat = data.latitude,
                      let lon = data.longitude
                    {
                      Task {
                        await calculatePreviewTravelTime(
                          from: userCoord,
                          to: CLLocationCoordinate2D(latitude: lat, longitude: lon)
                        )
                      }
                    }
                  }

                  Button {
                    showingBluetoothSheet.toggle()
                  } label: {
                    Label {
                      HStack {
                        Text("Phone Connection")
                        Spacer()
                        if bluetooth.isConnected {
                          Text("Connected")
                            .font(.caption)
                            .foregroundStyle(.green)
                        } else if bluetooth.bleEnabled {
                          Text(bluetooth.statusMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                      }
                    } icon: {
                      SettingsBoxView(
                        icon: bluetooth.isConnected
                          ? "antenna.radiowaves.left.and.right"
                          : "antenna.radiowaves.left.and.right.slash",
                        color: bluetooth.isConnected ? .green : .gray
                      )
                    }
                  }
                  .tint(.primary)
                  .contentShape(.rect)
                  .buttonStyle(
                    FluidZoomTransitionStyle(
                      id: "bluetoothSheet", namespace: namespace, shape: .rect,
                      applyGlass: false))

                  Button {
                    showingVehicleSettingsSheet.toggle()
                  } label: {
                    Label {
                      Text("Vehicle Settings")
                    } icon: {
                      SettingsBoxView(icon: "car.fill", color: .blue)
                    }
                  }
                  .tint(.primary)
                  .contentShape(.rect)
                  .buttonStyle(
                    FluidZoomTransitionStyle(
                      id: "vehicleSettingsSheet", namespace: namespace, shape: .rect,
                      applyGlass: false))

                  LabeledContent("Speed") {
                    HStack {
                      Text(
                        "\(Text("\(data.speedMph)").font(.title2).foregroundStyle(.primary))/\(data.speedLimitMph)MPH"
                      )
                      .contentTransition(.numericText(value: 0))
                      .foregroundStyle(
                        data.isSpeeding ? .red : .secondary
                      )

                      if data.isSpeeding {
                        Image(
                          systemName: "exclamationmark.triangle.fill"
                        )
                        .foregroundStyle(.red)
                      }
                    }
                  }

                  LabeledContent("Heading") {
                    HStack {
                      Image(systemName: "location.north.fill")
                        .rotationEffect(
                          .degrees(
                            Double(data.headingDegrees)
                          )
                        )
                        .foregroundStyle(.blue)
                      Text(
                        "\(data.headingDegrees)° \(data.compassDirection)"
                      )
                    }
                  }

                  LabeledContent("Driver Status") {
                    DriverStatusBadge(status: data.driverStatus)
                  }

                  LabeledContent("Risk Score") {
                    Text("\(data.intoxicationScore)/6")
                      .foregroundStyle(
                        intoxicationColor(
                          for: data.intoxicationScore
                        )
                      )
                  }

                  // GPS Satellites
                  if let satellites = data.satellites {
                    LabeledContent("GPS Satellites") {
                      HStack {
                        Image(systemName: "location.fill")
                          .foregroundStyle(satellites > 0 ? .green : .gray)
                        Text("\(satellites)")
                          .foregroundStyle(satellites > 0 ? .primary : .secondary)
                      }
                    }
                  }

                  // Distraction indicators
                  if data.isPhoneDetected == true || data.isDrinkingDetected == true {
                    LabeledContent("Distraction") {
                      HStack(spacing: 8) {
                        if data.isPhoneDetected == true {
                          Label("Phone", systemImage: "iphone.gen3")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.red.opacity(0.2))
                            .foregroundStyle(.red)
                            .clipShape(.capsule)
                        }
                        if data.isDrinkingDetected == true {
                          Label("Drinking", systemImage: "cup.and.saucer.fill")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.orange.opacity(0.2))
                            .foregroundStyle(.orange)
                            .clipShape(.capsule)
                        }
                      }
                    }
                  }
                }
                .onAppear {
                  Task {
                    do {
                      currentLiveActivity = try Activity<VehicleLiveActivityAttributes>
                        .request(
                          attributes: VehicleLiveActivityAttributes(
                            name: vehicle.name,
                            speedLimit: 65
                          ),
                          content: .init(
                            state: .init(
                              speed: data.speedMph, riskScore: data.intoxicationScore,
                              driverStatus: data.driverStatus),
                            staleDate: .now
                              .addingTimeInterval(
                                60 * 60
                              ))
                        )
                    } catch {
                      print(error.localizedDescription)
                    }
                  }
                }
                .onChange(of: data.speedMph) {
                  Task {
                    if let currentLiveActivity {
                      await currentLiveActivity.update(
                        ActivityContent(
                          state: .init(
                            speed: data.speedMph, riskScore: data.intoxicationScore,
                            driverStatus: data.driverStatus),
                          staleDate: .now
                            .addingTimeInterval(60 * 60)))
                    }
                  }
                }
              }
            }
            .padding([.horizontal, .bottom])
            .frame(maxWidth: .infinity)
          }
        }
        .scrollIndicators(.hidden)
        .onScrollGeometryChange(for: CGFloat.self) { geo in
          geo.contentOffset.y
        } action: { _, newValue in
          scrollOffset = newValue
        }
      }
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) {
          Button {
            Haptics.impact()
            showingAccountSheet.toggle()
          } label: {
            ProfileToolbarImage()
          }
          .containerShape(.circle)
          .contentShape(.circle)
          .buttonStyle(
            FluidZoomTransitionStyle(id: "settingsSheet", namespace: namespace, shape: .circle))
        }
      }
    }
    .sheet(isPresented: $showingTripsSheet) {
      HomeView(vehicle: vehicle)
        .navigationTransition(.zoom(sourceID: "tripsSheet", in: namespace))
    }
    .sheet(isPresented: $showingLiveLocationSheet) {
      Group {
        if let data = vehicle.realtimeData {
          VehicleLiveLocationView(
            vehicleData: data,
            vehicleName: vehicle.name,
            cachedRoute: cachedRoute,
            cachedStreetName: vehicleStreetName,
            cachedUserLocation: cachedUserCoordinate
          )
        } else {
          Text("Unable to load location.")
        }
      }
      .navigationTransition(.zoom(sourceID: "liveLocationSheet", in: namespace))
    }
    .sheet(isPresented: $showingVehicleSettingsSheet) {
      VehicleSettingsView(vehicle: vehicle.vehicle)
        .navigationTransition(.zoom(sourceID: "vehicleSettingsSheet", in: namespace))
    }
    .sheet(isPresented: $showingBluetoothSheet) {
      BluetoothConnectionView()
        .navigationTransition(.zoom(sourceID: "bluetoothSheet", in: namespace))
    }
    .sheet(isPresented: $showingAlertSheet) {
      VehicleAlertControlView(
        vehicleId: vehicle.vehicle.id,
        vehicleName: vehicle.name,
        initialBuzzerActive: cachedBuzzerActive,
        initialBuzzerType: cachedBuzzerType
      )
      .navigationTransition(.zoom(sourceID: "alertSheet", in: namespace))
    }
    .sheet(isPresented: $showingAccountSheet) {
      V2AccountView()
        .navigationTransition(.zoom(sourceID: "settingsSheet", in: namespace))
    }
  }

  // MARK: - Driver Alert Section

  @ViewBuilder
  private func driverAlertSection(data: VehicleRealtime) -> some View {
    // Speeding alert
    if data.isSpeeding {
      Label {
        VStack(alignment: .leading) {
          Text("Speeding!")
            .bold()
          Text("\(data.speedMph) MPH in a \(data.speedLimitMph) MPH zone")
            .font(.caption)
        }
      } icon: {
        SettingsBoxView(icon: "speedometer", color: .red)
      }
      .padding()
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.red.opacity(0.15), in: .rect(cornerRadius: 10))
    }

    // Phone distraction alert (highest priority)
    if data.isPhoneDetected == true
      || data.driverStatus.lowercased() == "distracted_phone"
    {
      Label {
        VStack(alignment: .leading) {
          Text("Phone Detected!")
            .bold()
          Text("Driver is looking at phone - dangerous!")
            .font(.caption)
        }
      } icon: {
        SettingsBoxView(icon: "iphone.gen3.radiowaves.left.and.right", color: .red)
      }
      .padding()
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.red.opacity(0.15), in: .rect(cornerRadius: 10))
    }
    // Drinking alert
    else if data.isDrinkingDetected == true
      || data.driverStatus.lowercased() == "distracted_drinking"
    {
      Label {
        VStack(alignment: .leading) {
          Text("Drinking Detected")
            .bold()
          Text("Driver is drinking - stay focused")
            .font(.caption)
        }
      } icon: {
        SettingsBoxView(icon: "cup.and.saucer.fill", color: .orange)
      }
      .padding()
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.orange.opacity(0.1), in: .rect(cornerRadius: 10))
    }
    // Impaired alert
    else if data.intoxicationScore >= 4
      || data.driverStatus.lowercased() == "impaired"
    {
      Label {
        VStack(alignment: .leading) {
          Text("Driver May Be Impaired")
            .bold()
          Text("Intoxication score: \(data.intoxicationScore)/6")
            .font(.caption)
        }
      } icon: {
        SettingsBoxView(icon: "exclamationmark.triangle.fill", color: .red)
      }
      .padding()
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.red.opacity(0.1), in: .rect(cornerRadius: 10))
    }
  }

  // MARK: - Helper Methods

  private func intoxicationColor(for score: Int) -> Color {
    if score >= 4 { return .red }
    if score >= 2 { return .orange }
    return .green
  }

  private func fetchBuzzerStatus() async {
    // When BLE is connected, buzzer state is local — skip Supabase fetch
    guard !bluetooth.isConnected else { return }

    do {
      let response: [VehicleRealtime] = try await supabase.client
        .from("vehicle_realtime")
        .select()
        .eq("vehicle_id", value: vehicle.vehicle.id)
        .execute()
        .value

      if let data = response.first {
        cachedBuzzerActive = data.buzzerActive
        cachedBuzzerType = data.buzzerType
      }
    } catch {
      // Keep cached values as nil, will fetch fresh in VehicleAlertControlView
    }
  }

  private func fetchVehicleLocationPreview(data: VehicleRealtime) async {
    guard let lat = data.latitude, let lon = data.longitude else { return }

    // Reverse geocode to get street name
    let geocoder = CLGeocoder()
    let location = CLLocation(latitude: lat, longitude: lon)

    do {
      let placemarks = try await geocoder.reverseGeocodeLocation(location)
      if let placemark = placemarks.first {
        var components: [String] = []
        if let street = placemark.thoroughfare {
          components.append(street)
        }
        if let city = placemark.locality {
          components.append(city)
        }
        if !components.isEmpty {
          vehicleStreetName = components.joined(separator: ", ")
        }
      }
    } catch {
      // Keep vehicleStreetName as nil, will show fallback
    }

    // Request user location to calculate travel time
    previewLocationManager.requestLocation()
  }

  private func calculatePreviewTravelTime(
    from source: CLLocationCoordinate2D,
    to destination: CLLocationCoordinate2D
  ) async {
    // Check if we can use cached route (coordinates haven't changed significantly)
    if let cachedRoute = cachedRoute,
      let cachedUser = cachedUserCoordinate,
      let cachedVehicle = cachedVehicleCoordinate,
      isCoordinateNearby(source, cachedUser, thresholdMeters: 100),
      isCoordinateNearby(destination, cachedVehicle, thresholdMeters: 100)
    {
      // Use cached data, no need to recalculate
      let formatter = DateComponentsFormatter()
      formatter.allowedUnits = [.hour, .minute]
      formatter.unitsStyle = .short
      formatter.maximumUnitCount = 2

      if let formatted = formatter.string(from: cachedRoute.expectedTravelTime) {
        vehicleTravelTime = "\(formatted) away"
      }
      return
    }

    let request = MKDirections.Request()
    request.source = MKMapItem(placemark: MKPlacemark(coordinate: source))
    request.destination = MKMapItem(placemark: MKPlacemark(coordinate: destination))
    request.transportType = .automobile

    let directions = MKDirections(request: request)

    do {
      let response = try await directions.calculate()
      if let route = response.routes.first {
        // Cache the route and coordinates
        cachedRoute = route
        cachedUserCoordinate = source
        cachedVehicleCoordinate = destination

        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute]
        formatter.unitsStyle = .short
        formatter.maximumUnitCount = 2

        if let formatted = formatter.string(from: route.expectedTravelTime) {
          vehicleTravelTime = "\(formatted) away"
        }
      }
    } catch {
      // Keep vehicleTravelTime as nil, will show coordinates as fallback
    }
  }

  private func isCoordinateNearby(
    _ coord1: CLLocationCoordinate2D,
    _ coord2: CLLocationCoordinate2D,
    thresholdMeters: Double
  ) -> Bool {
    let location1 = CLLocation(latitude: coord1.latitude, longitude: coord1.longitude)
    let location2 = CLLocation(latitude: coord2.latitude, longitude: coord2.longitude)
    return location1.distance(from: location2) < thresholdMeters
  }
}

// MARK: - Driver Status Badge (reused from VehicleListView)

struct DriverStatusBadge: View {
  let status: String

  private var statusColor: Color {
    switch status.lowercased() {
    case "alert": return .green
    case "drowsy": return .orange
    case "impaired": return .red
    case "distracted_phone": return .red
    case "distracted_drinking": return .orange
    default: return .gray
    }
  }

  private var statusIcon: String {
    switch status.lowercased() {
    case "alert": return "checkmark.circle.fill"
    case "drowsy": return "moon.fill"
    case "impaired": return "exclamationmark.triangle.fill"
    case "distracted_phone": return "iphone.gen3"
    case "distracted_drinking": return "cup.and.saucer.fill"
    default: return "questionmark.circle.fill"
    }
  }

  private var displayName: String {
    switch status.lowercased() {
    case "distracted_phone": return "Phone"
    case "distracted_drinking": return "Drinking"
    default: return status.capitalized
    }
  }

  var body: some View {
    Label(displayName, systemImage: statusIcon)
      .font(.caption)
      .padding(.horizontal, 8)
      .padding(.vertical, 4)
      .background(statusColor.opacity(0.2))
      .foregroundStyle(statusColor)
      .clipShape(.capsule)
  }
}

// MARK: - Profile Toolbar Image

struct ProfileToolbarImage: View {
  var body: some View {
    if let avatarPath = supabase.userProfile?.avatarPath,
      let avatarURL = supabase.getUserAvatarURL(path: avatarPath)
    {
      AsyncImage(url: avatarURL) { phase in
        switch phase {
        case .success(let image):
          image
            .resizable()
            .scaledToFill()
            .frame(width: 30, height: 30)
            .clipShape(.circle)
        default:
          defaultImage
        }
      }
    } else {
      defaultImage
    }
  }

  private var defaultImage: some View {
    Circle()
      .fill(.gray.gradient)
      .frame(width: 30, height: 30)
      .overlay {
        Image(systemName: "person.fill")
          .font(.caption)
          .foregroundStyle(.white)
      }
  }
}

// MARK: - CLLocationCoordinate2D Equatable

extension CLLocationCoordinate2D: @retroactive Equatable {
  public static func == (lhs: CLLocationCoordinate2D, rhs: CLLocationCoordinate2D) -> Bool {
    lhs.latitude == rhs.latitude && lhs.longitude == rhs.longitude
  }
}

// MARK: - Location Manager

@MainActor
class UserLocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
  private let locationManager = CLLocationManager()

  @Published var userLocation: CLLocationCoordinate2D?
  @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined
  @Published var locationError: String?

  override init() {
    super.init()
    locationManager.delegate = self
    locationManager.desiredAccuracy = kCLLocationAccuracyBest
    authorizationStatus = locationManager.authorizationStatus
  }

  func requestLocation() {
    locationError = nil

    switch authorizationStatus {
    case .notDetermined:
      locationManager.requestWhenInUseAuthorization()
    case .authorizedWhenInUse, .authorizedAlways:
      locationManager.requestLocation()
    case .denied, .restricted:
      locationError = "Location access denied. Please enable in Settings."
    @unknown default:
      locationError = "Unknown authorization status"
    }
  }

  nonisolated func locationManager(
    _ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus
  ) {
    Task { @MainActor in
      authorizationStatus = status
      if status == .authorizedWhenInUse || status == .authorizedAlways {
        locationManager.requestLocation()
      }
    }
  }

  nonisolated func locationManager(
    _ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]
  ) {
    Task { @MainActor in
      if let location = locations.last {
        userLocation = location.coordinate
      }
    }
  }

  nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
    Task { @MainActor in
      locationError = error.localizedDescription
    }
  }
}

// MARK: - Vehicle Live Location View

struct VehicleLiveLocationView: View {
  @Environment(\.dismiss) private var dismiss

  let vehicleData: VehicleRealtime
  let vehicleName: String
  let cachedRoute: MKRoute?
  let cachedStreetName: String?
  let cachedUserLocation: CLLocationCoordinate2D?

  @StateObject private var locationManager = UserLocationManager()
  @State private var mapCameraPosition: MapCameraPosition = .automatic
  @State private var route: MKRoute?
  @State private var streetName: String = "Loading..."
  @State private var travelTime: String = "Locating you..."
  @State private var hasCalculatedRoute = false
  @State private var hasInitializedFromCache = false

  private var vehicleCoordinate: CLLocationCoordinate2D? {
    guard let lat = vehicleData.latitude, let lon = vehicleData.longitude else {
      return nil
    }
    return CLLocationCoordinate2D(latitude: lat, longitude: lon)
  }

  var body: some View {
    NavigationStack {
      Group {
        if let vehicleCoord = vehicleCoordinate {
          Map(position: $mapCameraPosition) {
            // Vehicle marker
            Annotation(vehicleName, coordinate: vehicleCoord) {
              ZStack {
                Circle()
                  .fill(.blue)
                  .frame(width: 44, height: 44)
                Image(systemName: "car.fill")
                  .font(.title2)
                  .foregroundStyle(.white)
              }
            }

            // User location
            UserAnnotation()

            // Route polyline
            if let route {
              MapPolyline(route.polyline)
                .stroke(.blue, lineWidth: 5)
            }
          }
          .mapControls {
            MapUserLocationButton()
            MapCompass()
            MapScaleView()
          }
          .mapStyle(.standard(elevation: .realistic))
          .safeAreaInset(edge: .bottom) {
            if #available(iOS 26, macOS 26, watchOS 26, tvOS 26, visionOS 26, *) {
              locationInfoCard
                .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 32))
                .padding()
            } else {
              locationInfoCard
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                .padding()
            }
          }
          .onAppear {
            initializeFromCache(vehicleCoord: vehicleCoord)
            if cachedRoute == nil {
              locationManager.requestLocation()
            }
          }
          .task {
            if cachedStreetName == nil {
              await reverseGeocode(coordinate: vehicleCoord)
            }
          }
          .onChange(of: locationManager.userLocation) { _, newLocation in
            if let userCoord = newLocation, let vehicleCoord = vehicleCoordinate,
              !hasCalculatedRoute
            {
              hasCalculatedRoute = true
              Task {
                await calculateRoute(from: userCoord, to: vehicleCoord)
              }
            }
          }
          .onChange(of: locationManager.locationError) { _, error in
            if let error {
              travelTime = error
              if let vehicleCoord = vehicleCoordinate {
                mapCameraPosition = .region(
                  MKCoordinateRegion(
                    center: vehicleCoord,
                    span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
                  ))
              }
            }
          }
        } else {
          ContentUnavailableView {
            Label("No Location Data", systemImage: "location.slash")
          } description: {
            Text("Vehicle GPS coordinates are not available.")
          }
        }
      }
      .navigationTitle("Vehicle Location")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          CloseButton {
            dismiss()
          }
        }
      }
    }
  }

  private var locationInfoCard: some View {
    VStack(spacing: 4) {
      Text(streetName)
        .font(.headline)
        .bold()

      HStack(spacing: 4) {
        Image(systemName: "clock")
          .font(.caption)
        Text(travelTime)
          .font(.subheadline)
      }
      .foregroundStyle(.secondary)

      if vehicleData.isMoving {
        HStack(spacing: 4) {
          Circle()
            .fill(.green)
            .frame(width: 8, height: 8)
          Text("Moving at \(vehicleData.speedMph) mph \(vehicleData.compassDirection)")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
    }
    .padding()
    .frame(maxWidth: .infinity)
  }

  private func initializeFromCache(vehicleCoord: CLLocationCoordinate2D) {
    guard !hasInitializedFromCache else { return }
    hasInitializedFromCache = true

    // Use cached street name if available
    if let cachedStreetName {
      streetName = cachedStreetName
    }

    // Use cached route if available
    if let cachedRoute {
      route = cachedRoute
      hasCalculatedRoute = true

      let formatter = DateComponentsFormatter()
      formatter.allowedUnits = [.hour, .minute]
      formatter.unitsStyle = .full
      formatter.maximumUnitCount = 2

      if let formatted = formatter.string(from: cachedRoute.expectedTravelTime) {
        travelTime = "\(formatted) away"
      }

      // Set map position to show the cached route
      let rect = cachedRoute.polyline.boundingMapRect
      mapCameraPosition = .rect(rect.insetBy(dx: -rect.width * 0.2, dy: -rect.height * 0.2))
    }
  }

  private func reverseGeocode(coordinate: CLLocationCoordinate2D) async {
    let geocoder = CLGeocoder()
    let location = CLLocation(latitude: coordinate.latitude, longitude: coordinate.longitude)

    do {
      let placemarks = try await geocoder.reverseGeocodeLocation(location)
      if let placemark = placemarks.first {
        var components: [String] = []
        if let street = placemark.thoroughfare {
          components.append(street)
        }
        if let city = placemark.locality {
          components.append(city)
        }
        streetName = components.isEmpty ? "Unknown Location" : components.joined(separator: ", ")
      }
    } catch {
      streetName = "Unknown Location"
    }
  }

  private func calculateRoute(
    from source: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D
  ) async {
    let request = MKDirections.Request()
    request.source = MKMapItem(placemark: MKPlacemark(coordinate: source))
    request.destination = MKMapItem(placemark: MKPlacemark(coordinate: destination))
    request.transportType = .automobile

    let directions = MKDirections(request: request)

    do {
      let response = try await directions.calculate()
      if let calculatedRoute = response.routes.first {
        route = calculatedRoute
        travelTime = formatTravelTime(calculatedRoute.expectedTravelTime)

        // Adjust map to show entire route
        let rect = calculatedRoute.polyline.boundingMapRect
        mapCameraPosition = .rect(rect.insetBy(dx: -rect.width * 0.2, dy: -rect.height * 0.2))
      }
    } catch {
      travelTime = "Route unavailable"
      mapCameraPosition = .region(
        MKCoordinateRegion(
          center: destination,
          span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
        ))
    }
  }

  private func formatTravelTime(_ seconds: TimeInterval) -> String {
    let formatter = DateComponentsFormatter()
    formatter.allowedUnits = [.hour, .minute]
    formatter.unitsStyle = .full
    formatter.maximumUnitCount = 2

    if let formatted = formatter.string(from: seconds) {
      return "\(formatted) away"
    }
    return "Calculating..."
  }
}

// MARK: - Vehicle Alert Control View

struct VehicleAlertControlView: View {
  @Environment(\.dismiss) private var dismiss

  let vehicleId: String
  let vehicleName: String
  let initialBuzzerActive: Bool?
  let initialBuzzerType: String?

  @State private var buzzerActive = false
  @State private var buzzerType: BuzzerType = .alert
  @State private var isLoading = false
  @State private var errorMessage: String?

  enum BuzzerType: String, CaseIterable {
    case alert = "alert"
    case emergency = "emergency"
    case warning = "warning"

    var icon: String {
      switch self {
      case .alert: return "bell.fill"
      case .emergency: return "exclamationmark.triangle.fill"
      case .warning: return "exclamationmark.circle.fill"
      }
    }

    var color: Color {
      switch self {
      case .alert: return .orange
      case .emergency: return .red
      case .warning: return .yellow
      }
    }

    var displayName: String {
      rawValue.capitalized
    }
  }

  var body: some View {
    NavigationStack {
      List {
        Section("Buzzer Type") {
          Picker("Type", selection: $buzzerType) {
            ForEach(BuzzerType.allCases, id: \.self) { type in
              Label(type.displayName, systemImage: type.icon)
                .tag(type)
            }
          }
          .pickerStyle(.segmented)
          .disabled(buzzerActive)
        }

        Section {
          Button {
            Haptics.impact()
            Task {
              await toggleBuzzer()
            }
          } label: {
            HStack {
              Spacer()
              if isLoading {
                ProgressView()
                  .tint(.white)
              } else {
                Image(systemName: buzzerActive ? "stop.fill" : "play.fill")
                Text(buzzerActive ? "Stop Buzzer" : "Start Buzzer")
                  .bold()
              }
              Spacer()
            }
            .padding()
            .foregroundStyle(.white)
          }
          .listRowBackground(buzzerActive ? Color.red : buzzerType.color)
          .disabled(isLoading)
        }

        if let errorMessage {
          Section {
            Label {
              Text(errorMessage)
                .foregroundStyle(.red)
            } icon: {
              Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            }
          }
        }

        Section {
          Label {
            VStack(alignment: .leading) {
              Text("Remote Control")
                .font(.headline)
              Text(
                "This controls the buzzer on \(vehicleName). The tone plays continuously until stopped."
              )
              .font(.caption)
              .foregroundStyle(.secondary)
            }
          } icon: {
            Image(systemName: "info.circle.fill")
              .foregroundStyle(.blue)
          }
        }
      }
      .navigationTitle("Alert Control")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          CloseButton {
            dismiss()
          }
        }
      }
      .task {
        if let initialActive = initialBuzzerActive,
          let initialType = initialBuzzerType,
          let type = BuzzerType(rawValue: initialType)
        {
          buzzerActive = initialActive
          buzzerType = type
        } else {
          await fetchBuzzerState()
        }
      }
    }
  }

  private func fetchBuzzerState() async {
    // When BLE is connected, buzzer state is managed locally — skip Supabase
    guard !bluetooth.isConnected else { return }

    do {
      let response: [VehicleRealtime] = try await supabase.client
        .from("vehicle_realtime")
        .select()
        .eq("vehicle_id", value: vehicleId)
        .execute()
        .value

      if let data = response.first {
        buzzerActive = data.buzzerActive ?? false
        if let typeString = data.buzzerType,
          let type = BuzzerType(rawValue: typeString)
        {
          buzzerType = type
        }
      }
    } catch {
      errorMessage = "Failed to fetch buzzer state: \(error.localizedDescription)"
    }
  }

  private func toggleBuzzer() async {
    isLoading = true
    errorMessage = nil

    // Use BLE direct command when connected
    if bluetooth.isConnected {
      if buzzerActive {
        bluetooth.writeBuzzerCommand(active: false)
        buzzerActive = false
        Haptics.notification(.success)
      } else {
        bluetooth.writeBuzzerCommand(active: true, type: buzzerType.rawValue)
        buzzerActive = true
        Haptics.notification(.warning)
      }
      isLoading = false
      return
    }

    // Fall back to Supabase RPC
    do {
      if buzzerActive {
        try await supabase.client.rpc(
          "deactivate_vehicle_buzzer",
          params: ["p_vehicle_id": vehicleId]
        ).execute()

        buzzerActive = false
        Haptics.notification(.success)
      } else {
        try await supabase.client.rpc(
          "activate_vehicle_buzzer",
          params: [
            "p_vehicle_id": vehicleId,
            "p_buzzer_type": buzzerType.rawValue,
          ]
        ).execute()

        buzzerActive = true
        Haptics.notification(.warning)
      }
    } catch {
      errorMessage =
        "Failed to \(buzzerActive ? "stop" : "start") buzzer: \(error.localizedDescription)"
      Haptics.notification(.error)
    }

    isLoading = false
  }
}

#Preview {
  VehicleView(
    vehicle: V2Profile(
      id: "111", name: "AA", icon: "benji", vehicleId: "111",
      vehicle: Vehicle(
        id: "", createdAt: .now, updatedAt: .now, name: "", description: "",
        ownerId: UUID())))
}

#Preview("Live Location") {
  NavigationStack {
    VehicleLiveLocationView(
      vehicleData: VehicleRealtime(
        vehicleId: "test",
        updatedAt: .now,
        latitude: 37.3349,
        longitude: -122.0090,
        speedMph: 45,
        speedLimitMph: 65,
        headingDegrees: 270,
        compassDirection: "W",
        isSpeeding: false,
        isMoving: true,
        driverStatus: "alert",
        intoxicationScore: 0,
        satellites: 12,
        isPhoneDetected: false,
        isDrinkingDetected: false
      ),
      vehicleName: "Test Vehicle",
      cachedRoute: nil,
      cachedStreetName: nil,
      cachedUserLocation: nil
    )
  }
}

#Preview("Alert Control") {
  NavigationStack {
    VehicleAlertControlView(
      vehicleId: "test-vehicle",
      vehicleName: "Test Vehicle",
      initialBuzzerActive: nil,
      initialBuzzerType: nil
    )
  }
}
