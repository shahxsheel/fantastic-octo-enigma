//
//  HomeView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/12/26.
//

import SwiftUI

struct HomeView: View {
  @Environment(\.dismiss) private var dismiss

  var vehicle: V2Profile

  @State private var trips: [Trip] = []
  @State private var todayTrips: [Trip] = []
  @State private var isLoading = true
  @State private var errorMessage: String?

  @State private var progressOuter = CGFloat.zero
  @State private var progressMiddle = CGFloat.zero
  @State private var progressInner = CGFloat.zero

  @Namespace private var namespace

  // Computed properties for today's stats
  private var todayTripCount: Int {
    todayTrips.count
  }

  private var todayDailyScore: DailyScore {
    DrivingScoreCalculator.dailyScore(for: todayTrips)
  }

  var body: some View {
    NavigationStack {
      Group {
        if isLoading {
          ProgressView("Loading trips...")
        } else if let errorMessage {
          ContentUnavailableView {
            Label("Error", systemImage: "exclamationmark.triangle")
          } description: {
            Text(errorMessage)
          } actions: {
            Button("Retry") {
              Task {
                await loadTrips()
              }
            }
          }
        } else if trips.isEmpty {
          ContentUnavailableView {
            Label("No Trips", systemImage: "car.side")
          } description: {
            Text("No trips recorded yet for this vehicle.")
          }
        } else {
          tripsList
        }
      }
      .navigationTitle("Trips")
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          CloseButton {
            dismiss()
          }
        }
      }
      .navigationDestination(for: String.self) { destination in
        switch destination {
        case Constants.HomeRouteAnnouncer.trips.rawValue:
          allTripsView
        default:
          Text("Unknown destination: \(destination)")
        }
      }
      .navigationDestination(for: Trip.self) { trip in
        TripDetailView(trip: trip, namespace: namespace)
      }
      .refreshable {
        await loadTrips()
      }
      .task {
        await loadTrips()
      }
    }
  }

  private var tripsList: some View {
    List {
      // Today's summary section
      Section {
        VStack(spacing: 16) {
          HStack(alignment: .top) {
            // Score ring — hero element
            TripScoreRing(score: todayDailyScore.overall, size: 90)

            Spacer(minLength: 16)

            // Sub-score gauges
            VStack(spacing: 10) {
              DailySubScoreGauge(
                label: "Attentiveness",
                score: todayDailyScore.attentiveness,
                icon: "eye.fill",
                color: .cyan
              )
              DailySubScoreGauge(
                label: "Safety",
                score: todayDailyScore.safety,
                icon: "shield.fill",
                color: .blue
              )
              DailySubScoreGauge(
                label: "Impairment",
                score: todayDailyScore.impairment,
                icon: "brain.head.profile.fill",
                color: .purple
              )
            }
          }

          // Rings
          RingsView(
            size: 80,
            lineWidth: 12,
            progressOuter: $progressOuter,
            progressMiddle: $progressMiddle,
            progressInner: $progressInner
          )
          .frame(maxWidth: .infinity, alignment: .center)
        }
        .padding(.vertical, 4)
        .onAppear {
          updateRings()
        }
        .onChange(of: todayTrips.count) {
          updateRings()
        }
      } header: {
        HStack(alignment: .firstTextBaseline) {
          Text("Today")
            .foregroundStyle(Color.primary)
            .font(.system(.title2, design: .rounded))
            .bold()

          Spacer()

          if !todayTrips.isEmpty {
            Text(
              "\(todayTripCount) trip\(todayTripCount == 1 ? "" : "s")"
            )
            .font(.system(.subheadline, design: .rounded))
            .foregroundStyle(.secondary)
          }
        }
      }

      // Recent trips section
      Section {
        ForEach(trips.prefix(8)) { trip in
          NavigationLink(value: trip) {
            TripInfoView(trip: trip, namespace: namespace)
          }
        }
      } header: {
        NavigationLink(value: Constants.HomeRouteAnnouncer.trips.rawValue) {
          HStack(alignment: .firstTextBaseline) {
            Text("Recent Trips")
              .font(.system(.title2, design: .rounded))
              .bold()

            Image(systemName: "chevron.right")
              .font(.system(.caption, design: .rounded))
              .bold()
              .foregroundStyle(.tertiary)

            Spacer()
          }
        }
        .foregroundStyle(.primary)
      }
    }
  }

  private var allTripsView: some View {
    List {
      ForEach(trips) { trip in
        NavigationLink(value: trip) {
          TripInfoView(trip: trip, namespace: namespace)
        }
      }
    }
    .navigationTitle("All Trips")
  }

  private func updateRings() {
    let daily = todayDailyScore
    withAnimation(.easeInOut(duration: 0.5)) {
      // Outer ring: overall driving score
      progressOuter = CGFloat(daily.overall) / 100.0
      // Middle ring: attentiveness sub-score
      progressMiddle = CGFloat(daily.attentiveness) / 100.0
      // Inner ring: safety sub-score
      progressInner = CGFloat(daily.safety) / 100.0
    }
  }

  private func loadTrips() async {
    isLoading = true
    errorMessage = nil

    do {
      async let allTripsTask = supabase.fetchCombinedTrips(for: vehicle.vehicleId)
      async let todayTripsTask = supabase.fetchCombinedTripsForToday(for: vehicle.vehicleId)

      let (fetchedTrips, fetchedTodayTrips) = try await (allTripsTask, todayTripsTask)

      await MainActor.run {
        self.trips = fetchedTrips.map { Trip(vehicleTrip: $0) }
        self.todayTrips = fetchedTodayTrips.map { Trip(vehicleTrip: $0) }
        self.isLoading = false
        updateRings()
      }
    } catch {
      await MainActor.run {
        self.errorMessage = error.localizedDescription
        self.isLoading = false
      }
    }
  }
}

// MARK: - Daily Sub-Score Gauge

struct DailySubScoreGauge: View {
  let label: String
  let score: Int
  let icon: String
  let color: Color

  private var effectiveColor: Color {
    switch DrivingScoreCalculator.scoreCategory(for: score) {
    case .good: color
    case .moderate: .orange
    case .poor: .red
    }
  }

  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: icon)
        .font(.system(size: 10))
        .foregroundStyle(effectiveColor)
        .frame(width: 14)

      VStack(alignment: .leading, spacing: 2) {
        HStack {
          Text(label)
            .font(.system(.caption2, design: .rounded))
            .foregroundStyle(.secondary)

          Spacer(minLength: 0)

          Text("\(score)")
            .font(.system(.caption2, design: .rounded))
            .bold()
            .foregroundStyle(effectiveColor)
        }

        Gauge(value: Double(score), in: 0...100) {
          EmptyView()
        }
        .gaugeStyle(.linearCapacity)
        .tint(effectiveColor.gradient)
      }
    }
  }
}

// Make Trip conform to Hashable for NavigationLink
extension Trip: Hashable {
  static func == (lhs: Trip, rhs: Trip) -> Bool {
    lhs.id == rhs.id
  }

  func hash(into hasher: inout Hasher) {
    hasher.combine(id)
  }
}

#Preview {
  HomeView(
    vehicle: V2Profile(
      id: "test",
      name: "Test Vehicle",
      icon: "benji",
      vehicleId: "test",
      vehicle: Vehicle(
        id: "test",
        createdAt: .now,
        updatedAt: .now,
        name: "Test",
        description: nil,
        ownerId: nil
      )
    )
  )
}
