//
//  TripDetailView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/12/26.
//

import MapKit
import SwiftUI

struct TripDetailView: View {
  var trip: Trip
  var namespace: Namespace.ID
  var previewRouteCoordinates: [CLLocationCoordinate2D] = []

  @State private var mapCameraPosition: MapCameraPosition = .automatic

  private var routeCoordinates: [CLLocationCoordinate2D] {
    let route = trip.routeCoordinates
    return route.isEmpty ? previewRouteCoordinates : route
  }

  private var riskScoreColor: Color {
    if trip.maxIntoxicationScore >= 4 { return .red }
    if trip.maxIntoxicationScore >= 2 { return .orange }
    return .green
  }

  var body: some View {
    ZStack {
      LinearGradient(
        colors: [
          trip.tripColor,
          trip.tripColor.opacity(0.9),
          .clear,
          .clear,
          .clear,
          .clear,
          .clear,
        ],
        startPoint: .top,
        endPoint: .bottom
      )
      .ignoresSafeArea()

      List {
        // Hero header
        Section {
          VStack(spacing: 16) {
            TripScoreRing(score: trip.score.overall, size: 120)
              .stableMatchedTransition(id: trip.id, in: namespace)

            Text(trip.tripStatus)
              .font(.system(.title3, design: .rounded))
              .bold()
              .titleVisibilityAnchor()

            if trip.isOngoing {
              HStack(spacing: 4) {
                Circle()
                  .fill(.red)
                  .frame(width: 6, height: 6)
                Text("Trip in progress")
                  .font(.system(.caption, design: .rounded))
              }
              .foregroundStyle(.red)
            }

            if trip.crashDetected {
              HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                Text("Crash Detected")
                  .font(.system(.caption, design: .rounded, weight: .semibold))
                if let severity = trip.crashSeverity {
                  Text("(\(severity))")
                    .font(.system(.caption2, design: .rounded))
                }
              }
              .foregroundStyle(.white)
              .padding(.horizontal, 12)
              .padding(.vertical, 6)
              .background(.red, in: .capsule)
            }

            HStack(spacing: 8) {
              DetailPill(
                icon: "calendar",
                text: trip.timeStarted.formatted(
                  .dateTime.month(.abbreviated).day().hour().minute())
              )

              DetailPill(icon: "clock.fill", text: trip.formattedDuration)
            }

            TripScoreBreakdown(trip: trip)
          }
          .frame(maxWidth: .infinity)
          .padding(.vertical, 4)
        }
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)

        Section("Trip Statistics") {
          LabeledContent("Max Speed") {
            Text("\(trip.maxSpeedMph) mph")
              .foregroundStyle(trip.maxSpeedMph > 65 ? .red : .primary)
          }

          LabeledContent("Average Speed") {
            Text(trip.avgSpeedMph, format: .number.precision(.fractionLength(1)))
              + Text(" mph")
          }

          LabeledContent("Risk Score") {
            Text("\(trip.maxIntoxicationScore)/6")
              .foregroundStyle(riskScoreColor)
          }

          LabeledContent("Session ID") {
            Text(trip.sessionId.uuidString)
              .font(.caption)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
        }

        Section("Event Summary") {
          eventRow(
            title: "Speeding Events",
            value: trip.speedingEventCount,
            icon: "speedometer",
            color: .orange
          )
          eventRow(
            title: "Phone Distraction",
            value: trip.phoneDistractionEventCount,
            icon: "iphone.gen3",
            color: .red
          )
          eventRow(
            title: "Drinking Events",
            value: trip.drinkingEventCount,
            icon: "cup.and.saucer.fill",
            color: .orange
          )
        }

        if !routeCoordinates.isEmpty {
          Section("Route") {
            Map(position: $mapCameraPosition) {
              if let first = routeCoordinates.first {
                Annotation("Start", coordinate: first) {
                  Image(systemName: "play.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.green)
                    .background(.white, in: Circle())
                }
              }

              if let last = routeCoordinates.last {
                Annotation("End", coordinate: last) {
                  Image(systemName: "stop.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.red)
                    .background(.white, in: Circle())
                }
              }

              MapPolyline(coordinates: routeCoordinates)
                .stroke(.blue, lineWidth: 5)
            }
            .frame(height: 260)
            .clipShape(.rect(cornerRadius: 16))
            .onAppear {
              setCameraForRoute()
            }
          }
        }
      }
      .scrollContentBackground(.hidden)
    }
    .navigationTitle("Trip Details")
    .navigationBarTitleDisplayMode(.inline)
  }

  @ViewBuilder
  private func eventRow(title: String, value: Int, icon: String, color: Color) -> some View {
    HStack {
      Label(title, systemImage: icon)
        .foregroundStyle(value > 0 ? color : .secondary)
      Spacer()
      Text("\(value)")
        .font(.system(.body, design: .rounded, weight: .semibold))
        .foregroundStyle(value > 0 ? color : .secondary)
    }
  }

  private func setCameraForRoute() {
    guard !routeCoordinates.isEmpty else { return }

    var rect = MKMapRect.null
    for coord in routeCoordinates {
      let point = MKMapPoint(coord)
      rect = rect.union(MKMapRect(origin: point, size: MKMapSize(width: 0, height: 0)))
    }

    if !rect.isNull {
      mapCameraPosition = .rect(rect.insetBy(dx: -rect.size.width * 0.2, dy: -rect.size.height * 0.2))
    }
  }
}

private struct DetailPill: View {
  let icon: String
  let text: String

  var body: some View {
    HStack(spacing: 6) {
      Image(systemName: icon)
      Text(text)
    }
    .font(.system(.caption, design: .rounded))
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
    .background(.ultraThinMaterial, in: Capsule())
  }
}

private struct TripScoreBreakdown: View {
  let trip: Trip

  private var score: TripScore { trip.score }

  var body: some View {
    VStack(spacing: 10) {
      scoreRow(title: "Attentiveness", score: score.attentiveness, color: .cyan)
      scoreRow(title: "Safety", score: score.safety, color: .blue)
      scoreRow(title: "Impairment", score: score.impairment, color: .purple)
    }
    .frame(maxWidth: .infinity)
  }

  @ViewBuilder
  private func scoreRow(title: String, score: Int, color: Color) -> some View {
    HStack(spacing: 10) {
      Text(title)
        .font(.caption)
        .foregroundStyle(.secondary)
        .frame(width: 96, alignment: .leading)

      GeometryReader { geo in
        let width = geo.size.width
        let progress = max(0, min(1, Double(score) / 100.0))

        ZStack(alignment: .leading) {
          Capsule()
            .fill(.white.opacity(0.18))
          Capsule()
            .fill(color.gradient)
            .frame(width: width * progress)
        }
      }
      .frame(height: 8)

      Text("\(score)")
        .font(.caption)
        .bold()
        .frame(width: 32, alignment: .trailing)
    }
  }
}

// Apple Park to Golden Gate Bridge sample route
private let sampleRouteAppleParkToGoldenGate: [CLLocationCoordinate2D] = [
  CLLocationCoordinate2D(latitude: 37.3349, longitude: -122.0090),  // Apple Park
  CLLocationCoordinate2D(latitude: 37.3400, longitude: -122.0300),
  CLLocationCoordinate2D(latitude: 37.3600, longitude: -122.0800),
  CLLocationCoordinate2D(latitude: 37.4000, longitude: -122.1200),
  CLLocationCoordinate2D(latitude: 37.4500, longitude: -122.1700),
  CLLocationCoordinate2D(latitude: 37.5000, longitude: -122.2100),
  CLLocationCoordinate2D(latitude: 37.5600, longitude: -122.2600),
  CLLocationCoordinate2D(latitude: 37.6200, longitude: -122.3100),
  CLLocationCoordinate2D(latitude: 37.6800, longitude: -122.3800),
  CLLocationCoordinate2D(latitude: 37.7400, longitude: -122.4400),
  CLLocationCoordinate2D(latitude: 37.7700, longitude: -122.4500),
  CLLocationCoordinate2D(latitude: 37.8199, longitude: -122.4783),  // Golden Gate Bridge
]

#Preview {
  @Previewable @Namespace var namespace

  NavigationStack {
    TripDetailView(
      trip: .sample,
      namespace: namespace,
      previewRouteCoordinates: sampleRouteAppleParkToGoldenGate
    )
  }
}
