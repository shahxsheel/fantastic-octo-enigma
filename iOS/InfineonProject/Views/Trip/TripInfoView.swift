//
//  TripInfoView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/12/26.
//

import SwiftUI

struct TripInfoView: View {
  var trip: Trip
  var namespace: Namespace.ID

  private var tripScore: TripScore { trip.score }

  var body: some View {
    HStack(spacing: 14) {
      // Score ring
      TripScoreRing(score: tripScore.overall, size: 52)
        .stableMatchedTransition(id: trip.id, in: namespace)

      // Trip info
      VStack(alignment: .leading, spacing: 4) {
        // Top line: status + duration
        HStack(alignment: .firstTextBaseline) {
          Text(trip.tripStatus)
            .font(.system(.subheadline, design: .rounded))
            .bold()

          if trip.isOngoing {
            Text("LIVE")
              .font(.system(.caption2, design: .rounded))
              .bold()
              .padding(.horizontal, 5)
              .padding(.vertical, 1)
              .background(.red.gradient)
              .foregroundStyle(.white)
              .clipShape(.capsule)
          }

          Spacer()

          Text(trip.formattedDuration)
            .font(.system(.caption, design: .rounded))
            .foregroundStyle(.tertiary)
        }

        // Date
        Text(
          trip.timeStarted.formatted(
            .dateTime.weekday(.abbreviated).month(.abbreviated).day().hour().minute())
        )
        .font(.caption)
        .foregroundStyle(.secondary)

        // Sub-score bars
        HStack(spacing: 3) {
          MiniScoreBar(value: tripScore.attentiveness, color: .cyan, label: "ATT")
          MiniScoreBar(value: tripScore.safety, color: .blue, label: "SAF")
          MiniScoreBar(value: tripScore.impairment, color: .purple, label: "IMP")
        }
        .padding(.top, 2)

        // Event pills
        if hasEvents {
          HStack(spacing: 4) {
            if trip.speedingEventCount > 0 {
              EventPill(icon: "bolt.fill", count: trip.speedingEventCount, color: .orange)
            }
            if trip.phoneDistractionEventCount > 0 {
              EventPill(icon: "iphone.gen3", count: trip.phoneDistractionEventCount, color: .red)
            }
            if trip.drinkingEventCount > 0 {
              EventPill(icon: "cup.and.saucer.fill", count: trip.drinkingEventCount, color: .orange)
            }

            Spacer(minLength: 0)

            // Max speed
            if trip.maxSpeedMph > 0 {
              Text("\(trip.maxSpeedMph)")
                .font(.system(.caption2, design: .rounded))
                .bold()
                .foregroundStyle(trip.maxSpeedMph > 80 ? .red : .secondary)
                + Text(" mph")
                .font(.system(.caption2, design: .rounded))
                .foregroundStyle(.tertiary)
            }
          }
        }
      }
    }
    .padding(.vertical, 6)
  }

  private var hasEvents: Bool {
    trip.speedingEventCount > 0
      || trip.phoneDistractionEventCount > 0
      || trip.drinkingEventCount > 0
      || trip.maxSpeedMph > 0
  }
}

// MARK: - Trip Score Ring

struct TripScoreRing: View {
  let score: Int
  let size: CGFloat

  @State private var ringProgress = CGFloat.zero

  private var color: Color {
    switch DrivingScoreCalculator.scoreCategory(for: score) {
    case .good: .green
    case .moderate: .orange
    case .poor: .red
    }
  }

  var body: some View {
    ZStack {
      ProgressRing(
        size: size,
        lineWidth: max(3, size * 0.07),
        progress: $ringProgress,
        foregroundStyle: color
      )

      // Score text
      Text("\(score)")
        .font(.system(size: size * 0.32, weight: .bold, design: .rounded))
        .contentTransition(.numericText(value: Double(score)))
        .foregroundStyle(color)
    }
    .onAppear {
      ringProgress = CGFloat(score) / 100.0
    }
    .onChange(of: score) {
      ringProgress = CGFloat(score) / 100.0
    }
  }
}

// MARK: - Mini Score Bar

struct MiniScoreBar: View {
  let value: Int
  let color: Color
  let label: String

  private var effectiveColor: Color {
    value >= 80 ? color : (value >= 50 ? .orange : .red)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Gauge(value: Double(value), in: 0...100) {
        EmptyView()
      }
      .gaugeStyle(.linearCapacity)
      .tint(effectiveColor.gradient)
      .frame(height: 3)

      Text(label)
        .font(.system(size: 8, weight: .medium, design: .rounded))
        .foregroundStyle(.quaternary)
    }
    .frame(maxWidth: .infinity)
  }
}

// MARK: - Event Pill

struct EventPill: View {
  let icon: String
  let count: Int
  let color: Color

  var body: some View {
    HStack(spacing: 2) {
      Image(systemName: icon)
        .font(.system(size: 8))
      Text("\(count)")
        .font(.system(size: 9, weight: .semibold, design: .rounded))
    }
    .foregroundStyle(color)
    .padding(.horizontal, 5)
    .padding(.vertical, 2)
    .background(color.opacity(0.12))
    .clipShape(.capsule)
  }
}

#Preview {
  @Previewable @Namespace var namespace

  List {
    TripInfoView(trip: Trip.sample, namespace: namespace)
    TripInfoView(trip: Trip.sampleWarning, namespace: namespace)
    TripInfoView(trip: Trip.sampleDanger, namespace: namespace)
  }
}
