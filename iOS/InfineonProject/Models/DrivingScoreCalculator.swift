//
//  DrivingScoreCalculator.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/20/26.
//

import Foundation

/// The result of scoring a single trip across three dimensions.
struct TripScore {
  /// Overall driving score (0-100), weighted combination of sub-scores.
  let overall: Int
  /// Measures driver focus from distraction detections (0-100).
  let attentiveness: Int
  /// Measures driving behavior: speeding, max speed, avg speed (0-100).
  let safety: Int
  /// Measures impairment indicators: intoxication score, drinking (0-100).
  let impairment: Int
}

/// Aggregated daily score across multiple trips.
struct DailyScore {
  let overall: Int
  let attentiveness: Int
  let safety: Int
  let impairment: Int
  let tripCount: Int
  let totalDurationSeconds: TimeInterval
}

/// Pure scoring engine with no UI dependencies.
enum DrivingScoreCalculator {

  // MARK: - Attentiveness Penalties (per event per hour)

  static let phonePenaltyPerHour: Double = 15
  static let phoneMaxPenalty: Double = 60

  // MARK: - Safety Penalties

  static let speedingPenaltyPerHour: Double = 5
  static let speedingMaxPenalty: Double = 40

  static let avgSpeedThreshold: Double = 75
  static let avgSpeedPenaltyPerMph: Double = 0.5
  static let avgSpeedMaxPenalty: Double = 15

  // Max speed tier penalties: (upper bound, penalty)
  static let maxSpeedTiers: [(upperBound: Int, penalty: Double)] = [
    (80, 0),
    (90, 10),
    (100, 20),
  ]
  static let maxSpeedExtremePenalty: Double = 30

  // MARK: - Impairment Penalties

  /// Exponential-curve penalties indexed by intoxication score (0-6).
  static let intoxicationPenalties: [Double] = [0, 5, 15, 30, 50, 65, 80]

  static let drinkingPenaltyPerHour: Double = 20
  static let drinkingMaxPenalty: Double = 40

  // MARK: - Overall Weights

  static let attentivenessWeight: Double = 0.30
  static let safetyWeight: Double = 0.25
  static let impairmentWeight: Double = 0.45

  // MARK: - Thresholds

  /// Trips shorter than this are considered too short to score meaningfully.
  static let minimumTripDurationSeconds: TimeInterval = 120

  // MARK: - Per-Trip Scoring

  static func score(for trip: Trip) -> TripScore {
    let durationSeconds = trip.duration
    let durationHours = durationSeconds / 3600.0

    let hasCriticalEvent =
      trip.maxIntoxicationScore >= 4
      || trip.phoneDistractionEventCount > 0
      || trip.drinkingEventCount > 0

    // Very short trips get a perfect score unless something critical happened
    if durationSeconds < minimumTripDurationSeconds && !hasCriticalEvent {
      return TripScore(
        overall: 100,
        attentiveness: 100,
        safety: 100,
        impairment: 100
      )
    }

    // Attentiveness
    let attentiveness = calculateAttentiveness(
      phoneCount: trip.phoneDistractionEventCount,
      durationHours: durationHours
    )

    // Safety
    let safety = calculateSafety(
      speedingCount: trip.speedingEventCount,
      maxSpeed: trip.maxSpeedMph,
      avgSpeed: trip.avgSpeedMph,
      durationHours: durationHours
    )

    // Impairment
    let impairment = calculateImpairment(
      intoxicationScore: trip.maxIntoxicationScore,
      drinkingCount: trip.drinkingEventCount,
      durationHours: durationHours
    )

    let overall = Int(
      (Double(attentiveness) * attentivenessWeight)
        + (Double(safety) * safetyWeight)
        + (Double(impairment) * impairmentWeight)
    )

    return TripScore(
      overall: overall,
      attentiveness: attentiveness,
      safety: safety,
      impairment: impairment
    )
  }

  // MARK: - Daily Aggregation

  /// Duration-weighted aggregation of trip scores. Longer trips count more.
  static func dailyScore(for trips: [Trip]) -> DailyScore {
    guard !trips.isEmpty else {
      return DailyScore(
        overall: 100,
        attentiveness: 100,
        safety: 100,
        impairment: 100,
        tripCount: 0,
        totalDurationSeconds: 0
      )
    }

    var weightedOverall: Double = 0
    var weightedAttentiveness: Double = 0
    var weightedSafety: Double = 0
    var weightedImpairment: Double = 0
    var totalDuration: TimeInterval = 0

    for trip in trips {
      let tripScore = score(for: trip)
      // Use at least 1 second of weight so zero-duration trips don't vanish
      let weight = max(trip.duration, 1)
      totalDuration += weight

      weightedOverall += Double(tripScore.overall) * weight
      weightedAttentiveness += Double(tripScore.attentiveness) * weight
      weightedSafety += Double(tripScore.safety) * weight
      weightedImpairment += Double(tripScore.impairment) * weight
    }

    return DailyScore(
      overall: Int(weightedOverall / totalDuration),
      attentiveness: Int(weightedAttentiveness / totalDuration),
      safety: Int(weightedSafety / totalDuration),
      impairment: Int(weightedImpairment / totalDuration),
      tripCount: trips.count,
      totalDurationSeconds: totalDuration
    )
  }

  // MARK: - Sub-Score Calculations

  private static func calculateAttentiveness(
    phoneCount: Int,
    durationHours: Double
  ) -> Int {
    let effectiveDuration = max(durationHours, 1.0 / 60.0)

    let phoneRate = Double(phoneCount) / effectiveDuration

    let penalty = min(phoneRate * phonePenaltyPerHour, phoneMaxPenalty)

    return max(0, 100 - Int(penalty))
  }

  private static func calculateSafety(
    speedingCount: Int,
    maxSpeed: Int,
    avgSpeed: Double,
    durationHours: Double
  ) -> Int {
    let effectiveDuration = max(durationHours, 1.0 / 60.0)

    let speedingRate = Double(speedingCount) / effectiveDuration
    let speedingPenalty = min(speedingRate * speedingPenaltyPerHour, speedingMaxPenalty)

    // Max speed tier penalty
    var maxSpeedPenalty = maxSpeedExtremePenalty
    for tier in maxSpeedTiers {
      if maxSpeed <= tier.upperBound {
        maxSpeedPenalty = tier.penalty
        break
      }
    }

    // High average speed penalty
    let avgSpeedPenalty = min(
      max(0, (avgSpeed - avgSpeedThreshold) * avgSpeedPenaltyPerMph), avgSpeedMaxPenalty)

    let penalty = speedingPenalty + maxSpeedPenalty + avgSpeedPenalty
    return max(0, 100 - Int(penalty))
  }

  private static func calculateImpairment(
    intoxicationScore: Int,
    drinkingCount: Int,
    durationHours: Double
  ) -> Int {
    let safeScore = min(max(intoxicationScore, 0), intoxicationPenalties.count - 1)
    let intoxPenalty = intoxicationPenalties[safeScore]

    let effectiveDuration = max(durationHours, 1.0 / 60.0)
    let drinkingRate = Double(drinkingCount) / effectiveDuration
    let drinkingPenalty = min(drinkingRate * drinkingPenaltyPerHour, drinkingMaxPenalty)

    return max(0, 100 - Int(intoxPenalty + drinkingPenalty))
  }

  // MARK: - Helpers

  enum ScoreCategory {
    case good
    case moderate
    case poor
  }

  static func scoreCategory(for score: Int) -> ScoreCategory {
    switch score {
    case 80...: return .good
    case 50..<80: return .moderate
    default: return .poor
    }
  }
}
