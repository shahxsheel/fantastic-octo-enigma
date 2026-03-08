//
//  DrivingScoreCalculatorTests.swift
//  InfineonProjectTests
//
//  Created by Aaron Ma on 2/20/26.
//

import Foundation
import Testing

@testable import InfineonProject

struct DrivingScoreCalculatorTests {

  // MARK: - Helpers

  /// Creates a Trip from a VehicleTrip with the given parameters.
  private func makeTrip(
    durationSeconds: TimeInterval = 3600,
    maxSpeedMph: Int = 60,
    avgSpeedMph: Double = 40,
    maxIntoxicationScore: Int = 0,
    speedingEventCount: Int = 0,
    drowsyEventCount: Int = 0,
    excessiveBlinkingEventCount: Int = 0,
    unstableEyesEventCount: Int = 0,
    faceDetectionCount: Int = 50,
    phoneDistractionEventCount: Int = 0,
    drinkingEventCount: Int = 0
  ) -> Trip {
    let start = Date.now.addingTimeInterval(-durationSeconds)
    let end = Date.now

    return Trip(
      vehicleTrip: VehicleTrip(
        id: UUID(),
        createdAt: .now,
        vehicleId: "test",
        sessionId: UUID(),
        driverProfileId: nil,
        startedAt: start,
        endedAt: end,
        status: "ok",
        maxSpeedMph: maxSpeedMph,
        avgSpeedMph: avgSpeedMph,
        maxIntoxicationScore: maxIntoxicationScore,
        speedingEventCount: speedingEventCount,
        drowsyEventCount: drowsyEventCount,
        excessiveBlinkingEventCount: excessiveBlinkingEventCount,
        unstableEyesEventCount: unstableEyesEventCount,
        faceDetectionCount: faceDetectionCount,
        speedSampleCount: 100,
        speedSampleSum: Int(avgSpeedMph * 100),
        phoneDistractionEventCount: phoneDistractionEventCount,
        drinkingEventCount: drinkingEventCount
      )
    )
  }

  // MARK: - Perfect Trip

  @Test func perfectTripScores100() {
    let trip = makeTrip()
    let score = DrivingScoreCalculator.score(for: trip)

    #expect(score.overall == 100)
    #expect(score.attentiveness == 100)
    #expect(score.safety == 100)
    #expect(score.impairment == 100)
    #expect(score.isConfident)
    #expect(score.isCameraAvailable)
  }

  // MARK: - Very Short Trip

  @Test func veryShortTripWithNoEventsScores100() {
    let trip = makeTrip(durationSeconds: 60)
    let score = DrivingScoreCalculator.score(for: trip)

    #expect(score.overall == 100)
  }

  @Test func veryShortTripWithCriticalEventStillPenalizes() {
    let trip = makeTrip(
      durationSeconds: 60,
      maxIntoxicationScore: 5
    )
    let score = DrivingScoreCalculator.score(for: trip)

    #expect(score.overall < 100)
    #expect(score.impairment < 50)
  }

  // MARK: - Attentiveness Penalties

  @Test func phoneDistractionReducesAttentiveness() {
    let trip = makeTrip(phoneDistractionEventCount: 4)  // 4 events in 1 hour = 4/hr
    let score = DrivingScoreCalculator.score(for: trip)

    // 4 * 15 = 60, capped at 50 -> attentiveness = 50
    #expect(score.attentiveness == 50)
    #expect(score.safety == 100)
    #expect(score.impairment == 100)
  }

  @Test func drowsinessReducesAttentiveness() {
    let trip = makeTrip(drowsyEventCount: 3)  // 3 events in 1 hour
    let score = DrivingScoreCalculator.score(for: trip)

    // 3 * 8 = 24 -> attentiveness = 76
    #expect(score.attentiveness == 76)
  }

  @Test func excessiveBlinkingReducesAttentiveness() {
    let trip = makeTrip(excessiveBlinkingEventCount: 5)  // 5 events in 1 hour
    let score = DrivingScoreCalculator.score(for: trip)

    // 5 * 3 = 15, capped at 15 -> attentiveness = 85
    #expect(score.attentiveness == 85)
  }

  // MARK: - Safety Penalties

  @Test func speedingReducesSafety() {
    let trip = makeTrip(speedingEventCount: 6)
    let score = DrivingScoreCalculator.score(for: trip)

    // 6/hr * 5 = 30 -> safety = 70
    #expect(score.safety == 70)
    #expect(score.attentiveness == 100)
  }

  @Test func highMaxSpeedReducesSafety() {
    let trip = makeTrip(maxSpeedMph: 95)
    let score = DrivingScoreCalculator.score(for: trip)

    // 91-100 tier = 20 penalty -> safety = 80
    #expect(score.safety == 80)
  }

  @Test func extremeMaxSpeedReducesSafetyMore() {
    let trip = makeTrip(maxSpeedMph: 110)
    let score = DrivingScoreCalculator.score(for: trip)

    // 101+ = 30 penalty -> safety = 70
    #expect(score.safety == 70)
  }

  @Test func highAvgSpeedReducesSafety() {
    let trip = makeTrip(avgSpeedMph: 85)
    let score = DrivingScoreCalculator.score(for: trip)

    // (85 - 75) * 0.5 = 5 penalty -> safety = 95
    #expect(score.safety == 95)
  }

  // MARK: - Impairment Penalties

  @Test func intoxicationScore4ReducesImpairment() {
    let trip = makeTrip(maxIntoxicationScore: 4)
    let score = DrivingScoreCalculator.score(for: trip)

    // Intoxication 4 = 50 penalty -> impairment = 50
    #expect(score.impairment == 50)
  }

  @Test func intoxicationScore6IsMaxPenalty() {
    let trip = makeTrip(maxIntoxicationScore: 6)
    let score = DrivingScoreCalculator.score(for: trip)

    // Intoxication 6 = 80 penalty -> impairment = 20
    #expect(score.impairment == 20)
  }

  @Test func drinkingEventsReduceImpairment() {
    let trip = makeTrip(drinkingEventCount: 2)  // 2/hr
    let score = DrivingScoreCalculator.score(for: trip)

    // 2 * 20 = 40 penalty -> impairment = 60
    #expect(score.impairment == 60)
  }

  // MARK: - No Camera Data

  @Test func noCameraDataUsesSafetyOnly() {
    let trip = makeTrip(
      speedingEventCount: 4,
      faceDetectionCount: 0
    )
    let score = DrivingScoreCalculator.score(for: trip)

    #expect(!score.isCameraAvailable)
    // Overall should equal the safety score when no camera
    #expect(score.overall == score.safety)
  }

  @Test func lowFaceDetectionsMarksNotConfident() {
    let trip = makeTrip(faceDetectionCount: 3)
    let score = DrivingScoreCalculator.score(for: trip)

    #expect(!score.isConfident)
    #expect(score.isCameraAvailable)
  }

  // MARK: - Duration Normalization

  @Test func shorterTripWithSameEventsScoresWorse() {
    // Same number of events but half the duration = double the rate
    let longTrip = makeTrip(durationSeconds: 3600, drowsyEventCount: 2)
    let shortTrip = makeTrip(durationSeconds: 1800, drowsyEventCount: 2)

    let longScore = DrivingScoreCalculator.score(for: longTrip)
    let shortScore = DrivingScoreCalculator.score(for: shortTrip)

    #expect(shortScore.attentiveness < longScore.attentiveness)
  }

  // MARK: - Daily Aggregation

  @Test func dailyScoreWithNoTripsIs100() {
    let daily = DrivingScoreCalculator.dailyScore(for: [])
    #expect(daily.overall == 100)
    #expect(daily.tripCount == 0)
  }

  @Test func dailyScoreWeightsByDuration() {
    // A long perfect trip should outweigh a short bad trip
    let goodTrip = makeTrip(durationSeconds: 3600)  // 1 hour, score 100
    let badTrip = makeTrip(
      durationSeconds: 600,  // 10 minutes
      maxIntoxicationScore: 6,
      speedingEventCount: 10,
      phoneDistractionEventCount: 5
    )

    let daily = DrivingScoreCalculator.dailyScore(for: [goodTrip, badTrip])

    // Good trip has 6x the weight, so daily score should be closer to 100 than to the bad trip's score
    #expect(daily.overall > 70)
    #expect(daily.tripCount == 2)
  }

  @Test func dailyScoreSingleTripEqualsTripScore() {
    let trip = makeTrip(speedingEventCount: 3)
    let tripScore = DrivingScoreCalculator.score(for: trip)
    let daily = DrivingScoreCalculator.dailyScore(for: [trip])

    #expect(daily.overall == tripScore.overall)
    #expect(daily.attentiveness == tripScore.attentiveness)
    #expect(daily.safety == tripScore.safety)
    #expect(daily.impairment == tripScore.impairment)
  }

  // MARK: - Sample Trip Validation

  @Test func sampleOkTripScores100() {
    let score = Trip.sample.score
    #expect(score.overall == 100)
  }

  @Test func sampleWarningTripScoresBelowPerfect() {
    let score = Trip.sampleWarning.score
    #expect(score.overall < 100)
    #expect(score.overall > 0)
    // Should have penalties in all dimensions
    #expect(score.attentiveness < 100)
    #expect(score.safety < 100)
    #expect(score.impairment < 100)
  }

  @Test func sampleDangerTripScoresVeryLow() {
    let score = Trip.sampleDanger.score
    #expect(score.overall < 30)
    #expect(score.attentiveness < 30)
    #expect(score.impairment < 30)
  }

  // MARK: - Score Category

  @Test func scoreCategoryThresholds() {
    #expect(DrivingScoreCalculator.scoreCategory(for: 100) == .good)
    #expect(DrivingScoreCalculator.scoreCategory(for: 80) == .good)
    #expect(DrivingScoreCalculator.scoreCategory(for: 79) == .moderate)
    #expect(DrivingScoreCalculator.scoreCategory(for: 50) == .moderate)
    #expect(DrivingScoreCalculator.scoreCategory(for: 49) == .poor)
    #expect(DrivingScoreCalculator.scoreCategory(for: 0) == .poor)
  }

  // MARK: - Rate Per Hour Utility

  @Test func ratePerHourCalculation() {
    // 10 events in 1 hour = 10/hr
    let rate = DrivingScoreCalculator.ratePerHour(count: 10, durationSeconds: 3600)
    #expect(abs(rate - 10.0) < 0.01)
  }

  @Test func ratePerHourClampsMinimumDuration() {
    // Very short duration should not explode
    let rate = DrivingScoreCalculator.ratePerHour(count: 1, durationSeconds: 1)
    #expect(rate.isFinite)
    #expect(rate > 0)
  }
}
