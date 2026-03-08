//
//  VehicleAnimationView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/14/26.
//

import SwiftUI

struct VehicleAnimationView: View {
  var isParked: Bool
  var speed: Int
  var isReversed: Bool = false

  var body: some View {
    TimelineView(
      .animation(minimumInterval: nil, paused: speed <= 0)
    ) { context in
      AnimatedLaneContent(
        isParked: isParked,
        speed: speed,
        isReversed: isReversed,
        date: context.date
      )
    }
  }
}

private struct AnimatedLaneContent: View {
  var isParked: Bool
  var speed: Int
  var isReversed: Bool
  var date: Date

  // ~10° from horizontal to match the car's near-horizontal orientation
  private let angleFromVertical: CGFloat = 80

  private var phase: Double {
    Double(speed) * date.timeIntervalSinceReferenceDate * 14.0 * (isReversed ? -1.0 : 1.0)
  }

  var body: some View {
    ZStack {
      Canvas { context, size in
        if !isParked {
          drawLaneMarkers(in: context, size: size)
        }
      }

      Image(.modelY)
        .resizable()
        .scaledToFit()
        .padding()
    }
  }

  private func drawLaneMarkers(in context: GraphicsContext, size: CGSize) {
    let angleRad = angleFromVertical * .pi / 180

    // Direction along the road (lower-left to upper-right)
    let dirX = sin(angleRad)
    let dirY = -cos(angleRad)

    // Perpendicular to road direction
    let perpX = cos(angleRad)
    let perpY = sin(angleRad)

    let centerX = size.width / 2
    let centerY = size.height / 2
    let laneHalfWidth = size.width * 0.22
    let lineColor = Color.white.opacity(0.25)
    let lineWidth: CGFloat = 2
    let lineLength = hypot(size.width, size.height) * 1.5

    // Left solid line
    let leftCX = centerX + perpX * laneHalfWidth * -1.0
    let leftCY = centerY + perpY * laneHalfWidth * -1.0

    var leftPath = Path()
    leftPath.move(
      to: CGPoint(
        x: leftCX - dirX * lineLength / 2,
        y: leftCY - dirY * lineLength / 2
      ))
    leftPath.addLine(
      to: CGPoint(
        x: leftCX + dirX * lineLength / 2,
        y: leftCY + dirY * lineLength / 2
      ))
    context.stroke(leftPath, with: .color(lineColor), lineWidth: lineWidth)

    // Dashed center line (faded ends)
    let centerDash: CGFloat = 50
    let centerGap: CGFloat = 70
    let centerPeriod = centerDash + centerGap
    let centerPhaseOffset = phase.truncatingRemainder(
      dividingBy: Double(centerPeriod)
    )
    let centerCount = Int(lineLength / centerPeriod) + 2

    for i in 0..<centerCount {
      let t =
        -lineLength / 2 + CGFloat(
          i
        ) * centerPeriod + centerPhaseOffset
      let start = CGPoint(x: centerX + dirX * t, y: centerY + dirY * t)
      let end = CGPoint(
        x: centerX + dirX * (t + centerDash),
        y: centerY + dirY * (t + centerDash)
      )

      drawFadingDash(
        in: context, start: start, end: end, perpX: perpX, perpY: perpY, lineWidth: lineWidth,
        color: lineColor)
    }

    // Right dashed line (longer dashes, faded ends)
    let rightCX = centerX + perpX * laneHalfWidth
    let rightCY = centerY + perpY * laneHalfWidth
    let rightDash: CGFloat = 140
    let rightGap: CGFloat = 50
    let rightPeriod = rightDash + rightGap
    let rightPhaseOffset = phase.truncatingRemainder(
      dividingBy: Double(rightPeriod)
    )
    let rightCount = Int(lineLength / rightPeriod) + 2

    for i in 0..<rightCount {
      let t =
        -lineLength / 2 + CGFloat(
          i
        ) * rightPeriod + rightPhaseOffset
      let start = CGPoint(x: rightCX + dirX * t, y: rightCY + dirY * t)
      let end = CGPoint(
        x: rightCX + dirX * (t + rightDash),
        y: rightCY + dirY * (t + rightDash)
      )

      drawFadingDash(
        in: context, start: start, end: end, perpX: perpX, perpY: perpY, lineWidth: lineWidth,
        color: lineColor)
    }
  }
  private func drawFadingDash(
    in context: GraphicsContext, start: CGPoint, end: CGPoint, perpX: CGFloat, perpY: CGFloat,
    lineWidth: CGFloat, color: Color
  ) {
    let halfWidth = lineWidth / 2

    var rect = Path()
    rect
      .move(
        to: CGPoint(
          x: start.x - perpX * halfWidth,
          y: start.y - perpY * halfWidth
        )
      )
    rect
      .addLine(
        to: CGPoint(
          x: start.x + perpX * halfWidth,
          y: start.y + perpY * halfWidth
        )
      )
    rect
      .addLine(
        to: CGPoint(
          x: end.x + perpX * halfWidth,
          y: end.y + perpY * halfWidth
        )
      )
    rect
      .addLine(
        to: CGPoint(
          x: end.x - perpX * halfWidth,
          y: end.y - perpY * halfWidth
        )
      )
    rect.closeSubpath()

    context.fill(
      rect,
      with: .linearGradient(
        Gradient(stops: [
          .init(color: color.opacity(0), location: 0),
          .init(color: color, location: 0.15),
          .init(color: color, location: 0.85),
          .init(color: color.opacity(0), location: 1.0),
        ]),
        startPoint: start,
        endPoint: end
      ))
  }
}

#Preview {
  VehicleAnimationView(isParked: false, speed: 90)
    .background(.black)
}

#Preview("Reversed") {
  VehicleAnimationView(isParked: false, speed: 90, isReversed: true)
    .background(.black)
}
