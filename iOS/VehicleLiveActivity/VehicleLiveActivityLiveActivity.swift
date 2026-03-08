//
//  VehicleLiveActivityLiveActivity.swift
//  VehicleLiveActivity
//
//  Created by Aaron Ma on 1/19/26.
//

import ActivityKit
import SwiftUI
import WidgetKit

struct VehicleLiveActivityLiveActivity: Widget {
  private func intoxicationColor(for score: Int) -> Color {
    if score >= 4 { return .red }
    if score >= 2 { return .orange }
    return .green
  }

  var body: some WidgetConfiguration {
    ActivityConfiguration(
      for: VehicleLiveActivityAttributes.self
    ) { context in
      // Lock screen/banner UI goes here
      VStack {
        let isSpeeding = context.state.speed > context.attributes.speedLimit
        let isSpeedingColor = isSpeeding ? Color.red : .green

        Label {
          Text(isSpeeding ? "Speeding" : "OK")
        } icon: {
          Image(systemName: isSpeeding ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
            .foregroundStyle(isSpeedingColor)
        }
        .font(.title2)
        .bold()

        HStack {
          Text("\(context.state.speed)")
            .contentTransition(.numericText(value: 0))
            .bold()
            .font(.title)

          Text("mph")
        }

        Text("Riskiness: \(context.state.riskScore)/6, driver is: \(context.state.driverStatus)")
          .multilineTextAlignment(.center)
          .contentTransition(.numericText(value: 0))
          .foregroundStyle(
            intoxicationColor(
              for: context.state.riskScore
            )
          )
      }
      .padding()
      .activityBackgroundTint(.clear)
      .activitySystemActionForegroundColor(Color.black)
    } dynamicIsland: { context in
      let isSpeeding = context.state.speed > context.attributes.speedLimit
      let isSpeedingColor = isSpeeding ? Color.red : .green

      return DynamicIsland {
        // Expanded UI goes here.  Compose the expanded UI through
        // various regions, like leading/trailing/center/bottom
        DynamicIslandExpandedRegion(.leading) {
          VStack(alignment: .leading) {
            Image(systemName: "car.side.fill")

            Text(context.attributes.name)
              .bold()
          }
        }
        DynamicIslandExpandedRegion(.trailing) {
          Text("\(context.state.speed)")
            .contentTransition(.numericText(value: 0))
            .bold()
            .font(.largeTitle)

          Text("mph")
        }
        DynamicIslandExpandedRegion(.bottom) {
          Text("Riskiness: \(context.state.riskScore)/6, driver is: \(context.state.driverStatus)")
            .multilineTextAlignment(.center)
            .contentTransition(.numericText(value: 0))
            .foregroundStyle(
              intoxicationColor(
                for: context.state.riskScore
              )
            )
        }
      } compactLeading: {
        Image(systemName: "car.side.fill")
      } compactTrailing: {
        HStack(spacing: 1) {
          Text("\(context.state.speed)")
            .contentTransition(.numericText(value: 0))
            .foregroundStyle(isSpeedingColor)

          Text("mph")
        }
        .bold()
      } minimal: {
        Image(systemName: isSpeeding ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
          .foregroundStyle(isSpeedingColor)
      }
      .widgetURL(URL(string: "http://www.apple.com"))
      .keylineTint(isSpeedingColor)
    }
  }
}

extension VehicleLiveActivityAttributes {
  fileprivate static var preview: VehicleLiveActivityAttributes {
    VehicleLiveActivityAttributes(name: "World", speedLimit: 65)
  }
}

extension VehicleLiveActivityAttributes.ContentState {
  fileprivate static var smiley: VehicleLiveActivityAttributes.ContentState {
    VehicleLiveActivityAttributes.ContentState(speed: 65, riskScore: 2, driverStatus: "alert")
  }

  fileprivate static var starEyes: VehicleLiveActivityAttributes.ContentState {
    VehicleLiveActivityAttributes.ContentState(speed: 98, riskScore: 5, driverStatus: "drowsy")
  }
}

#Preview(
  "Notification",
  as: .content,
  using: VehicleLiveActivityAttributes.preview
) {
  VehicleLiveActivityLiveActivity()
} contentStates: {
  VehicleLiveActivityAttributes.ContentState.smiley
  VehicleLiveActivityAttributes.ContentState.starEyes
}
