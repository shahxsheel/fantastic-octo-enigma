//
//  VehicleLiveActivityAttributes.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/19/26.
//

import ActivityKit

struct VehicleLiveActivityAttributes: ActivityAttributes {
  public struct ContentState: Codable, Hashable {
    // Dynamic stateful properties about your activity go here!
    var speed: Int
    var riskScore: Int
    var driverStatus: String
  }

  // Fixed non-changing properties about your activity go here!
  var name: String
  var speedLimit: Int
}
