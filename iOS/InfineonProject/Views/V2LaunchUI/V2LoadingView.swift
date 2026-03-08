//
//  V2LoadingView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/16/26.
//

import SwiftUI

struct V2LoadingView: View {
  @State private var isSpinning = false

  var body: some View {
    Circle()
      .stroke(
        .linearGradient(
          colors: [
            .accentColor, .accentColor, .accentColor, .accentColor,
            .accentColor
              .opacity(0.7),
            .accentColor
              .opacity(0.4),
            .accentColor
              .opacity(0.1), .clear,
          ], startPoint: .top, endPoint: .bottom), lineWidth: 6
      )
      .rotationEffect(.init(degrees: isSpinning ? 360 : 0))
      .onAppear {
        withAnimation(
          .linear(duration: 0.7).repeatForever(autoreverses: false)
        ) {
          isSpinning = true
        }
      }
  }
}

#Preview {
  V2LoadingView()
    .frame(width: 100, height: 100)
    .preferredColorScheme(.dark)
}
