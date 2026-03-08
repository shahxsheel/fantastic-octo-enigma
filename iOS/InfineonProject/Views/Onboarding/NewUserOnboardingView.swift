//
//  NewUserOnboardingView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/19/26.
//

import SwiftUI

@available(iOS 26, *)
struct Item: Identifiable {
  var id: UUID
  var title: String
  var subtitle: String
  var image: UIImage?
  var zoomScale: CGFloat = 1
  var zoomAnchor: UnitPoint = .center
}

@available(iOS 26, *)
struct NewUserOnboardingView: View {
  var items: [Item]
  var tint = Color.blue
  var hideBezels = false
  var onComplete: () -> Void

  @State private var currentStep = 0
  @State private var screenShotSize = CGSize.zero

  private var animation: Animation {
    .interpolatingSpring(duration: 0.65, bounce: 0, initialVelocity: 0)
  }

  private var iphoneCornerRadius: CGFloat {
    if let imageSize = items.first?.image?.size {
      let ratio = screenShotSize.height / imageSize.height

      return ratio * 32
    }

    return 0
  }

  var body: some View {
    ZStack(alignment: .bottom) {
      screenShotView()
        .compositingGroup()
        .scaleEffect(items[currentStep].zoomScale, anchor: items[currentStep].zoomAnchor)
        .padding(.top, 35)
        .padding(.horizontal, 30)
        .padding(.bottom, 220)

      VStack(spacing: 10) {
        textView()

        indicatorView()

        continueButton()
      }
      .padding(.top, 20)
      .padding(.horizontal, 15)
      .frame(height: 210)
      .background {
        customScrollEdgeEffect(radius: 18)
      }

      backButton()
    }
    .preferredColorScheme(.dark)
  }

  @ViewBuilder
  private func screenShotView() -> some View {
    let shape = ConcentricRectangle(corners: .concentric, isUniform: true)

    GeometryReader {
      let size = $0.size

      Rectangle()
        .fill(.black)

      ScrollView(.horizontal) {
        HStack(spacing: 12) {
          ForEach(items.indices, id: \.self) { i in
            let item = items[i]

            Group {
              if let image = item.image {
                Image(uiImage: image)
                  .resizable()
                  .aspectRatio(contentMode: .fit)
                  .onGeometryChange(for: CGSize.self, of: { $0.size }) { newValue in
                    screenShotSize = newValue
                  }
                  .clipShape(shape)
              } else {
                Rectangle()
                  .fill(.black)
              }
            }
            .frame(width: size.width, height: size.height)
          }
        }
        .scrollTargetLayout()
      }
      .scrollDisabled(true)
      .scrollTargetBehavior(.viewAligned)
      .scrollIndicators(.hidden)
      .scrollPosition(id: .init(get: { return currentStep }, set: { _ in }))
    }
    .clipShape(shape)
    .overlay {
      if screenShotSize != .zero && !hideBezels {
        ZStack {
          shape
            .stroke(.white, lineWidth: 6)

          shape
            .stroke(.black, lineWidth: 4)

          shape
            .stroke(.black, lineWidth: 6)
            .padding(4)
        }
        .padding(-6)
      }
    }
    .frame(
      maxWidth: screenShotSize.width == 0 ? nil : screenShotSize.width,
      maxHeight: screenShotSize.height == 0 ? nil : screenShotSize.height
    )
    .containerShape(RoundedRectangle(cornerRadius: iphoneCornerRadius))
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }

  @ViewBuilder
  private func textView() -> some View {
    GeometryReader {
      let size = $0.size

      ScrollView(.horizontal) {
        HStack(spacing: 0) {
          ForEach(items.indices, id: \.self) { i in
            let item = items[i]
            let isActive = currentStep == i

            VStack(spacing: 6) {
              Text(item.title)
                .font(.title2)
                .fontWeight(.semibold)
                .lineLimit(1)
                .foregroundStyle(.white)

              Text(item.subtitle)
                .font(.callout)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.8))
            }
            .frame(width: size.width)
            .compositingGroup()
            .blur(radius: isActive ? 0 : 30)
            .opacity(isActive ? 1 : 0)
          }
        }
        .scrollTargetLayout()
      }
      .scrollIndicators(.hidden)
      .scrollDisabled(true)
      .scrollTargetBehavior(.paging)
      .scrollPosition(id: .init(get: { return currentStep }, set: { _ in }))
    }
  }

  @ViewBuilder
  private func customScrollEdgeEffect(radius: CGFloat) -> some View {
    let tint = Color.black.opacity(0.5)

    Rectangle()
      .fill(.clear)
      .glassEffect(.clear.tint(tint), in: .rect)
      .blur(radius: radius)
      .padding([.horizontal, .bottom], -radius * 2)
      .opacity(items[currentStep].zoomScale != 1 ? 1 : 0)
      .ignoresSafeArea()
  }

  @ViewBuilder
  private func indicatorView() -> some View {
    HStack(spacing: 6) {
      ForEach(items.indices, id: \.self) { i in
        let isActive = currentStep == i

        Capsule()
          .fill(.white)
          .frame(width: isActive ? 25 : 6, height: 6)
      }
    }
    .padding(.bottom, 5)
  }

  @ViewBuilder
  private func backButton() -> some View {
    Button {
      withAnimation(animation) {
        currentStep = max(currentStep - 1, 0)
      }
    } label: {
      Image(systemName: "chevron.left")
        .font(.title3)
        .frame(width: 20, height: 30)
    }
    .buttonStyle(.glass)
    .buttonBorderShape(.circle)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .padding(.leading, 15)
    .padding(.top, 5)
    .opacity(currentStep == 0 ? 0 : 1)
  }

  @ViewBuilder
  private func continueButton() -> some View {
    Button {
      if currentStep == items.count - 1 {
        onComplete()
      }

      withAnimation(animation) {
        currentStep = min(currentStep + 1, items.count - 1)
      }
    } label: {
      Text(currentStep == items.count - 1 ? "Get Started" : "Continue")
        .fontWeight(.medium)
        .contentTransition(.numericText())
        .padding(.vertical, 6)
    }
    .tint(tint)
    .buttonStyle(.glassProminent)
    .buttonSizing(.automatic)
    .padding(.horizontal, 30)
  }
}

#Preview {
  if #available(iOS 26, *) {
    NewUserOnboardingView(
      items: [
        Item(
          id: UUID(), title: "Welcome to InfineonProject",
          subtitle:
            "Here, you'll find your vehicle's live status, including driver alertness metrics.",
          image: UIImage(imageLiteralResourceName: "sampleAppScreenshot")),
        Item(
          id: UUID(), title: "Live Location", subtitle: "See where your vehicle is at.",
          image: UIImage(imageLiteralResourceName: "sampleAppScreenshot"), zoomScale: 1.3,
          zoomAnchor: .bottom),
        Item(
          id: UUID(), title: "Live Telemetry", subtitle: "",
          image: UIImage(imageLiteralResourceName: "sampleAppScreenshot"), zoomScale: 1.2,
          zoomAnchor: .init(x: 0.5, y: -0.1)),
        Item(
          id: UUID(), title: "Test1", subtitle: "test1",
          image: UIImage(imageLiteralResourceName: "sampleAppScreenshot")),
      ], tint: .blue, hideBezels: false
    ) {}
  } else {
    Text("Update Xcode SDK to iOS 26 or later")
  }
}
