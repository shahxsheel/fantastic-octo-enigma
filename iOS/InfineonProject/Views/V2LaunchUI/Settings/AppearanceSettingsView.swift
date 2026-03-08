//
//  AppearanceSettingsView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/19/26.
//

import SwiftUI

private enum AppTheme: String, CaseIterable {
  case system = "System"
  case light = "Light"
  case dark = "Dark"

  var icon: String {
    switch self {
    case .system:
      "circle.righthalf.filled"
    case .light:
      "sun.max.fill"
    case .dark:
      "moon.fill"
    }
  }

  var colorScheme: ColorScheme? {
    switch self {
    case .system:
      nil
    case .light:
      .light
    case .dark:
      .dark
    }
  }
}

struct ThemeSwitcher<Content: View>: View {
  @ViewBuilder var content: Content
  @AppStorage("_appTheme") private var appTheme = AppTheme.system

  var body: some View {
    content
      .preferredColorScheme(appTheme.colorScheme)
  }
}

struct AppearanceSettingsView: View {
  @AppStorage("_appTheme") private var appTheme = AppTheme.system

  var body: some View {
    List {
      Section {
        HStack {
          ForEach(AppTheme.allCases, id: \.self) {
            appearanceButton($0)
          }
        }
      }
      .listRowInsets(EdgeInsets())
      .listRowBackground(Color.clear)
      .listRowSeparator(.hidden)
      .listSectionSeparator(.hidden)

      Section {
        Picker(selection: .constant("English")) {
          Text("English")
            .tag("English")
        } label: {
          Label {
            Text("Language")
          } icon: {
            SettingsBoxView(
              icon: "globe", color: .pink
            )
          }
        }
      }
    }
    .navigationTitle("Appearance")
  }

  @ViewBuilder
  private func appearanceButton(_ theme: AppTheme) -> some View {
    Button {
      appTheme = theme
    } label: {
      VStack {
        RoundedRectangle(cornerRadius: 16)
          .fill(
            appTheme == theme
              ? AnyShapeStyle(.ultraThinMaterial)
              : AnyShapeStyle(
                .background
              )
          )
          .frame(height: 120)
          .frame(maxWidth: .infinity)
          .overlay {
            Image(systemName: theme.icon)
              .font(.title)
              .foregroundStyle(Color.primary)
          }

        Text(theme.rawValue)
          .foregroundStyle(Color.primary)
      }
    }
    .buttonStyle(.plain)
    .modifier(BackgroundShadowModifier())
  }
}

#Preview("Light Mode") {
  NavigationStack {
    AppearanceSettingsView()
      .preferredColorScheme(.light)
  }
}

#Preview("Dark Mode") {
  NavigationStack {
    AppearanceSettingsView()
      .preferredColorScheme(.dark)
  }
}
