//
//  FilterToggleButtonView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/25/26.
//

import SwiftUI

struct FilterToggleButtonSheetView: View {
  @Environment(\.dismiss) private var dismiss

  @Binding var selectedFilters: [String]

  private let filterOptions = [Filters.evenFilter.rawValue, Filters.oddFilter.rawValue]

  var body: some View {
    NavigationStack {
      List(filterOptions, id: \.self) { filter in
        Button {
          Haptics.impact()

          withAnimation(.bouncy) {
            if selectedFilters.contains(filter) {
              selectedFilters.removeAll { $0 == filter }
            } else {
              selectedFilters.append(filter)
            }
          }
        } label: {
          HStack {
            Text(filter)
              .foregroundStyle(Color.primary)

            Spacer()

            Image(systemName: "checkmark")
              .resizable()
              .frame(width: 18, height: 18)
              .foregroundStyle(.tint)
              .opacity(selectedFilters.contains(filter) ? 1 : 0)
          }
          .padding(.horizontal)
        }
      }
      .navigationTitle("Filters")
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          CloseButton {
            dismiss()
          }
        }
      }
    }
  }
}

struct FilterToggleButtonView<FilterOptions: View>: View {
  @Binding var filterShown: Bool
  let filteredByText: String

  @ViewBuilder var filterOptions: FilterOptions

  @Namespace private var namespace

  @State private var showingFilterSheet = false

  var body: some View {
    Toggle(
      "Toggle Filter", systemImage: "line.3.horizontal.decrease", isOn: $filterShown.animation())

    if filterShown {
      Button {
        showingFilterSheet.toggle()
      } label: {
        VStack(alignment: .leading, spacing: 0) {
          Text("Filtered by")
            .font(.caption)
            .fontWeight(.semibold)

          Text("\(filteredByText) \(Image(systemName: "chevron.down"))")
            .foregroundStyle(.tint)
            .font(.caption2)
            .fontWeight(.medium)
        }
        .padding(.trailing, 4)
        .frame(maxWidth: 150)
        .stableMatchedTransition(id: "filterOptionsSheet", in: namespace)
      }
      .sheet(isPresented: $showingFilterSheet) {
        filterOptions
          .navigationTransition(.zoom(sourceID: "filterOptionsSheet", in: namespace))
      }
    }
  }
}

private enum Filters: String {
  case evenFilter = "Even"
  case oddFilter = "Odd"
}

#Preview {
  @Previewable @State var filterShown = false
  @Previewable @State var filteredByText = ""

  @Previewable @State var selectedFilters: [String] = [Filters.evenFilter.rawValue]

  @Previewable @State var numbers = Array(1...25)

  var filteredNumbers: [Int] {
    guard filterShown else {
      return numbers
    }

    let includeEvenNumbers = selectedFilters.contains(Filters.evenFilter.rawValue)
    let includeOddNumbers = selectedFilters.contains(Filters.oddFilter.rawValue)

    return numbers.filter { number in
      let isEven = number.isMultiple(of: 2)
      return isEven ? includeEvenNumbers : includeOddNumbers
    }
  }

  NavigationStack {
    List(filteredNumbers, id: \.self) {
      Text("\($0)")
    }
    .navigationTitle("Numbers List")
    .toolbar {
      ToolbarItemGroup(placement: .bottomBar) {
        FilterToggleButtonView(
          filterShown: $filterShown, filteredByText: selectedFilters.joined(separator: ", ")
        ) {
          FilterToggleButtonSheetView(selectedFilters: $selectedFilters)
        }
      }

      if #available(iOS 26.0, *) {
        ToolbarSpacer(placement: .bottomBar)
      }
    }
  }
}

#Preview("Filter Sheet") {
  FilterToggleButtonSheetView(
    selectedFilters: .constant([Filters.oddFilter.rawValue, Filters.evenFilter.rawValue]))
}
