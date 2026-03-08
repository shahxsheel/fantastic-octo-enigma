//
//  JoinVehicleView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/13/26.
//

import Foundation
import SwiftUI

private struct NHTSAMake: Codable, Identifiable, Hashable {
  let id: Int
  let name: String

  enum CodingKeys: String, CodingKey {
    case id = "Make_ID"
    case name = "Make_Name"
  }
}

private struct NHTSAModel: Codable, Identifiable, Hashable {
  let id: Int
  let name: String

  enum CodingKeys: String, CodingKey {
    case id = "Model_ID"
    case name = "Model_Name"
  }
}

private struct NHTSAResponse<T: Decodable>: Decodable {
  let results: [T]

  enum CodingKeys: String, CodingKey {
    case results = "Results"
  }
}

struct JoinVehicleView: View {
  @Environment(\.dismiss) private var dismiss

  @State private var makes: [NHTSAMake] = []
  @State private var models: [NHTSAModel] = []
  @State private var selectedVehicleId: String?
  @State private var selectedMake: NHTSAMake?
  @State private var selectedModel: NHTSAModel?
  @State private var makeQuery = ""

  @State private var isLoadingMakes = false
  @State private var isLoadingModels = false
  @State private var isApplying = false
  @State private var errorMessage: String?

  private var filteredMakes: [NHTSAMake] {
    guard !makeQuery.isEmpty else { return makes }
    return makes.filter { $0.name.localizedCaseInsensitiveContains(makeQuery) }
  }

  private var canApply: Bool {
    selectedVehicleId != nil && selectedMake != nil && selectedModel != nil && !isApplying
  }

  var body: some View {
    NavigationStack {
      Form {
        if supabase.vehicles.isEmpty {
          Section {
            Label("No vehicles found", systemImage: "car.slash")
              .foregroundStyle(.secondary)
            Text("Connect a Pi vehicle once so it appears in the list, then rename it from the internet catalog.")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        } else {
          Section("Connected Vehicle") {
            Picker("Vehicle", selection: Binding(get: {
              selectedVehicleId ?? supabase.vehicles.first?.id ?? ""
            }, set: { selectedVehicleId = $0 })) {
              ForEach(supabase.vehicles) { vehicle in
                Text(vehicle.name ?? vehicle.id)
                  .tag(vehicle.id)
              }
            }
            .pickerStyle(.navigationLink)
          }

          Section("Find Make") {
            TextField("Search make", text: $makeQuery)
              .textInputAutocapitalization(.words)

            if isLoadingMakes {
              ProgressView("Loading NHTSA makes...")
            } else {
              Picker("Make", selection: $selectedMake) {
                Text("Select make")
                  .tag(Optional<NHTSAMake>.none)
                ForEach(filteredMakes, id: \.self) { make in
                  Text(make.name)
                    .tag(Optional(make))
                }
              }
              .pickerStyle(.navigationLink)
            }
          }

          Section("Choose Model") {
            if isLoadingModels {
              ProgressView("Loading models...")
            } else {
              Picker("Model", selection: $selectedModel) {
                Text("Select model")
                  .tag(Optional<NHTSAModel>.none)
                ForEach(models, id: \.self) { model in
                  Text(model.name)
                    .tag(Optional(model))
                }
              }
              .pickerStyle(.navigationLink)
              .disabled(selectedMake == nil || models.isEmpty)
            }
          }

          Section {
            Button {
              Haptics.impact()
              Task {
                await applySelection()
              }
            } label: {
              HStack {
                Spacer()
                if isApplying {
                  ProgressView()
                    .controlSize(.large)
                } else {
                  Text("Apply Vehicle Name")
                    .bold()
                }
                Spacer()
              }
              .padding(.vertical, 8)
              .padding(.horizontal, 15)
            }
            .foregroundStyle(.white)
            .possibleGlassEffect(Color.accentColor, in: Capsule())
            .buttonStyle(.borderedProminent)
            .disabled(!canApply)
            .listRowInsets(EdgeInsets())
          }
        }

        if let errorMessage {
          Section {
            Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
              .foregroundStyle(.red)
          }
        }
      }
      .navigationTitle("Join Vehicle")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        if !supabase.vehicles.isEmpty {
          ToolbarItem(placement: .cancellationAction) {
            CloseButton {
              Haptics.impact()
              dismiss()
            }
          }
        }

        ToolbarItem(placement: .topBarTrailing) {
          Button("Reload") {
            Task { await loadMakes() }
          }
          .disabled(isLoadingMakes)
        }
      }
      .task {
        await supabase.loadVehicles()
        if selectedVehicleId == nil {
          selectedVehicleId = supabase.vehicles.first?.id
        }
        await loadMakes()
      }
      .onChange(of: selectedMake) { _, newMake in
        selectedModel = nil
        models = []
        guard let newMake else { return }
        Task {
          await loadModels(for: newMake)
        }
      }
    }
  }

  private func loadMakes() async {
    isLoadingMakes = true
    errorMessage = nil
    defer { isLoadingMakes = false }

    do {
      guard let url = URL(string: "https://vpic.nhtsa.dot.gov/api/vehicles/getallmakes?format=json")
      else {
        throw URLError(.badURL)
      }

      let (data, response) = try await URLSession.shared.data(from: url)
      guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
        throw URLError(.badServerResponse)
      }

      let decoded = try JSONDecoder().decode(NHTSAResponse<NHTSAMake>.self, from: data)
      makes = decoded.results.sorted { $0.name < $1.name }

      if selectedMake == nil {
        selectedMake = makes.first
      }
    } catch {
      errorMessage = "Failed to load internet car makes. Check connection and try again."
      print("NHTSA makes error: \(error)")
    }
  }

  private func loadModels(for make: NHTSAMake) async {
    isLoadingModels = true
    errorMessage = nil
    defer { isLoadingModels = false }

    do {
      let escaped = make.name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
        ?? make.name
      guard
        let url = URL(
          string: "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMake/\(escaped)?format=json")
      else {
        throw URLError(.badURL)
      }

      let (data, response) = try await URLSession.shared.data(from: url)
      guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
        throw URLError(.badServerResponse)
      }

      let decoded = try JSONDecoder().decode(NHTSAResponse<NHTSAModel>.self, from: data)
      models = decoded.results.sorted { $0.name < $1.name }
    } catch {
      errorMessage = "Failed to load models for \(make.name)."
      print("NHTSA models error: \(error)")
    }
  }

  private func applySelection() async {
    guard
      let vehicleId = selectedVehicleId,
      let make = selectedMake,
      let model = selectedModel
    else {
      return
    }

    isApplying = true
    errorMessage = nil

    do {
      let newName = "\(make.name) \(model.name)"
      try await supabase.updateVehicle(
        vehicleId: vehicleId,
        name: newName
      )
      await supabase.loadVehicles()
      await MainActor.run {
        isApplying = false
        dismiss()
      }
    } catch {
      await MainActor.run {
        isApplying = false
        errorMessage = "Could not rename vehicle."
      }
      print("Apply selection error: \(error)")
    }
  }
}

#Preview {
  JoinVehicleView()
}
