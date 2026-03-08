//
//  VehicleSettingsView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 2/16/26.
//

import SwiftUI

struct VehicleSettingsView: View {
  @Environment(V2AppData.self) private var appData
  @Environment(\.dismiss) private var dismiss

  let vehicle: Vehicle

  @State private var vehicleName = ""
  @State private var vehicleDescription = ""
  @State private var isSaving = false
  @State private var errorMessage: String?

  // Feature toggles
  @State private var enableYolo = true
  @State private var enableMicrophone = true
  @State private var enableCamera = true
  @State private var enableDashcam = false

  private var canEdit: Bool { true }

  var body: some View {
    NavigationStack {
      List {
        Section {
          LabeledContent("Vehicle ID") {
            Text(vehicle.id)
              .font(.caption)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
        } header: {
          Text("Vehicle ID")
        } footer: {
          Text("This is your hardware's unique identifier.")
        }

        Section("Vehicle Name") {
          TextField("Vehicle name", text: $vehicleName)
            .disabled(!canEdit)
        }

        Section("Description") {
          TextField("Vehicle description", text: $vehicleDescription)
            .disabled(!canEdit)
        }

        if let errorMessage {
          Section {
            Text(errorMessage)
              .foregroundStyle(.red)
          }
        }

        Section("Note") {
          Text("These changes will take effect when the device restarts.")
        }

        Section("Hardware") {
          featureToggle(
            "Camera",
            icon: "camera.fill",
            slashIcon: "camera.slash.fill",
            description: "Capture frames on the device",
            color: .blue,
            isOn: $enableCamera
          )

          featureToggle(
            "Microphone",
            icon: "mic.fill",
            slashIcon: "mic.slash.fill",
            description: "Capture audio on the device",
            color: .orange,
            isOn: $enableMicrophone
          )
        }

        Section("Features") {
          featureToggle(
            "AI Detection",
            icon: "eye.fill",
            slashIcon: "eye.slash.fill",
            description: "Detect phone usage and drinking with YOLO",
            color: .purple,
            isOn: $enableYolo
          )

          featureToggle(
            "Dashcam",
            icon: "record.circle",
            slashIcon: "record.circle.fill",
            description: "Record annotated dashcam video on device",
            color: .red,
            isOn: $enableDashcam
          )
        }
      }
      .navigationTitle("Vehicle Settings")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          CloseButton {
            dismiss()
          }
        }

        if canEdit {
          ToolbarItem(placement: .topBarTrailing) {
            if isSaving {
              ProgressView()
            } else {
              Button("Save") {
                Task {
                  await saveVehicleSettings()
                }
              }
              .disabled(vehicleName.isEmpty)
            }
          }
        }
      }
      .onAppear {
        vehicleName = vehicle.name ?? ""
        vehicleDescription = vehicle.description ?? ""
        enableYolo = vehicle.enableYolo
        enableMicrophone = vehicle.enableMicrophone
        enableCamera = vehicle.enableCamera
        enableDashcam = vehicle.enableDashcam
      }
      .onDisappear {
        hideKeyboard()
      }
      .overlay {
        if isSaving {
          Color.black.opacity(0.3)
            .ignoresSafeArea()
            .overlay {
              ProgressView("Saving...")
                .padding()
                .background(.regularMaterial, in: .rect(cornerRadius: 12))
            }
        }
      }
    }
  }

  private func saveVehicleSettings() async {
    isSaving = true
    errorMessage = nil

    do {
      try await supabase.updateVehicle(
        vehicleId: vehicle.id,
        name: vehicleName,
        description: vehicleDescription.isEmpty ? nil : vehicleDescription,
        enableYolo: enableYolo,
        enableMicrophone: enableMicrophone,
        enableCamera: enableCamera,
        enableDashcam: enableDashcam
      )

      // Write settings to BLE immediately (if connected)
      if bluetooth.isConnected {
        bluetooth.writeSettings(
          BLESettingsData(
            from: Vehicle(
              id: vehicle.id,
              createdAt: vehicle.createdAt,
              updatedAt: .now,
              name: vehicleName,
              description: vehicleDescription,
              ownerId: vehicle.ownerId,
              enableYolo: enableYolo,
              enableMicrophone: enableMicrophone,
              enableCamera: enableCamera,
              enableDashcam: enableDashcam
            )))
      }

      // Update the profile in appData so the UI reflects the change
      await MainActor.run {
        if var profile = appData.watchingProfile {
          let updatedVehicle = supabase.vehicles.first { $0.id == vehicle.id }
          if let updatedVehicle {
            profile.vehicle = updatedVehicle
            profile.name = updatedVehicle.name ?? profile.name
            appData.watchingProfile = profile
          }
        }
        isSaving = false
        dismiss()
      }
    } catch {
      await MainActor.run {
        errorMessage = error.localizedDescription
        isSaving = false
      }
    }
  }

  @ViewBuilder
  private func featureToggle(
    _ title: String,
    icon: String,
    slashIcon: String,
    description: String,
    color: Color,
    isOn: Binding<Bool>
  ) -> some View {
    Toggle(isOn: isOn.animation()) {
      Label {
        VStack(alignment: .leading) {
          Text(title)
          Text(description)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      } icon: {
        SettingsBoxView(icon: isOn.wrappedValue ? icon : slashIcon, color: color)
      }
    }
    .disabled(!canEdit)
  }
}

#Preview {
  VehicleSettingsView(
    vehicle: Vehicle(
      id: "",
      createdAt: .now,
      updatedAt: .now,
      name: "",
      description: "",
      ownerId: UUID()))
}
