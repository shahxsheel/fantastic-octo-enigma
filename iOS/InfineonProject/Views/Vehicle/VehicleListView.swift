//
//  VehicleListView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/13/26.
//

import SwiftUI

struct VehicleListView: View {
  @State private var showingJoinVehicle = false
  @State private var selectedVehicle: Vehicle?

  var body: some View {
    NavigationStack {
      Group {
        if supabase.vehicles.isEmpty {
          ContentUnavailableView {
            Label("No Vehicles", systemImage: "car.fill")
          } description: {
            Text("Add a vehicle and assign a make/model from the internet catalog.")
          } actions: {
            Button("Add Vehicle") {
              showingJoinVehicle = true
            }
            .buttonStyle(.borderedProminent)
          }
        } else {
          List {
            ForEach(supabase.vehicles) { vehicle in
              VehicleRowView(
                vehicle: vehicle,
                realtimeData: supabase.vehicleRealtimeData[vehicle.id]
              )
              .onTapGesture {
                selectedVehicle = vehicle
              }
            }
          }
        }
      }
      .navigationTitle("My Vehicles")
      .toolbar {
        ToolbarItem(placement: .primaryAction) {
          Button {
            showingJoinVehicle = true
          } label: {
            Image(systemName: "plus")
          }
        }
      }
      .sheet(isPresented: $showingJoinVehicle) {
        JoinVehicleView()
      }
      .sheet(item: $selectedVehicle) { vehicle in
        VehicleDetailView(
          vehicle: vehicle,
          realtimeData: supabase.vehicleRealtimeData[vehicle.id]
        )
      }
      .refreshable {
        await supabase.loadVehicles()
      }
    }
  }
}

struct VehicleRowView: View {
  let vehicle: Vehicle
  let realtimeData: VehicleRealtime?

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text(vehicle.name ?? vehicle.id)
          .font(.headline)

        Spacer()

        if let data = realtimeData {
          DriverStatusBadge(status: data.driverStatus)
        }
      }

      if let data = realtimeData {
        HStack(spacing: 16) {
          // Speed
          HStack(spacing: 4) {
            Image(systemName: "speedometer")
              .foregroundStyle(data.isSpeeding ? .red : .secondary)
            Text("\(data.speedMph) mph")
              .foregroundStyle(data.isSpeeding ? .red : .primary)
          }

          // Heading
          HStack(spacing: 4) {
            Image(systemName: "location.north.fill")
              .rotationEffect(.degrees(Double(data.headingDegrees)))
              .foregroundStyle(.secondary)
            Text("\(data.headingDegrees)° \(data.compassDirection)")
          }

          Spacer()

          // Last updated
          Text(data.updatedAt.formatted(date: .abbreviated, time: .standard))
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .font(.subheadline)
      } else {
        Text("No data available")
          .font(.subheadline)
          .foregroundStyle(.secondary)
      }
    }
    .padding(.vertical, 4)
  }
}

struct VehicleDetailView: View {
  let vehicle: Vehicle
  let realtimeData: VehicleRealtime?

  var body: some View {
    NavigationStack {
      List {
        Section("Vehicle Info") {
          LabeledContent("ID", value: vehicle.id)
          LabeledContent("Name", value: vehicle.name ?? "Unknown")
          LabeledContent("Description", value: vehicle.description ?? "No description")
        }

        if let data = realtimeData {
          Section("Speed & Direction") {
            LabeledContent("Current Speed", value: "\(data.speedMph) mph")
            LabeledContent("Speed Limit", value: "\(data.speedLimitMph) mph")
            LabeledContent("Heading", value: "\(data.headingDegrees)° \(data.compassDirection)")

            if data.isSpeeding {
              Label("Speeding!", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            }
          }

          Section("Driver Status") {
            HStack {
              Text("Status")
              Spacer()
              DriverStatusBadge(status: data.driverStatus)
            }
            LabeledContent("Risk Score", value: "\(data.intoxicationScore)/6")
          }

          Section("Activity") {
            LabeledContent("Moving", value: data.isMoving ? "Yes" : "No")
            HStack {
              Text("Last Updated")
              Spacer()
              Text(data.updatedAt, style: .relative)
                .foregroundStyle(.secondary)
            }
          }
        } else {
          Section {
            ContentUnavailableView {
              Label("No Data", systemImage: "antenna.radiowaves.left.and.right.slash")
            } description: {
              Text("Vehicle is not currently transmitting data.")
            }
          }
        }
      }
      .navigationTitle(vehicle.name ?? "Vehicle")
      .navigationBarTitleDisplayMode(.inline)
    }
  }
}

#Preview {
  VehicleListView()
}
