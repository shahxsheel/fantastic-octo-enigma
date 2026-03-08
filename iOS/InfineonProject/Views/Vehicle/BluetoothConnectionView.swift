//
//  BluetoothConnectionView.swift
//  InfineonProject
//
//  Created by Aaron Ma on 3/4/26.
//

import SwiftUI

struct BluetoothConnectionView: View {
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      List {
        Section {
          Toggle(
            isOn: Binding(get: { bluetooth.bleEnabled }, set: { bluetooth.bleEnabled = $0 })
              .animation()
          ) {
            Label {
              VStack(alignment: .leading) {
                Text("Phone Connection")
                Group {
                  if bluetooth.isConnected {
                    Text("Connected")
                      .foregroundStyle(.green)
                  } else if bluetooth.bleEnabled {
                    Text(bluetooth.statusMessage)
                      .foregroundStyle(
                        bluetooth.statusMessage.contains("not found")
                          || bluetooth.statusMessage.contains("not authorized") ? .red : .secondary)
                  } else {
                    Text("Off")
                      .foregroundStyle(.secondary)
                  }
                }
                .font(.caption)
              }
            } icon: {
              SettingsBoxView(
                icon: bluetooth.isConnected
                  ? "antenna.radiowaves.left.and.right" : "antenna.radiowaves.left.and.right.slash",
                color: bluetooth.isConnected ? .green : .gray
              )
            }
          }
        } header: {
          Text("Bluetooth")
        } footer: {
          Text(
            "Connect directly to your vehicle over Bluetooth for local telemetry relay and buzzer control."
          )
        }

        if bluetooth.isConnected {
          Section {
            LabeledContent("Device") {
              Text(bluetooth.connectedDeviceName ?? "Unknown")
            }

            LabeledContent("Status") {
              if bluetooth.isPiDataActive {
                HStack(spacing: 6) {
                  Circle()
                    .fill(.green)
                    .frame(width: 8, height: 8)
                  Text("Pi Streaming")
                    .foregroundStyle(.green)
                }
              } else {
                HStack(spacing: 6) {
                  Circle()
                    .fill(.orange)
                    .frame(width: 8, height: 8)
                  Text("Pi Inactive")
                    .foregroundStyle(.orange)
                }
              }
            }

            Button(role: .destructive) {
              bluetooth.disconnect()
            } label: {
              HStack {
                Spacer()
                Text("Disconnect")
                Spacer()
              }
            }
          } header: {
            Text("Connection Info")
          } footer: {
            Text(
              "When connected via Bluetooth, the app can relay vehicle data to the cloud if your phone has internet access."
            )
          }
        } else if bluetooth.bleEnabled {
          Section {
            if bluetooth.isScanning {
              HStack(spacing: 12) {
                ProgressView()
                Text("Scanning for nearby devices...")
                  .foregroundStyle(.secondary)
              }
            }

            if bluetooth.discoveredDevices.isEmpty && !bluetooth.isScanning {
              HStack {
                Image(systemName: "antenna.radiowaves.left.and.right.slash")
                  .foregroundStyle(.secondary)
                Text("No devices found")
                  .foregroundStyle(.secondary)
              }
            }

            ForEach(bluetooth.discoveredDevices) { device in
              Button {
                bluetooth.connectToDevice(device)
              } label: {
                HStack {
                  Image(systemName: device.isVehicle ? "car.fill" : "dot.radiowaves.left.and.right")
                    .foregroundStyle(device.isVehicle ? .blue : .secondary)
                    .frame(width: 24)

                  VStack(alignment: .leading) {
                    HStack(spacing: 6) {
                      Text(device.name)
                      if device.isVehicle {
                        Text("Vehicle")
                          .font(.caption2)
                          .fontWeight(.medium)
                          .padding(.horizontal, 6)
                          .padding(.vertical, 2)
                          .background(.blue.opacity(0.15))
                          .foregroundStyle(.blue)
                          .clipShape(Capsule())
                      }
                    }
                    Text(signalDescription(rssi: device.rssi))
                      .font(.caption)
                      .foregroundStyle(.secondary)
                  }

                  Spacer()

                  signalIcon(rssi: device.rssi)
                    .foregroundStyle(signalColor(rssi: device.rssi))
                }
              }
              .tint(.primary)
            }
          } header: {
            Text("Nearby Devices")
          }
        }
      }
      .navigationTitle("Phone Connection")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          CloseButton {
            dismiss()
          }
        }
      }
    }
  }

  private func signalIcon(rssi: Int) -> Image {
    switch rssi {
    case -50...0: Image(systemName: "wifi", variableValue: 1.0)
    case -70..<(-50): Image(systemName: "wifi", variableValue: 0.67)
    case -85..<(-70): Image(systemName: "wifi", variableValue: 0.33)
    default: Image(systemName: "wifi", variableValue: 0.01)
    }
  }

  private func signalColor(rssi: Int) -> Color {
    switch rssi {
    case -50...0: .green
    case -70..<(-50): .yellow
    case -85..<(-70): .orange
    default: .red
    }
  }

  private func signalDescription(rssi: Int) -> String {
    switch rssi {
    case -50...0: "Excellent signal (\(rssi) dBm)"
    case -70..<(-50): "Good signal (\(rssi) dBm)"
    case -85..<(-70): "Fair signal (\(rssi) dBm)"
    default: "Weak signal (\(rssi) dBm)"
    }
  }
}

#Preview {
  BluetoothConnectionView()
}
