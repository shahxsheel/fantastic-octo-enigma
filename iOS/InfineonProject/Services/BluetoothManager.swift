//
//  BluetoothManager.swift
//  InfineonProject
//
//  Created by Aaron Ma on 3/4/26.
//

import CoreBluetooth
import Foundation

// MARK: - UUIDs (must match RPi bluetooth.py)

private let serviceUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-1234567890AB")
private let realtimeCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780001")
private let settingsCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780002")
private let buzzerCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780003")
private let tripCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780004")
private let relayCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780005")
// MARK: - Compact BLE payload models

struct BLERealtimeData: Codable {
  let spd: Int
  let hdg: Int
  let lat: Double
  let lng: Double
  let dir: String
  let ds: String
  let ph: Bool
  let dr: Bool
  let ix: Int
  let sp: Bool
  let sat: Int

  /// Convert compact BLE JSON to the app's existing VehicleRealtime model.
  func toVehicleRealtime(vehicleId: String) -> VehicleRealtime {
    VehicleRealtime(
      vehicleId: vehicleId,
      updatedAt: .now,
      latitude: lat,
      longitude: lng,
      speedMph: spd,
      speedLimitMph: 0,
      headingDegrees: hdg,
      compassDirection: dir,
      isSpeeding: sp,
      isMoving: spd > 0,
      driverStatus: ds,
      intoxicationScore: ix,
      satellites: sat,
      isPhoneDetected: ph,
      isDrinkingDetected: dr
    )
  }
}

struct BLESettingsData: Codable {
  var yolo: Bool
  var mic: Bool
  var cam: Bool
  var dash: Bool

  /// Build from existing Vehicle toggle values.
  init(from vehicle: Vehicle) {
    yolo = vehicle.enableYolo
    mic = vehicle.enableMicrophone
    cam = vehicle.enableCamera
    dash = vehicle.enableDashcam
  }
}

struct BLETripData: Codable {
  let tid: String
  let dur: Int
  let mx_spd: Int
  let avg_spd: Double
  let spd_ev: Int
  let drw_ev: Int
  let ph_ev: Int
  let ix_max: Int
}

/// Relay data: full Supabase-format records for iOS to upload on Pi's behalf.
struct BLERelayData: Codable {
  let rt: [String: AnyCodable]?
  let trip: [String: AnyCodable]?
}

/// Type-erased Codable wrapper for relay JSON values.
enum AnyCodable: Codable {
  case string(String)
  case int(Int)
  case double(Double)
  case bool(Bool)
  case null

  init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let v = try? container.decode(Bool.self) {
      self = .bool(v)
    } else if let v = try? container.decode(Int.self) {
      self = .int(v)
    } else if let v = try? container.decode(Double.self) {
      self = .double(v)
    } else if let v = try? container.decode(String.self) {
      self = .string(v)
    } else if container.decodeNil() {
      self = .null
    } else {
      self = .null
    }
  }

  func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let v): try container.encode(v)
    case .int(let v): try container.encode(v)
    case .double(let v): try container.encode(v)
    case .bool(let v): try container.encode(v)
    case .null: try container.encodeNil()
    }
  }

}

// MARK: - Discovered BLE device

struct DiscoveredBLEDevice: Identifiable {
  let id: UUID
  let peripheral: CBPeripheral
  let name: String
  var rssi: Int
  let isVehicle: Bool

  init(peripheral: CBPeripheral, rssi: Int, isVehicle: Bool) {
    self.id = peripheral.identifier
    self.peripheral = peripheral
    self.name = peripheral.name ?? "Unknown Device"
    self.rssi = rssi
    self.isVehicle = isVehicle
  }
}

// MARK: - BluetoothManager

@Observable
final class BluetoothManager: NSObject {
  // Public state
  var bleEnabled = false {
    didSet {
      if bleEnabled {
        startScanning()
      } else {
        disconnect()
      }
    }
  }

  private(set) var isConnected = false
  private(set) var isScanning = false
  private(set) var statusMessage = "Off"
  private(set) var latestRealtime: BLERealtimeData?
  private(set) var latestTrip: BLETripData?
  private(set) var latestRelay: BLERelayData?
  private(set) var discoveredDevices: [DiscoveredBLEDevice] = []
  private(set) var connectedDeviceName: String?

  /// Set by the UI when a vehicle profile is selected while BLE is connected.
  /// The data polling timer uses this to feed BLE data into `vehicleRealtimeData`.
  var connectedVehicleId: String?

  // CoreBluetooth
  private var centralManager: CBCentralManager?
  private var connectedPeripheral: CBPeripheral?
  private var settingsCharacteristic: CBCharacteristic?
  private var buzzerCharacteristic: CBCharacteristic?

  // Reconnect / timeout
  private var scanTimer: Timer?
  private var reconnectWorkItem: DispatchWorkItem?

  // BLE → vehicleRealtimeData polling
  private var dataTimer: Timer?
  // BLE relay → Supabase (upload Pi data on its behalf)
  private var relayTimer: Timer?

  private static let scanTimeout: TimeInterval = 15
  private static let reconnectDelay: TimeInterval = 2

  override init() {
    super.init()
  }

  // MARK: - Data Polling Timer

  private func startDataTimer() {
    guard dataTimer == nil else { return }
    dataTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
      guard let self, let vid = self.connectedVehicleId else { return }
      supabase.updateFromBLE(vehicleId: vid)
    }
  }

  private func stopDataTimer() {
    dataTimer?.invalidate()
    dataTimer = nil
  }

  // MARK: - Relay Timer (upload Pi data to Supabase + poll buzzer)

  private func startRelayTimer() {
    guard relayTimer == nil else { return }
    relayTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
      guard let self, let vid = self.connectedVehicleId else { return }
      Task {
        await supabase.relayBLEDataToSupabase(vehicleId: vid)
        // Poll buzzer state from Supabase and relay back to Pi
        let buzzerState = await supabase.fetchBuzzerStateForRelay(vehicleId: vid)
        if let buzzerState {
          await MainActor.run {
            self.writeBuzzerCommand(active: buzzerState.active, type: buzzerState.type)
          }
        }
      }
    }
  }

  private func stopRelayTimer() {
    relayTimer?.invalidate()
    relayTimer = nil
  }

  // MARK: - Public Write Methods

  func writeSettings(_ settings: BLESettingsData) {
    guard let char = settingsCharacteristic, let peripheral = connectedPeripheral else { return }
    guard let data = try? JSONEncoder().encode(settings) else { return }
    peripheral.writeValue(data, for: char, type: .withResponse)
  }

  func writeBuzzerCommand(active: Bool, type: String = "alert") {
    guard let char = buzzerCharacteristic, let peripheral = connectedPeripheral else { return }
    let payload: [String: Any] = ["active": active, "type": type]
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
    peripheral.writeValue(data, for: char, type: .withResponse)
  }

  func writeCustomBuzzerCommand(
    active: Bool, frequency: Int, onDuration: Double, offDuration: Double, dutyCycle: Int
  ) {
    guard let char = buzzerCharacteristic, let peripheral = connectedPeripheral else { return }
    let payload: [String: Any] = [
      "active": active, "type": "custom",
      "freq": frequency, "on": onDuration, "off": offDuration, "duty": dutyCycle,
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
    peripheral.writeValue(data, for: char, type: .withResponse)
  }

  // MARK: - Public Connect

  func connectToDevice(_ device: DiscoveredBLEDevice) {
    guard let cm = centralManager else { return }
    cm.stopScan()
    isScanning = false
    scanTimer?.invalidate()
    statusMessage = "Connecting..."
    connectedPeripheral = device.peripheral
    device.peripheral.delegate = self
    cm.connect(device.peripheral, options: nil)
  }

  // MARK: - Scanning

  private func startScanning() {
    if centralManager == nil {
      centralManager = CBCentralManager(delegate: self, queue: .main)
      // Scanning starts in centralManagerDidUpdateState once powered on
    } else if centralManager?.state == .poweredOn {
      beginScan()
    }
  }

  private func beginScan() {
    guard let cm = centralManager, cm.state == .poweredOn else { return }
    discoveredDevices = []
    isScanning = true
    statusMessage = "Scanning..."
    cm.scanForPeripherals(
      withServices: nil, options: [CBCentralManagerScanOptionAllowDuplicatesKey: false])

    // Scan timeout
    scanTimer?.invalidate()
    scanTimer = Timer.scheduledTimer(withTimeInterval: Self.scanTimeout, repeats: false) {
      [weak self] _ in
      guard let self, !self.isConnected else { return }
      self.centralManager?.stopScan()
      self.isScanning = false
      if self.discoveredDevices.isEmpty {
        self.statusMessage = "Device not found"
      } else {
        self.statusMessage = "Select a device"
      }
    }
  }

  func disconnect() {
    scanTimer?.invalidate()
    reconnectWorkItem?.cancel()
    if let peripheral = connectedPeripheral {
      centralManager?.cancelPeripheralConnection(peripheral)
    }
    centralManager?.stopScan()
    cleanup()
    if bleEnabled {
      beginScan()
    } else {
      statusMessage = "Off"
    }
  }

  private func cleanup() {
    stopDataTimer()
    stopRelayTimer()
    isConnected = false
    isScanning = false
    connectedPeripheral = nil
    settingsCharacteristic = nil
    buzzerCharacteristic = nil
    latestRealtime = nil
    latestTrip = nil
    latestRelay = nil
    connectedVehicleId = nil
    connectedDeviceName = nil
    discoveredDevices = []
  }

  private func scheduleReconnect() {
    guard bleEnabled else { return }
    statusMessage = "Reconnecting..."
    reconnectWorkItem?.cancel()
    let work = DispatchWorkItem { [weak self] in
      self?.beginScan()
    }
    reconnectWorkItem = work
    DispatchQueue.main.asyncAfter(deadline: .now() + Self.reconnectDelay, execute: work)
  }
}

// MARK: - CBCentralManagerDelegate

extension BluetoothManager: CBCentralManagerDelegate {
  func centralManagerDidUpdateState(_ central: CBCentralManager) {
    switch central.state {
    case .poweredOn:
      if bleEnabled { beginScan() }
    case .poweredOff:
      statusMessage = "Bluetooth is off"
      cleanup()
    case .unauthorized:
      statusMessage = "Bluetooth not authorized"
    case .unsupported:
      statusMessage = "BLE not supported"
    default:
      break
    }
  }

  func centralManager(
    _ central: CBCentralManager,
    didDiscover peripheral: CBPeripheral,
    advertisementData: [String: Any],
    rssi RSSI: NSNumber
  ) {
    let rssi = RSSI.intValue
    guard rssi != 127 else { return }  // Invalid RSSI
    let advertisedServices =
      advertisementData[CBAdvertisementDataServiceUUIDsKey] as? [CBUUID] ?? []
    let isVehicle = advertisedServices.contains(serviceUUID)

    // Update existing or add new
    if let idx = discoveredDevices.firstIndex(where: {
      $0.peripheral.identifier == peripheral.identifier
    }) {
      discoveredDevices[idx].rssi = rssi
    } else {
      let device = DiscoveredBLEDevice(peripheral: peripheral, rssi: rssi, isVehicle: isVehicle)
      // Insert vehicles at the top, others at the bottom
      if isVehicle {
        discoveredDevices.insert(device, at: 0)
      } else {
        discoveredDevices.append(device)
      }

      // Auto-connect to vehicle devices
      if isVehicle {
        connectToDevice(device)
      }
    }
  }

  func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
    isConnected = true
    isScanning = false
    connectedDeviceName = peripheral.name ?? "Unknown Device"
    statusMessage = "Connected"
    discoveredDevices = []
    startDataTimer()
    startRelayTimer()
    peripheral.discoverServices([serviceUUID])
  }

  func centralManager(
    _ central: CBCentralManager,
    didFailToConnect peripheral: CBPeripheral,
    error: Error?
  ) {
    cleanup()
    scheduleReconnect()
  }

  func centralManager(
    _ central: CBCentralManager,
    didDisconnectPeripheral peripheral: CBPeripheral,
    error: Error?
  ) {
    cleanup()
    scheduleReconnect()
  }
}

// MARK: - CBPeripheralDelegate

extension BluetoothManager: CBPeripheralDelegate {
  func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
    guard let service = peripheral.services?.first(where: { $0.uuid == serviceUUID }) else {
      return
    }
    peripheral.discoverCharacteristics(
      [realtimeCharUUID, settingsCharUUID, buzzerCharUUID, tripCharUUID, relayCharUUID],
      for: service
    )
  }

  func peripheral(
    _ peripheral: CBPeripheral,
    didDiscoverCharacteristicsFor service: CBService,
    error: Error?
  ) {
    guard let chars = service.characteristics else { return }
    for char in chars {
      switch char.uuid {
      case realtimeCharUUID:
        peripheral.setNotifyValue(true, for: char)
        peripheral.readValue(for: char)
      case settingsCharUUID:
        settingsCharacteristic = char
      case buzzerCharUUID:
        buzzerCharacteristic = char
      case tripCharUUID:
        peripheral.setNotifyValue(true, for: char)
        peripheral.readValue(for: char)
      case relayCharUUID:
        peripheral.setNotifyValue(true, for: char)
      default:
        break
      }
    }
  }

  func peripheral(
    _ peripheral: CBPeripheral,
    didUpdateValueFor characteristic: CBCharacteristic,
    error: Error?
  ) {
    guard let data = characteristic.value else { return }
    let decoder = JSONDecoder()

    switch characteristic.uuid {
    case realtimeCharUUID:
      if let parsed = try? decoder.decode(BLERealtimeData.self, from: data) {
        latestRealtime = parsed
      }
    case tripCharUUID:
      if let parsed = try? decoder.decode(BLETripData.self, from: data) {
        latestTrip = parsed
      }
    case relayCharUUID:
      if let parsed = try? decoder.decode(BLERelayData.self, from: data) {
        latestRelay = parsed
      }
    default:
      break
    }
  }

}

// MARK: - Global singleton

let bluetooth = BluetoothManager()
