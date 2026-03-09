//
//  BluetoothManager.swift
//  InfineonProject
//
//  Created by Aaron Ma on 3/4/26.
//

import CoreBluetooth
import Foundation

// MARK: - UUIDs (must match RPi bluetooth_peripheral.py)

private let serviceUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-1234567890AB")
private let realtimeCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780001")
private let settingsCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780002")
private let buzzerCharUUID = CBUUID(string: "A1B2C3D4-E5F6-7890-ABCD-123456780003")

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

// MARK: - Discovered BLE device

struct DiscoveredBLEDevice: Identifiable {
  let id: UUID
  let peripheral: CBPeripheral
  let name: String
  var rssi: Int
  let isVehicle: Bool
  var isConnectable: Bool?

  init(peripheral: CBPeripheral, rssi: Int, isVehicle: Bool, isConnectable: Bool?) {
    self.id = peripheral.identifier
    self.peripheral = peripheral
    self.name = peripheral.name ?? "Unknown Device"
    self.rssi = rssi
    self.isVehicle = isVehicle
    self.isConnectable = isConnectable
  }
}

// MARK: - BluetoothManager

@Observable
final class BluetoothManager: NSObject {
  enum PiConnectionState {
    case online
    case inactive
    case offline
  }

  enum ConnectionPhase {
    case idle
    case scanning
    case connecting
    case connected
  }

  private enum DefaultsKey {
    static let preferredVehicleId = "bluetooth.preferredVehicleId"
    static let targetPeripheralId = "bluetooth.targetPeripheralId"
  }

  // Public state
  var bleEnabled = false {
    didSet {
      if bleEnabled {
        allowAutoReconnect = true
        userInitiatedDisconnect = false
        reconnectAttempt = 0
        consecutiveConnectFailures = 0
        staleTargetRetrievalMisses = 0
        bypassCachedTargetReconnectOnce = false
        reconnectWorkItem?.cancel()
        reconnectScheduled = false
        startScanning()
      } else {
        allowAutoReconnect = false
        disconnect(manual: true)
      }
    }
  }

  private(set) var isConnected = false
  private(set) var isScanning = false
  private(set) var statusMessage = "Off"
  private(set) var latestRealtime: BLERealtimeData?
  private(set) var discoveredDevices: [DiscoveredBLEDevice] = []
  private(set) var connectedDeviceName: String?
  private(set) var connectionPhase: ConnectionPhase = .idle
  /// Timestamp of the last realtime BLE packet received from the Pi.
  private(set) var lastDataReceivedAt: Date?
  /// True while connected and realtime data has arrived within the last 5 seconds.
  private(set) var isPiDataActive = false
  /// Fine-grained Pi state for BLE freshness.
  private(set) var piConnectionState: PiConnectionState = .offline

  private static let dataOnlineThreshold: TimeInterval = 5
  private static let dataInactiveThreshold: TimeInterval = 10

  /// Set by the UI when a vehicle profile is selected while BLE is connected.
  var connectedVehicleId: String? {
    didSet {
      if let connectedVehicleId {
        UserDefaults.standard.set(connectedVehicleId, forKey: DefaultsKey.preferredVehicleId)
      } else {
        UserDefaults.standard.removeObject(forKey: DefaultsKey.preferredVehicleId)
      }
    }
  }

  // CoreBluetooth
  private var centralManager: CBCentralManager?
  private var connectedPeripheral: CBPeripheral?
  private var settingsCharacteristic: CBCharacteristic?
  private var buzzerCharacteristic: CBCharacteristic?

  // Reconnect / timeout
  private var scanTimer: Timer?
  private var reconnectWorkItem: DispatchWorkItem?
  private var reconnectAttempt = 0
  private var reconnectScheduled = false
  private var consecutiveConnectFailures = 0
  private var staleTargetRetrievalMisses = 0
  private var userInitiatedDisconnect = false
  private var allowAutoReconnect = true
  private var connectingPeripheralId: UUID?
  private var targetPeripheralId: UUID?
  private var bypassCachedTargetReconnectOnce = false

  // BLE → vehicleRealtimeData polling
  private var dataTimer: Timer?

  private static let scanTimeout: TimeInterval = 15
  private static let reconnectBaseDelay: TimeInterval = 1
  private static let reconnectMaxDelay: TimeInterval = 12
  private static let reconnectFailureLimit = 6
  private static let staleTargetMissLimit = 3

  var hasSavedPeripheral: Bool { targetPeripheralId != nil }
  var savedPeripheralId: String? { targetPeripheralId?.uuidString }

  override init() {
    if let rawVehicleId = UserDefaults.standard.string(forKey: DefaultsKey.preferredVehicleId),
      !rawVehicleId.isEmpty
    {
      connectedVehicleId = rawVehicleId
    }
    if let rawPeripheralId = UserDefaults.standard.string(forKey: DefaultsKey.targetPeripheralId) {
      targetPeripheralId = UUID(uuidString: rawPeripheralId)
    }
    super.init()
  }

  // MARK: - Data Polling Timer

  private func startDataTimer() {
    guard dataTimer == nil else { return }
    dataTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
      guard let self, let vid = self.connectedVehicleId else { return }
      supabase.updateFromBLE(vehicleId: vid)
      self.updatePiConnectionState()
    }
  }

  private func stopDataTimer() {
    dataTimer?.invalidate()
    dataTimer = nil
  }

  private func updatePiConnectionState() {
    guard let last = lastDataReceivedAt else {
      isPiDataActive = false
      piConnectionState = .offline
      return
    }
    let age = Date.now.timeIntervalSince(last)
    if age <= Self.dataOnlineThreshold {
      isPiDataActive = true
      piConnectionState = .online
    } else if age <= Self.dataInactiveThreshold {
      isPiDataActive = false
      piConnectionState = .inactive
    } else {
      isPiDataActive = false
      piConnectionState = .offline
    }
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
    guard bleEnabled else { return }
    if connectionPhase == .connecting, connectingPeripheralId == device.id {
      return
    }
    if connectionPhase == .connected, connectedPeripheral?.identifier == device.id {
      return
    }
    if device.isConnectable == false {
      statusMessage = "Device not connectable. Keep Pi BLE advertising and retry."
      bleLog(
        "connect_blocked_not_connectable",
        details: [
          "id": device.peripheral.identifier.uuidString,
          "name": device.name,
        ])
      return
    }
    allowAutoReconnect = true
    userInitiatedDisconnect = false
    reconnectAttempt = 0
    consecutiveConnectFailures = 0
    staleTargetRetrievalMisses = 0
    bypassCachedTargetReconnectOnce = false
    reconnectWorkItem?.cancel()
    reconnectScheduled = false
    cm.stopScan()
    connectionPhase = .idle
    isScanning = false
    scanTimer?.invalidate()
    targetPeripheralId = device.peripheral.identifier
    UserDefaults.standard.set(device.peripheral.identifier.uuidString, forKey: DefaultsKey.targetPeripheralId)
    bleLog(
      "connect_request",
      details: [
        "id": device.peripheral.identifier.uuidString,
        "name": device.name,
        "rssi": "\(device.rssi)",
        "connectable": connectableString(device.isConnectable),
      ])
    statusMessage = "Connecting..."
    connectPeripheral(device.peripheral)
  }

  func forgetSavedPeripheral() {
    bleLog("forget_saved_peripheral", details: ["had_saved_id": "\(targetPeripheralId != nil)"])
    clearTargetPeripheral(reason: "user_requested")
    reconnectAttempt = 0
    consecutiveConnectFailures = 0
    staleTargetRetrievalMisses = 0
    bypassCachedTargetReconnectOnce = false
    allowAutoReconnect = false

    if isConnected || connectionPhase == .connecting {
      disconnect(manual: true)
      return
    }

    if bleEnabled {
      statusMessage = "Select a device"
      beginScan()
    } else {
      statusMessage = "Off"
    }
  }

  // MARK: - Scanning

  private func startScanning() {
    if centralManager == nil {
      centralManager = CBCentralManager(delegate: self, queue: .main)
      bleLog("central_start")
    } else if centralManager?.state == .poweredOn {
      attemptReconnectOrScan()
    }
  }

  private func connectPeripheral(_ peripheral: CBPeripheral) {
    guard let cm = centralManager else { return }
    guard bleEnabled else { return }
    if connectionPhase == .connecting, connectingPeripheralId == peripheral.identifier {
      return
    }
    if connectionPhase == .connected, connectedPeripheral?.identifier == peripheral.identifier {
      return
    }
    scanTimer?.invalidate()
    cm.stopScan()
    isScanning = false
    reconnectWorkItem?.cancel()
    reconnectScheduled = false
    connectionPhase = .connecting
    connectingPeripheralId = peripheral.identifier
    let connectable = discoveredDevices.first(where: { $0.id == peripheral.identifier })?.isConnectable
    bleLog(
      "connect_attempt",
      details: [
        "id": peripheral.identifier.uuidString,
        "connectable": connectableString(connectable),
        "options": "nil",
      ])
    statusMessage = "Connecting..."
    connectedPeripheral = peripheral
    peripheral.delegate = self
    cm.connect(peripheral, options: nil)
  }

  private func attemptReconnectOrScan() {
    guard let cm = centralManager, cm.state == .poweredOn else { return }
    guard bleEnabled else { return }
    guard connectionPhase != .connecting, connectionPhase != .connected else { return }
    guard !isScanning else { return }
    if allowAutoReconnect, let targetPeripheralId {
      if bypassCachedTargetReconnectOnce {
        bypassCachedTargetReconnectOnce = false
        staleTargetRetrievalMisses = 0
        bleLog(
          "target_reconnect_scan_forced",
          details: [
            "id": targetPeripheralId.uuidString,
            "reason": "invalid_parameters",
          ])
        beginScan()
        return
      }
      let known = cm.retrievePeripherals(withIdentifiers: [targetPeripheralId])
      if let peripheral = known.first {
        staleTargetRetrievalMisses = 0
        statusMessage = "Reconnecting..."
        bleLog("target_reconnect_attempt", details: ["id": peripheral.identifier.uuidString])
        connectPeripheral(peripheral)
        return
      }
      staleTargetRetrievalMisses += 1
      bleLog(
        "target_reconnect_miss",
        details: [
          "id": targetPeripheralId.uuidString,
          "misses": "\(staleTargetRetrievalMisses)",
        ])
      if staleTargetRetrievalMisses >= Self.staleTargetMissLimit {
        clearTargetPeripheral(reason: "unresolvable_cached_peripheral")
        allowAutoReconnect = false
        statusMessage = "Select a device"
      }
    }
    beginScan()
  }

  private func beginScan() {
    guard let cm = centralManager, cm.state == .poweredOn else { return }
    guard bleEnabled else { return }
    guard connectionPhase != .connecting, connectionPhase != .connected else { return }
    guard !isScanning else { return }
    bleLog("scan_start")
    discoveredDevices = []
    isScanning = true
    connectionPhase = .scanning
    statusMessage = "Scanning..."
    cm.scanForPeripherals(
      withServices: [serviceUUID], options: [CBCentralManagerScanOptionAllowDuplicatesKey: false])

    scanTimer?.invalidate()
    scanTimer = Timer.scheduledTimer(withTimeInterval: Self.scanTimeout, repeats: false) {
      [weak self] _ in
      guard let self, !self.isConnected else { return }
      self.centralManager?.stopScan()
      self.isScanning = false
      self.connectionPhase = .idle
      if self.discoveredDevices.isEmpty {
        self.statusMessage = "Device not found"
        self.bleLog("scan_timeout", details: ["result": "empty"])
      } else {
        self.statusMessage = "Select a device"
        self.bleLog(
          "scan_timeout",
          details: ["result": "devices_found", "count": "\(self.discoveredDevices.count)"])
      }
    }
  }

  func disconnect(manual: Bool = true) {
    scanTimer?.invalidate()
    reconnectWorkItem?.cancel()
    reconnectScheduled = false
    if manual {
      userInitiatedDisconnect = true
      allowAutoReconnect = false
      reconnectAttempt = 0
      consecutiveConnectFailures = 0
      staleTargetRetrievalMisses = 0
      bypassCachedTargetReconnectOnce = false
    }
    if let peripheral = connectedPeripheral {
      centralManager?.cancelPeripheralConnection(peripheral)
    }
    centralManager?.stopScan()
    clearSessionState(clearSelectedContext: !bleEnabled)
    bleLog("disconnect", details: ["manual": "\(manual)"])
    statusMessage = bleEnabled ? "Disconnected" : "Off"
  }

  private func clearTransientConnectionState() {
    scanTimer?.invalidate()
    scanTimer = nil
    reconnectWorkItem?.cancel()
    reconnectWorkItem = nil
    reconnectScheduled = false
    stopDataTimer()
    centralManager?.stopScan()
    isConnected = false
    isScanning = false
    connectedPeripheral = nil
    settingsCharacteristic = nil
    buzzerCharacteristic = nil
    connectedDeviceName = nil
    connectingPeripheralId = nil
    connectionPhase = .idle
  }

  private func clearSessionState(clearSelectedContext: Bool) {
    clearTransientConnectionState()
    latestRealtime = nil
    discoveredDevices = []
    lastDataReceivedAt = nil
    isPiDataActive = false
    piConnectionState = .offline
    if clearSelectedContext {
      connectedVehicleId = nil
    }
  }

  private func scheduleReconnect() {
    guard bleEnabled, allowAutoReconnect else { return }
    guard consecutiveConnectFailures < Self.reconnectFailureLimit else { return }
    guard !reconnectScheduled else { return }
    guard connectionPhase != .connected, connectionPhase != .connecting else { return }
    statusMessage = "Reconnecting..."
    bleLog(
      "reconnect_scheduled",
      details: [
        "attempt": "\(reconnectAttempt + 1)",
        "failures": "\(consecutiveConnectFailures)",
      ])
    reconnectWorkItem?.cancel()
    reconnectScheduled = true
    reconnectAttempt += 1
    let exponent = max(0, reconnectAttempt - 1)
    let delay = min(
      Self.reconnectMaxDelay,
      Self.reconnectBaseDelay * pow(2.0, Double(exponent))
    )
    let work = DispatchWorkItem { [weak self] in
      guard let self else { return }
      self.reconnectScheduled = false
      self.bleLog("reconnect_execute", details: ["attempt": "\(self.reconnectAttempt)"])
      self.attemptReconnectOrScan()
    }
    reconnectWorkItem = work
    DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
  }

  private func clearTargetPeripheral(reason: String) {
    guard targetPeripheralId != nil else { return }
    bleLog("target_cleared", details: ["reason": reason, "id": targetPeripheralId?.uuidString ?? "nil"])
    targetPeripheralId = nil
    bypassCachedTargetReconnectOnce = false
    UserDefaults.standard.removeObject(forKey: DefaultsKey.targetPeripheralId)
  }

  private func bleLog(_ event: String, details: [String: String] = [:]) {
    let detailText = details
      .sorted(by: { $0.key < $1.key })
      .map { "\($0.key)=\($0.value)" }
      .joined(separator: " ")
    if detailText.isEmpty {
      print("[BLE][\(event)]")
    } else {
      print("[BLE][\(event)] \(detailText)")
    }
  }

  private func describe(error: Error?) -> String {
    guard let ns = error as NSError? else { return "none" }
    let message = ns.localizedDescription.replacing("\n", with: " ")
    return "\(ns.domain)(\(ns.code)) \(message)"
  }

  private func connectableString(_ flag: Bool?) -> String {
    guard let flag else { return "unknown" }
    return flag ? "true" : "false"
  }

  private func isInvalidParameterConnectError(_ error: Error?) -> Bool {
    guard let nsError = error as NSError? else { return false }
    return nsError.domain == CBErrorDomain
      && nsError.code == CBError.Code.invalidParameters.rawValue
  }
}

// MARK: - CBCentralManagerDelegate

extension BluetoothManager: CBCentralManagerDelegate {
  func centralManagerDidUpdateState(_ central: CBCentralManager) {
    switch central.state {
    case .poweredOn:
      if bleEnabled, connectionPhase == .idle {
        reconnectAttempt = 0
        attemptReconnectOrScan()
      }
    case .poweredOff:
      statusMessage = "Bluetooth is off"
      clearSessionState(clearSelectedContext: false)
      bleLog("powered_off")
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
    guard rssi != 127 else { return }
    guard connectionPhase != .connecting, connectionPhase != .connected else { return }
    let advertisedServices =
      advertisementData[CBAdvertisementDataServiceUUIDsKey] as? [CBUUID] ?? []
    let isVehicle = advertisedServices.contains(serviceUUID)
    let isConnectable: Bool? = {
      if let boolValue = advertisementData[CBAdvertisementDataIsConnectable] as? Bool {
        return boolValue
      }
      if let numberValue = advertisementData[CBAdvertisementDataIsConnectable] as? NSNumber {
        return numberValue.boolValue
      }
      return nil
    }()

    if let idx = discoveredDevices.firstIndex(where: {
      $0.peripheral.identifier == peripheral.identifier
    }) {
      discoveredDevices[idx].rssi = rssi
      discoveredDevices[idx].isConnectable = isConnectable
    } else {
      let device = DiscoveredBLEDevice(
        peripheral: peripheral,
        rssi: rssi,
        isVehicle: isVehicle,
        isConnectable: isConnectable
      )
      if isVehicle {
        discoveredDevices.insert(device, at: 0)
      } else {
        discoveredDevices.append(device)
      }
      bleLog(
        "device_discovered",
        details: [
          "id": peripheral.identifier.uuidString,
          "name": peripheral.name ?? "Unknown",
          "rssi": "\(rssi)",
          "connectable": connectableString(isConnectable),
          "vehicle": "\(isVehicle)",
        ])

      let isTarget = targetPeripheralId == peripheral.identifier
      if allowAutoReconnect, isTarget {
        if isConnectable == false {
          statusMessage = "Saved device found but not connectable. Keep Pi advertising."
          bleLog("target_not_connectable", details: ["id": peripheral.identifier.uuidString])
          return
        }
        statusMessage = "Reconnecting..."
        bleLog(
          "target_discovered",
          details: [
            "id": peripheral.identifier.uuidString,
            "connectable": connectableString(isConnectable),
          ])
        connectPeripheral(peripheral)
      }
    }
  }

  func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
    isConnected = true
    reconnectAttempt = 0
    consecutiveConnectFailures = 0
    staleTargetRetrievalMisses = 0
    bypassCachedTargetReconnectOnce = false
    reconnectScheduled = false
    userInitiatedDisconnect = false
    connectionPhase = .connected
    isScanning = false
    connectingPeripheralId = nil
    connectedDeviceName = peripheral.name ?? "Unknown Device"
    statusMessage = "Connected"
    discoveredDevices = []
    targetPeripheralId = peripheral.identifier
    UserDefaults.standard.set(peripheral.identifier.uuidString, forKey: DefaultsKey.targetPeripheralId)
    bleLog("connected", details: ["id": peripheral.identifier.uuidString, "name": peripheral.name ?? "Unknown"])
    startDataTimer()
    peripheral.discoverServices([serviceUUID])
  }

  func centralManager(
    _ central: CBCentralManager,
    didFailToConnect peripheral: CBPeripheral,
    error: Error?
  ) {
    let invalidParameters = isInvalidParameterConnectError(error)
    consecutiveConnectFailures += 1
    bleLog(
      "connect_failed",
      details: [
        "id": peripheral.identifier.uuidString,
        "error": describe(error: error),
        "failure": "\(consecutiveConnectFailures)",
        "invalid_parameters": "\(invalidParameters)",
      ])
    if invalidParameters {
      bypassCachedTargetReconnectOnce = true
      staleTargetRetrievalMisses = 0
      bleLog(
        "connect_failed_fresh_scan_required",
        details: [
          "id": peripheral.identifier.uuidString,
          "reason": "invalid_parameters",
        ])
    }
    clearTransientConnectionState()
    if consecutiveConnectFailures >= Self.reconnectFailureLimit {
      clearTargetPeripheral(reason: "consecutive_failures")
      consecutiveConnectFailures = 0
      staleTargetRetrievalMisses = 0
      reconnectWorkItem?.cancel()
      reconnectScheduled = false
      allowAutoReconnect = false
      statusMessage = "Saved device unavailable. Select a device."
      beginScan()
      return
    }
    scheduleReconnect()
  }

  func centralManager(
    _ central: CBCentralManager,
    didDisconnectPeripheral peripheral: CBPeripheral,
    error: Error?
  ) {
    bleLog(
      "disconnected",
      details: [
        "id": peripheral.identifier.uuidString,
        "error": describe(error: error),
      ])
    let wasManual = userInitiatedDisconnect
    userInitiatedDisconnect = false
    clearTransientConnectionState()
    if !wasManual {
      scheduleReconnect()
    }
  }
}

// MARK: - CBPeripheralDelegate

extension BluetoothManager: CBPeripheralDelegate {
  func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
    guard let service = peripheral.services?.first(where: { $0.uuid == serviceUUID }) else {
      return
    }
    peripheral.discoverCharacteristics(
      [realtimeCharUUID, settingsCharUUID, buzzerCharUUID],
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
    guard characteristic.uuid == realtimeCharUUID else { return }
    let decoder = JSONDecoder()
    if let parsed = try? decoder.decode(BLERealtimeData.self, from: data) {
      latestRealtime = parsed
      lastDataReceivedAt = .now
      isPiDataActive = true
      piConnectionState = .online
    }
  }
}

// MARK: - Global singleton

let bluetooth = BluetoothManager()
