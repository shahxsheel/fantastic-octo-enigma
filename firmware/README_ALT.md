# Infineon PSoC 6 AI Evaluation Kit - Driver Monitoring Firmware

```zsh
❯ west boards | grep e84
kit_pse84_eval
kit_pse84_ai: (2 valid board targets)
- kit_pse84_ai/pse846gps2dbzc4a/m33 (low power core)
- kit_pse84_ai/pse846gps2dbzc4a/m55 (primary application core at up to 400 MHz with Helium DSP and Ethos-U55 NPU)
```

**Running:**

```zsh
cd ~/zephyrproject
west init
west update # WARNING: 5-10 minutes depending on your internet connection

west build -b kit_pse84_ai/pse846gps2dbzc4a/m55 /Users/aaronma/Developer/infineon/firmware --pristine
```

```zsh
/Users/aaronma/zephyrproject/infineon-openocd/bin/openocd \
    -s /Users/aaronma/zephyrproject/infineon-openocd/scripts \
    -s /Users/aaronma/zephyrproject/zephyr/boards/infineon/kit_pse84_ai/support \
    -f interface/kitprog3.cfg \
    -f target/infineon/pse84xgxs2.cfg \
    -c "init" \
    -c "reset halt" \
    -c "flash write_image erase /Users/aaronma/zephyrproject/build/zephyr/zephyr.hex" \
    -c "reset run" \
    -c "exit"
```

---

AI-powered driver monitoring system firmware for the Infineon PSoC 6 AI Evaluation Kit (CY8CKIT-062S2-AI). This firmware implements real-time face detection, eye state monitoring, and drowsiness/intoxication detection using TensorFlow Lite Micro.

## Features

- **Face Detection**: Real-time face detection using TensorFlow Lite Micro
- **Eye State Monitoring**: Eye Aspect Ratio (EAR) algorithm for open/closed detection
- **Drowsiness Detection**: Monitors prolonged eye closure patterns
- **Intoxication Assessment**: Multi-indicator risk scoring system
- **Audio/Visual Alerts**: Buzzer and LED alerts for dangerous conditions
- **Low Power**: Optimized for embedded operation on Cortex-M4

## Hardware Requirements

- **Board**: Infineon PSoC 6 AI Evaluation Kit (CY8CKIT-062S2-AI)
- **MCU**: CY8C624ABZI-S2D44 (Cortex-M4 @ 150MHz, 2MB Flash, 1MB SRAM)
- **Camera**: Compatible camera module (e.g., OV7670, OV2640)
- **Optional**: External speaker/buzzer for audio alerts

## Development Options

| Option | Complexity | Best For |
|--------|------------|----------|
| **Zephyr RTOS** | Easy | Official PSoC 6 support, production-ready (Recommended) |
| **Mbed OS** | Medium | ARM ecosystem, cloud connectivity |
| **Bare Metal + GCC** | Advanced | Full control, minimal dependencies |

> **Note**: PlatformIO does not have official PSoC 6 platform support. Zephyr RTOS is recommended as it has [official Infineon PSoC 6 board support](https://docs.zephyrproject.org/latest/boards/cypress/cy8ckit_062_wifi_bt/doc/index.html).

## Project Structure

```
firmware/
├── include/           # Header files
│   ├── config.h          # Configuration parameters
│   ├── face_detector.h   # Face detection interface
│   ├── eye_analyzer.h    # EAR analysis interface
│   ├── alert_system.h    # Alert system interface
│   └── camera_driver.h   # Camera interface
├── src/               # Source files
│   ├── main.c            # Main application
│   ├── face_detector.c   # TFLite face detection
│   ├── eye_analyzer.c    # Eye analysis implementation
│   ├── alert_system.c    # Alert system
│   └── camera_driver.c   # Camera driver (placeholder)
├── models/            # TFLite model files
├── config/            # Linker scripts and configs
├── CMakeLists.txt     # CMake build configuration
├── Makefile           # Standalone Makefile
└── README.md          # This file
```

## macOS Setup Instructions

### Prerequisites (All Options)

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ARM GNU Toolchain
brew install --cask gcc-arm-embedded

# Verify installation
arm-none-eabi-gcc --version

# Install OpenOCD for flashing/debugging
brew install open-ocd

# Install Python for model conversion
brew install python@3.12
```

---

## Option 1: Zephyr RTOS (Recommended)

Zephyr is a mature RTOS with official PSoC 6 support and TensorFlow Lite Micro integration.

### Install Zephyr

```bash
# Install west (Zephyr's meta-tool)
pip3 install west

# Create workspace and initialize Zephyr
mkdir ~/zephyrproject && cd ~/zephyrproject
west init
west update

# Install Python dependencies
pip3 install -r zephyr/scripts/requirements.txt

# Install Zephyr SDK
brew install cmake ninja gperf ccache dfu-util dtc

# Download Zephyr SDK
cd ~
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.8/zephyr-sdk-0.16.8_macos-aarch64.tar.xz
tar xf zephyr-sdk-0.16.8_macos-aarch64.tar.xz
cd zephyr-sdk-0.16.8
./setup.sh
```

### Create Zephyr Project Structure

```bash
cd firmware

# Create Zephyr-compatible structure
mkdir -p boards/arm/cy8ckit_062s2_ai
```

Create `prj.conf`:

```
# Zephyr configuration

CONFIG_GPIO=y
CONFIG_PWM=y
CONFIG_I2C=y
CONFIG_VIDEO=y

# TensorFlow Lite Micro
CONFIG_TENSORFLOW_LITE_MICRO=y
CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN=y

# Logging
CONFIG_LOG=y
CONFIG_PRINTK=y

# Memory
CONFIG_MAIN_STACK_SIZE=8192
CONFIG_HEAP_MEM_POOL_SIZE=262144
```

Create `CMakeLists.txt` for Zephyr:

```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(driver_monitor)

target_sources(app PRIVATE
    src/main.c
    src/face_detector.c
    src/eye_analyzer.c
    src/alert_system.c
    src/camera_driver.c
)

target_include_directories(app PRIVATE include)
```

### Build with Zephyr

```bash
cd firmware

# Set Zephyr environment
source ~/zephyrproject/zephyr/zephyr-env.sh

# Build for PSoC 6
west build -b cy8ckit_062_wifi_bt

# Flash
west flash

# Monitor
west espmonitor /dev/cu.usbmodem*
```

---

## Option 2: Mbed OS (ARM Ecosystem)

Mbed OS has Infineon/Cypress PSoC 6 support via the Mbed CLI.

### Install Mbed CLI

```bash
# Install Mbed CLI 2
pip3 install mbed-tools

# Or Mbed CLI 1 (legacy but stable)
pip3 install mbed-cli
```

### Initialize Mbed Project

```bash
cd firmware

# Initialize as Mbed project
mbed-tools new .

# Add PSoC 6 target library
mbed-tools deploy

# Configure target
mbed-tools configure -m CY8CKIT_062_WIFI_BT -t GCC_ARM
```

### Create mbed_app.json

```json
{
    "target_overrides": {
        "*": {
            "platform.stdio-baud-rate": 115200,
            "platform.default-serial-baud-rate": 115200
        },
        "CY8CKIT_062_WIFI_BT": {
            "target.components_add": ["PSOC6HAL"]
        }
    }
}
```

### Build with Mbed

```bash
# Build
mbed-tools compile -m CY8CKIT_062_WIFI_BT -t GCC_ARM

# Flash (uses pyOCD)
mbed-tools compile -m CY8CKIT_062_WIFI_BT -t GCC_ARM --flash
```

---

## Option 3: Bare Metal with GCC (Advanced)

Full control without any framework dependencies.

### Download PSoC 6 Headers and CMSIS

```bash
mkdir -p ~/psoc6-sdk && cd ~/psoc6-sdk

# Clone core libraries
git clone https://github.com/Infineon/mtb-pdl-cat1.git pdl
git clone https://github.com/Infineon/core-lib.git
git clone https://github.com/ARM-software/CMSIS_5.git cmsis

# Clone TFLite Micro
git clone https://github.com/tensorflow/tflite-micro.git
```

### Update Makefile Paths

Edit `firmware/Makefile` to point to your SDK:

```makefile
# Update these paths
PDL_DIR := $(HOME)/psoc6-sdk/pdl
CMSIS_DIR := $(HOME)/psoc6-sdk/cmsis
TFLM_DIR := $(HOME)/psoc6-sdk/tflite-micro
```

### Build

```bash
cd firmware

# Build debug
make DEBUG=1

# Build release
make DEBUG=0

# Flash via OpenOCD
make flash
```

---

## Quick Start Summary

| Method | Install Command | Build Command |
|--------|-----------------|---------------|
| **Zephyr** (Recommended) | `pip3 install west` | `west build -b cy8ckit_062_wifi_bt && west flash` |
| **Mbed** | `pip3 install mbed-tools` | `mbed-tools compile --flash` |
| **Bare Metal** | Clone repos manually | `make && make flash` |

**Recommendation**: Use **Zephyr RTOS** - it has official Infineon PSoC 6 support and includes TensorFlow Lite Micro.

### Flashing the Firmware

#### Using KitProg3 (built into evaluation kit)

```bash
# Flash via OpenOCD
make flash

# Or manually:
openocd -f interface/kitprog3.cfg -f target/psoc6_2m.cfg \
    -c "program build/driver_monitor.hex verify reset exit"
```

#### Using ModusToolbox Programmer

1. Open ModusToolbox Programmer from the tools menu
2. Select the connected kit
3. Choose the `.hex` file from `build/`
4. Click "Program"

### Debugging

#### Using VS Code (Recommended)

1. Install the "Cortex-Debug" extension
2. Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug PSoC 6",
            "type": "cortex-debug",
            "request": "launch",
            "servertype": "openocd",
            "cwd": "${workspaceFolder}",
            "executable": "${workspaceFolder}/build/driver_monitor.elf",
            "configFiles": [
                "interface/kitprog3.cfg",
                "target/psoc6_2m.cfg"
            ],
            "searchDir": ["${env:HOME}/ModusToolbox/tools_3.2/openocd/scripts"],
            "runToMain": true,
            "svdFile": "${env:HOME}/mtb_shared/mtb-pdl-cat1/devices/COMPONENT_CAT1A/svd/psoc6_02.svd"
        }
    ]
}
```

3. Press `F5` to start debugging

#### Using GDB Directly

```bash
# Terminal 1: Start OpenOCD server
make debug

# Terminal 2: Connect GDB
arm-none-eabi-gdb build/driver_monitor.elf
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) load
(gdb) continue
```

### Model Conversion

To convert the MediaPipe face detection model for TFLite Micro:

```bash
# Install TensorFlow
pip3 install tensorflow

# Download the model
curl -L -o face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Convert to TFLite (simplified example)
python3 scripts/convert_model.py face_landmarker.task models/face_detection_model.c
```

**Note**: The MediaPipe `.task` format requires extraction of the TFLite model. See the `research/` directory for the original Python implementation.

### Serial Monitor

View debug output via USB serial:

```bash
# Find the serial port
ls /dev/cu.usbmodem*

# Connect with screen
screen /dev/cu.usbmodem14201 115200

# Or use minicom
brew install minicom
minicom -D /dev/cu.usbmodem14201 -b 115200
```

### Configuration

Edit `include/config.h` to adjust parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_WIDTH` | 320 | Camera capture width |
| `CAMERA_HEIGHT` | 240 | Camera capture height |
| `CAMERA_FPS` | 15 | Target frame rate |
| `EAR_THRESHOLD` | 0.21 | Eye closed threshold |
| `DROWSINESS_FRAME_THRESHOLD` | 20 | Frames for drowsy alert |
| `INTOX_HIGH_RISK_THRESHOLD` | 4 | Score for high risk alert |
| `ALERT_COOLDOWN_MS` | 3000 | Delay between alerts |

### Troubleshooting

#### "arm-none-eabi-gcc: command not found"

Ensure the toolchain is in your PATH:

```bash
export PATH="/opt/homebrew/bin:$PATH"
# or for Intel Mac:
export PATH="/usr/local/bin:$PATH"
```

#### "No KitProg3 detected"

1. Check USB connection
2. Install Infineon USB drivers (usually automatic on macOS)
3. Try a different USB port/cable

#### "Failed to initialize camera"

The camera driver is a placeholder. Implement the actual driver for your camera module in `src/camera_driver.c`.

#### "Model inference failed"

1. Ensure model file is correctly converted
2. Check tensor arena size in `face_detector.c`
3. Verify model input/output dimensions match config

### Memory Usage

Typical memory footprint:

| Region | Usage | Available |
|--------|-------|-----------|
| Flash | ~300 KB | 2 MB |
| RAM | ~200 KB | 1 MB |
| Tensor Arena | 128 KB | (in RAM) |

### Performance

Expected performance on CY8C624ABZI-S2D44:

| Metric | Value |
|--------|-------|
| Face Detection | ~50-100 ms/frame |
| EAR Calculation | <1 ms |
| Total Pipeline | ~60-120 ms/frame |
| Frame Rate | 8-15 FPS |

### License

This project is for educational and research purposes. See the main project LICENSE for details.

### References

- [Infineon PSoC 6 Documentation](https://www.infineon.com/cms/en/product/microcontroller/32-bit-psoc-arm-cortex-microcontroller/psoc-6-32-bit-arm-cortex-m4-mcu/)
- [ModusToolbox User Guide](https://www.infineon.com/dgdl/Infineon-ModusToolbox_3.2_User_Guide-UserManual-v01_00-EN.pdf)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [MediaPipe Face Detection](https://developers.google.com/mediapipe/solutions/vision/face_detector)
- [Eye Aspect Ratio Paper](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
