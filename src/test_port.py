#!/usr/bin/env python3
"""
HW-504 Joystick WASD Event Logger
- Every input �� NEW LINE printed
- No overwrite, no deduplication
- English comments only
"""

import serial
import time
import glob
import sys

# === Optional: uinput keyboard emulation ===
try:
    import uinput
    UINPUT_ENABLED = True
    device = uinput.Device([
        uinput.KEY_W, uinput.KEY_A, uinput.KEY_S, uinput.KEY_D
    ], name="WASD-Joystick")
    print("uinput enabled: keys will be emulated")
except Exception as e:
    UINPUT_ENABLED = False
    print(f"uinput disabled: {e}")

# === Auto-detect serial port ===
print("Searching for Arduino...")
ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
if not ports:
    print("No Arduino found! Check connection.")
    sys.exit(1)

port = ports[0]
print(f"Using port: {port}")

# === Open serial ===
try:
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print("Arduino connected. Waiting for WASD events...\n")
except Exception as e:
    print(f"Serial error: {e}")
    sys.exit(1)

# === Direction map ===
dir_map = {'D': 'UP', 'S': 'LEFT', 'A': 'DOWN', 'W': 'RIGHT'}

# === Header ===
print("=" * 60)
print("   WASD Event Logger (Ctrl+C to exit)")
print("   Push UP��D  LEFT��S  DOWN��A  RIGHT��W")
print("=" * 60)
print()  # Empty line for separation

# === Buffer for partial reads ===
buffer = ""

try:
    while True:
        if ser.in_waiting > 0:
            raw = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            buffer += raw

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip('\r\n')  # Clean \r and \n

                if not line:
                    continue

                if line in ['W', 'A', 'S', 'D']:
                    direction = dir_map[line]
                    # Print every time input is received
                    print(f"  INPUT �� {direction:>5}  ({line})")

                    if UINPUT_ENABLED:
                        code = getattr(uinput, f"KEY_{line}")
                        # Press and release instantly to simulate a tap
                        device.emit(code, 1)
                        device.emit(code, 0)
                        device.syn()

        else:
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
finally:
    ser.close()
    print("Serial closed.")
