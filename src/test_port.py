#!/usr/bin/env python3
"""
Arduino Serial �q�T����
�q Arduino Ū���n��ƭ�
"""

import serial
import time
import sys

print("="*60)
print("  Arduino Serial Port Test")
print("="*60)

# Searching for Arduino
print("\nSearching for Arduino Serial Port...")
import glob
ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')

if not ports:
    print("? No Arduino found")
    print("  Suggestions:")
    print("  1. Ensure Arduino is properly connected to Raspberry Pi (USB)")
    print("  2. Ensure Arduino is powered on")
    sys.exit(1)

print(f"? Found {len(ports)} Serial Port(s):")
for i, port in enumerate(ports):
    print(f"  {i+1}. {port}")

if len(ports) == 1:
    selected_port = ports[0]
    print(f"\nSelected Port: {selected_port}")
else:
    choice = int(input(f"\nSelect Port (1-{len(ports)}): ")) - 1
    selected_port = ports[choice]

# Connecting to Arduino
print(f"\nConnecting to {selected_port}...")
try:
    ser = serial.Serial(
        port=selected_port,
        baudrate=115200,
        timeout=1
    )
    
    # initial Arduino delay
    print("initial Arduino delay...")
    time.sleep(2)
    
    # buffer
    ser.reset_input_buffer()
    
    print("? Serial connection established")
    print("\nReading sensor data (press Ctrl+C to stop):")
    print("-"*60)
    print(f"{'X':<10} {'Y':<10} {'Button':<10} {'Omega':<10} {'Patch':<10}")
    print("-"*60)
    
    try:
        while True:
            # Read line
            line = ser.readline().decode('utf-8').strip()
            
            # Skip invalid lines
            if not line or ',' not in line:
                continue
            
            try:
                # Parse: "X,Y,B"
                parts = line.split(',')
                if len(parts) != 3:
                    continue
                
                x = int(parts[0])
                y = int(parts[1])
                button = int(parts[2])
                
                # Calculate omega
                # X (0-1023) to omega (0.80-0.98)
                omega = 0.80 + (x / 1023.0) * 0.18
                
                # Y (0-1023) to patch_size (7-31)
                patch_raw = 7 + int((y / 1023.0) * 24)
                patch_size = patch_raw if patch_raw % 2 == 1 else patch_raw + 1
                
                # Button
                button_text = "PRESSED" if button else "Released"
                print(f"\r{x:<10} {y:<10} {button_text:<10} {omega:<10.2f} {patch_size:<10}", end='')
                sys.stdout.flush()
            
            except ValueError:
                continue
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received")
        ser.close()

except serial.SerialException as e:
    print(f"? Serial error: {e}")
    print("\nSuggestions:")
    print("  1. Ensure Arduino is properly connected")
    print("  2. Check Serial Port permissions")
    print("  3. Restart the system to apply Serial Port changes")
    
    print("\nPermissions fix:")
    print(f"  sudo chmod 666 {selected_port}")
    print(f"  or")
    print(f"  sudo usermod -a -G dialout $USER")
    print(f"  (Log out and log back in)")

except Exception as e:
    print(f"? Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)