#!/usr/bin/env python3
"""
Phase 3: DCP Dehazing with Joystick Control
- Arduino WASD One-Shot → Adjust Omega & Patch Size
- M key: Save original + enhanced + comparison
- Every trigger → New log line
- dir_map: D=right, S=down, A=left, W=up
"""

import cv2
import serial
import time
import glob
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# === Output directory ===
OUTPUT_DIR = Path("./phase3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# === Optional: uinput ===
try:
    import uinput
    UINPUT_ENABLED = True
    device = uinput.Device([
        uinput.KEY_W, uinput.KEY_A, uinput.KEY_S, uinput.KEY_D
    ], name="DCP-Joystick")
    print("uinput enabled: keys emulated")
except Exception as e:
    UINPUT_ENABLED = False
    print(f"uinput disabled: {e}")

# === Auto-detect Arduino ===
print("Searching for Arduino...")
ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
if not ports:
    print("No Arduino found! Using keyboard fallback.")
    arduino = None
else:
    port = ports[0]
    print(f"Using port: {port}")
    try:
        arduino = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        arduino.reset_input_buffer()
        print("Arduino connected.")
    except Exception as e:
        print(f"Serial error: {e}")
        arduino = None

# === DCP Parameters ===
omega = 0.90
patch_size = 15
omega_step = 0.01
patch_step = 2

# === State tracking ===
active = {'W': False, 'A': False, 'S': False, 'D': False}
dir_map = {'D': 'right', 'S': 'down', 'A': 'left', 'W': 'up'}

# === Header ===
print("\n" + "=" * 70)
print("   Phase 3: DCP Dehazing with Joystick Control")
print("   RIGHT→D: +Patch  DOWN→S: -Patch  LEFT→A: -Omega  UP→W: +Omega")
print("   Press M to SAVE images with current parameters")
print("=" * 70)
print()

# === Camera ===
print("Opening camera with V4L2...")
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! "
    "appsink drop=1"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)




time.sleep(1)
ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("Error: Cannot read from camera! Check /dev/video0")
    sys.exit(1)

print(f"Camera initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")

# === Buffer ===
buffer = ""

# === DCP Functions ===
def dark_channel(img, size):
    return cv2.erode(img, np.ones((size, size), np.uint8))

def estimate_atmosphere(img, dark):
    h, w = img.shape[:2]
    num_pixels = h * w
    num_bright = max(int(num_pixels * 0.001), 1)
    dark_vec = dark.reshape(num_pixels)
    img_vec = img.reshape(num_pixels)
    indices = dark_vec.argsort()[-num_bright:]
    return img_vec[indices].max()

def refine_transmission(t, guide, radius=60, eps=0.001):
    try:
        return cv2.ximgproc.guidedFilter(guide, t, radius, eps)
    except:
        return cv2.GaussianBlur(t, (61, 61), 15)

def dcp_dehaze(img_bgr, omega_val, patch_val):
    img = img_bgr.astype(np.float32) / 255.0
    dark = dark_channel(img, patch_val)
    A = estimate_atmosphere(img, dark)
    t = 1 - omega_val * dark_channel(img / max(A, 1e-3), patch_val)
    t_refined = refine_transmission((t * 255).astype(np.uint8), (img * 255).astype(np.uint8)) / 255.0
    t0 = 0.1
    J = (img - A) / np.maximum(t_refined, t0) + A
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)

# === Save function ===
def save_images(original, enhanced, omega_val, patch_val):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = original.shape[:2]

    # 1. Enhanced image
    enhanced_path = OUTPUT_DIR / f"enhanced_{timestamp}_w{omega_val:.2f}_p{patch_val}.png"
    cv2.imwrite(str(enhanced_path), enhanced)

    # 2. Original image
    original_path = OUTPUT_DIR / f"original_{timestamp}.png"
    cv2.imwrite(str(original_path), original)

    # 3. Comparison image
    comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w+20:] = enhanced
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(comparison, f"DCP w={omega_val:.2f} p={patch_val}", (w+30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    comp_path = OUTPUT_DIR / f"comparison_{timestamp}.png"
    cv2.imwrite(str(comp_path), comparison)

    print(f"  [SAVED] {enhanced_path.name}")

# === Main loop ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === Serial input ===
        if arduino and arduino.in_waiting > 0:
            raw = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
            buffer += raw

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip('\r\n')
                if not line:
                    continue

                if line in ['W', 'A', 'S', 'D']:
                    key = line
                    direction = dir_map[key]

                    if not active[key]:
                        active[key] = True
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] TRIGGER → {direction.upper():>5}  ({key})")

                        if key == 'W': omega = min(0.98, omega + omega_step)
                        elif key == 'A': omega = max(0.80, omega - omega_step)
                        elif key == 'D': patch_size = min(31, patch_size + patch_step); patch_size += patch_size % 2
                        elif key == 'S': patch_size = max(7, patch_size - patch_step); patch_size -= patch_size % 2 and 1 or 0

                        if UINPUT_ENABLED:
                            code = getattr(uinput, f"KEY_{key}")
                            device.emit(code, 1)
                            device.syn()

                else:
                    for k in list(active.keys()):
                        if active[k]:
                            active[k] = False
                            if UINPUT_ENABLED:
                                code = getattr(uinput, f"KEY_{k}")
                                device.emit(code, 0)
                                device.syn()

        # === Apply DCP ===
        enhanced = dcp_dehaze(frame, omega, patch_size)

        # === Overlay ===
        info = enhanced.copy()
        cv2.putText(info, f"Omega: {omega:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info, f"Patch: {patch_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info, "Press M to SAVE", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # === Display ===
        cv2.imshow("Phase 3 - DCP Dehazing", info)

        # === Keyboard input ===
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):  # SAVE with M key
            save_images(frame, enhanced, omega, patch_size)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    print("Cleanup complete.")