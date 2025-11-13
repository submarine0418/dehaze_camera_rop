#!/usr/bin/env python3
"""
Phase 3: 搖桿整合版
使用 Arduino 讀取 HW-504 搖桿，控制 ROP 參數
"""

import cv2
import numpy as np
import serial
import time
import glob
from datetime import datetime
from pathlib import Path

# ============================================================
# Arduino 搖桿控制器
# ============================================================

class ArduinoJoystick:
    """Arduino 搖桿控制器 (透過 Serial)"""
    
    def __init__(self, port=None, baudrate=115200):
        """初始化 Arduino Serial 連接"""
        self.ser = None
        self.connected = False
        
        # 自動搜尋 Arduino
        if port is None:
            ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            if not ports:
                print("⚠ 找不到 Arduino，使用鍵盤控制")
                return
            port = ports[0]
            print(f"✓ 自動找到 Arduino: {port}")
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2)  # 等待 Arduino 重啟
            self.ser.reset_input_buffer()
            self.connected = True
            print(f"✓ Arduino 連接成功: {port}")
        except Exception as e:
            print(f"⚠ Arduino 連接失敗: {e}")
            print("  將使用鍵盤控制")
    
    def read(self):
        """
        讀取搖桿數值
        
        返回:
            x: 0-1023
            y: 0-1023
            button: 0/1 (1=按下)
        """
        if not self.connected:
            return None, None, 0
        
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        x = int(parts[0])
                        y = int(parts[1])
                        button = int(parts[2])
                        return x, y, button
        except:
            pass
        
        return None, None, 0
    
    def close(self):
        """關閉連接"""
        if self.ser:
            self.ser.close()


# ============================================================
# ROP 除霧 (LAB 模式)
# ============================================================

class ROPDehazeLAB:
    """在 LAB 色彩空間處理的 ROP 除霧"""
    
    def __init__(self, omega=0.90, patch_size=15):
        self.omega = omega
        self.patch_size = patch_size
    
    def set_params(self, omega=None, patch_size=None):
        """更新參數"""
        if omega is not None:
            self.omega = np.clip(omega, 0.8, 0.98)
        if patch_size is not None:
            self.patch_size = int(np.clip(patch_size, 7, 31))
            if self.patch_size % 2 == 0:
                self.patch_size += 1
    
    def process(self, bgr_image):
        """處理影像"""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)
        L_enhanced = self._enhance_L_channel(L)
        lab_enhanced = cv2.merge([L_enhanced, a, b])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)
    
    def _enhance_L_channel(self, L):
        """增強 L 通道"""
        L_norm = L.astype(np.float32) / 255.0
        dark_channel = self._get_dark_channel(L_norm)
        A = self._estimate_atmospheric_light(L_norm, dark_channel)
        t = self._estimate_transmission(L_norm, A, dark_channel)
        
        try:
            t_refined = cv2.ximgproc.guidedFilter(
                guide=(L_norm * 255).astype(np.uint8),
                src=(t * 255).astype(np.uint8),
                radius=60, eps=0.001
            ).astype(np.float32) / 255.0
        except:
            t_refined = cv2.GaussianBlur(t, (61, 61), 15)
        
        J = self._recover_scene_radiance(L_norm, A, t_refined)
        J = np.power(J, 5.0/6.0)
        return (np.clip(J, 0, 1) * 255).astype(np.uint8)
    
    def _get_dark_channel(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                          (self.patch_size, self.patch_size))
        return cv2.erode(img, kernel)
    
    def _estimate_atmospheric_light(self, img, dark_channel):
        h, w = img.shape
        num_pixels = h * w
        num_brightest = max(int(num_pixels * 0.001), 1)
        dark_vec = dark_channel.reshape(num_pixels)
        img_vec = img.reshape(num_pixels)
        indices = np.argsort(dark_vec)[-num_brightest:]
        return np.max(img_vec[indices])
    
    def _estimate_transmission(self, img, A, dark_channel):
        if A < 1e-6:
            return np.ones_like(img)
        norm_img = img / A
        dark_norm = self._get_dark_channel(norm_img)
        return 1 - self.omega * dark_norm
    
    def _recover_scene_radiance(self, img, A, t):
        t0 = 0.1
        t = np.maximum(t, t0)
        return (img - A) / t + A


# ============================================================
# Phase 3 主系統
# ============================================================

class Phase3System:
    """Phase 3 - 搖桿控制系統"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        print("\n" + "="*60)
        print("  Phase 3: 搖桿控制 ROP 系統")
        print("="*60)
        
        # 初始化 Camera
        print("\n初始化 Camera...")
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.camera.isOpened():
            raise RuntimeError("無法開啟 Camera")
        
        ret, test_frame = self.camera.read()
        if not ret:
            raise RuntimeError("無法讀取影像")
        print(f"✓ Camera 初始化成功 ({test_frame.shape[1]}×{test_frame.shape[0]})")
        
        # 初始化搖桿
        print("\n初始化搖桿...")
        self.joystick = ArduinoJoystick()
        
        # 初始化 ROP
        print("\n初始化 ROP 演算法...")
        self.rop = ROPDehazeLAB(omega=0.90, patch_size=15)
        print("✓ ROP 初始化完成")
        
        # 輸出目錄
        self.output_dir = Path("./phase3_output")
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ 輸出目錄: {self.output_dir}")
        
        # 統計
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        
        # 參數
        self.omega = 0.90
        self.patch_size = 15
        self.mode = 'rop'
        
        print("\n" + "="*60)
        print("系統準備完成！")
        print("="*60)
    
    def run(self):
        """主迴圈"""
        print("\n控制說明:")
        if self.joystick.connected:
            print("  搖桿 X 軸  - 調整 Omega")
            print("  搖桿 Y 軸  - 調整 Patch Size")
            print("  搖桿按鈕  - 儲存影像")
        print("  M鍵      - 切換模式")
        print("  空白鍵    - 儲存影像 (鍵盤)")
        print("  Q鍵      - 退出")
        print("="*60 + "\n")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        last_button = 0
        
        try:
            while True:
                # 讀取影像
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # 讀取搖桿
                save_flag = False
                if self.joystick.connected:
                    x, y, button = self.joystick.read()
                    
                    if x is not None and y is not None:
                        # X 軸 → omega (0.80-0.98)
                        self.omega = 0.80 + (x / 1023.0) * 0.18
                        
                        # Y 軸 → patch_size (7-31, 奇數)
                        patch_raw = 7 + int((y / 1023.0) * 24)
                        self.patch_size = patch_raw if patch_raw % 2 == 1 else patch_raw + 1
                    
                    # 按鈕 → 儲存 (偵測按下瞬間)
                    if button == 1 and last_button == 0:
                        save_flag = True
                    last_button = button
                
                # 讀取鍵盤
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m'):
                    self.mode = 'rop' if self.mode == 'original' else 'original'
                    print(f"模式: {self.mode.upper()}")
                elif key == ord(' '):
                    save_flag = True
                elif key == ord('q'):
                    break
                
                # 更新 ROP 參數
                self.rop.set_params(self.omega, self.patch_size)
                
                # 處理影像
                start_time = time.time()
                if self.mode == 'rop':
                    frame_out = self.rop.process(frame)
                else:
                    frame_out = frame
                self.processing_time = (time.time() - start_time) * 1000
                
                # 儲存影像
                if save_flag:
                    self._save_image(frame, frame_out)
                
                # 繪製資訊
                frame_display = self._draw_info(frame_out)
                
                # 顯示
                cv2.imshow('Phase 3 - Joystick Control', frame_display)
                
                # 計算 FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n程式被使用者中斷")
        
        finally:
            self._cleanup()
    
    def _draw_info(self, frame):
        """繪製資訊"""
        info_frame = frame.copy()
        
        # 半透明背景
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, info_frame, 0.4, 0, info_frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        
        # FPS
        cv2.putText(info_frame, f"FPS: {self.fps:.1f}", 
                   (20, y), font, 0.6, (0, 255, 0), 2)
        y += 25
        
        # 處理時間
        cv2.putText(info_frame, f"Processing: {self.processing_time:.1f}ms", 
                   (20, y), font, 0.6, (0, 255, 0), 2)
        y += 25
        
        # 控制方式
        control_text = "Joystick" if self.joystick.connected else "Keyboard"
        color = (0, 255, 255) if self.joystick.connected else (128, 128, 128)
        cv2.putText(info_frame, f"Control: {control_text}", 
                   (20, y), font, 0.6, color, 2)
        y += 25
        
        # Omega
        cv2.putText(info_frame, f"Omega: {self.omega:.2f}", 
                   (20, y), font, 0.6, (255, 255, 0), 2)
        y += 25
        
        # Patch Size
        cv2.putText(info_frame, f"Patch: {self.patch_size}", 
                   (20, y), font, 0.6, (255, 255, 0), 2)
        y += 25
        
        # 模式
        mode_text = "ROP (LAB)" if self.mode == 'rop' else "ORIGINAL"
        color = (0, 255, 255) if self.mode == 'rop' else (255, 0, 0)
        cv2.putText(info_frame, f"Mode: {mode_text}", 
                   (20, y), font, 0.6, color, 2)
        
        return info_frame
    
    def _save_image(self, frame, frame_out):
        """儲存影像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original_path = self.output_dir / f"original_{timestamp}.png"
        cv2.imwrite(str(original_path), frame)
        
        if self.mode == 'rop':
            enhanced_path = self.output_dir / f"enhanced_{timestamp}_w{self.omega:.2f}_p{self.patch_size}.png"
            cv2.imwrite(str(enhanced_path), frame_out)
            
            # 對比圖
            h, w = frame.shape[:2]
            comparison = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)
            comparison[40:, :w] = frame
            comparison[40:, w:] = frame_out
            
            cv2.putText(comparison, "Original", (20, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(comparison, f"ROP (w={self.omega:.2f}, p={self.patch_size})", 
                       (w + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            comparison_path = self.output_dir / f"comparison_{timestamp}.png"
            cv2.imwrite(str(comparison_path), comparison)
            
            print(f"\n✓ 影像已儲存:")
            print(f"  {enhanced_path.name}\n")
        else:
            print(f"\n✓ 原始影像已儲存: {original_path.name}\n")
    
    def _cleanup(self):
        """清理"""
        print("\n" + "="*60)
        print(f"total frame: {self.frame_count}")
        print(f"FPS: {self.fps:.1f}")
        print("="*60)
        
        self.joystick.close()
        self.camera.release()
        cv2.destroyAllWindows()
        print("\n✓ 清理完成")


# ============================================================
# 主程式
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 3 - 搖桿控制 ROP 系統')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640, help='寬度')
    parser.add_argument('--height', type=int, default=480, help='高度')
    
    args = parser.parse_args()
    
    try:
        system = Phase3System(
            camera_id=args.camera,
            width=args.width,
            height=args.height
        )
        system.run()
    except Exception as e:
        print(f"\nerror: {e}")
        import traceback
        traceback.print_exc()
