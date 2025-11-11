#!/usr/bin/env python3
"""
Phase 2: LAB 色彩空間處理
目標: 在 LAB 空間做 ROP，只處理 L 通道，保持色彩不變
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Phase 2: LAB 空間的 ROP 除霧
# ============================================================

class ROPDehazeLAB:
    """在 LAB 色彩空間處理的 ROP 除霧"""
    
    def __init__(self, omega=0.90, patch_size=15):
        self.omega = omega
        self.patch_size = patch_size
        
        print(f"✓ ROP 初始化 (LAB 模式)")
        print(f"  Omega: {omega}")
        print(f"  Patch Size: {patch_size}")
    
    def set_params(self, omega=None, patch_size=None):
        """更新參數"""
        if omega is not None:
            self.omega = np.clip(omega, 0.8, 0.98)
        if patch_size is not None:
            self.patch_size = int(np.clip(patch_size, 7, 31))
            if self.patch_size % 2 == 0:
                self.patch_size += 1
    
    def process(self, bgr_image):
        """
        處理 BGR 影像
        
        流程:
        1. BGR → RGB → LAB
        2. 分離 L, a, b 通道
        3. 只處理 L 通道 (除霧)
        4. 合併 [L_enhanced, a, b]
        5. LAB → RGB → BGR
        """
        # BGR → RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # RGB → LAB (使用 OpenCV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        
        # 分離通道
        L, a, b = cv2.split(lab)
        
        # 只處理 L 通道
        L_enhanced = self._enhance_L_channel(L)
        
        # 合併回去 (a, b 保持不變)
        lab_enhanced = cv2.merge([L_enhanced, a, b])
        
        # LAB → RGB
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # RGB → BGR
        bgr_enhanced = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)
        
        return bgr_enhanced
    
    def _enhance_L_channel(self, L):
        """
        增強 L 通道 (亮度)
        
        L 通道範圍: 0-255 (OpenCV 的 LAB 格式)
        實際代表: 0-100 (CIE LAB 標準)
        """
        # 正規化到 [0, 1]
        L_norm = L.astype(np.float32) / 255.0
        
        # 1. Dark Channel
        dark_channel = self._get_dark_channel(L_norm)
        
        # 2. 估計大氣光
        A = self._estimate_atmospheric_light(L_norm, dark_channel)
        
        # 3. 估計透射率
        t = self._estimate_transmission(L_norm, A, dark_channel)
        
        # 4. 細化透射率 (使用 guided filter)
        try:
            t_refined = cv2.ximgproc.guidedFilter(
                guide=(L_norm * 255).astype(np.uint8),
                src=(t * 255).astype(np.uint8),
                radius=60,
                eps=0.001
            ).astype(np.float32) / 255.0
        except:
            # 如果沒有 ximgproc，用簡單的 blur
            t_refined = cv2.GaussianBlur(t, (61, 61), 15)
        
        # 5. 恢復場景輻射
        J = self._recover_scene_radiance(L_norm, A, t_refined)
        
        # 6. Gamma 校正
        J = self._gamma_correction(J)
        
        # 轉回 uint8
        L_enhanced = (np.clip(J, 0, 1) * 255).astype(np.uint8)
        
        return L_enhanced
    
    def _get_dark_channel(self, img):
        """計算 dark channel (對單通道)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                          (self.patch_size, self.patch_size))
        return cv2.erode(img, kernel)
    
    def _estimate_atmospheric_light(self, img, dark_channel):
        """估計大氣光"""
        h, w = img.shape
        num_pixels = h * w
        num_brightest = max(int(num_pixels * 0.001), 1)
        
        # 找最暗的區域對應的最亮像素
        dark_vec = dark_channel.reshape(num_pixels)
        img_vec = img.reshape(num_pixels)
        
        indices = np.argsort(dark_vec)[-num_brightest:]
        brightest_pixels = img_vec[indices]
        
        return np.max(brightest_pixels)
    
    def _estimate_transmission(self, img, A, dark_channel):
        """估計透射率"""
        if A < 1e-6:
            return np.ones_like(img)
        
        norm_img = img / A
        dark_norm = self._get_dark_channel(norm_img)
        t = 1 - self.omega * dark_norm
        
        return t
    
    def _recover_scene_radiance(self, img, A, t):
        """恢復場景輻射"""
        t0 = 0.1  # 最小透射率，避免除以零
        t = np.maximum(t, t0)
        
        J = (img - A) / t + A
        return J
    
    def _gamma_correction(self, img):
        """Gamma 校正 (5/6 ≈ 0.833)"""
        return np.power(img, 5.0/6.0)


# ============================================================
# Phase 2: 主程式
# ============================================================

class Phase2System:
    """Phase 2 - LAB 色彩空間系統"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        print("\n" + "="*60)
        print("  Phase 2: LAB 色彩空間處理")
        print("="*60)
        
        # 初始化 Camera
        print("\n初始化 Camera...")
        pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
        "videoconvert ! "
        "appsink drop=true"
          )
        self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.camera.isOpened():
            raise RuntimeError("無法開啟 Camera")
        
        ret, test_frame = self.camera.read()
        if not ret:
            raise RuntimeError("無法讀取影像")
        
        print(f"✓ Camera 初始化成功")
        print(f"  解析度: {test_frame.shape[1]}×{test_frame.shape[0]}")
        
        # 初始化 ROP (LAB 模式)
        print("\n初始化 ROP 演算法 (LAB 模式)...")
        self.rop = ROPDehazeLAB(omega=0.90, patch_size=15)
        print("✓ ROP 初始化完成")
        
        # 輸出目錄
        self.output_dir = Path("./phase2_output")
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ 輸出目錄: {self.output_dir}")
        
        # 統計
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        self.rgb_to_lab_time = 0
        self.rop_time = 0
        self.lab_to_rgb_time = 0
        
        print("\n" + "="*60)
        print("系統準備完成！")
        print("="*60)
    
    def run(self):
        """主迴圈"""
        print("\n控制說明:")
        print("  W/S     - 調整 Omega (↑/↓)")
        print("  A/D     - 調整 Patch Size (↓/↑)")
        print("  M       - 切換模式 (Original/ROP)")
        print("  空白鍵   - 儲存當前影像")
        print("  Q       - 退出")
        print("="*60 + "\n")
        
        # 參數
        omega = 0.90
        patch_size = 15
        mode = 'rop'
        
        # FPS 計算
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                # 讀取影像
                ret, frame = self.camera.read()
                if not ret:
                    print("無法讀取影像")
                    break
                
                # 讀取鍵盤
                key = cv2.waitKey(1) & 0xFF
                
                # 更新參數
                if key == ord('w'):
                    omega = min(0.98, omega + 0.01)
                    print(f"Omega: {omega:.2f}")
                elif key == ord('s'):
                    omega = max(0.80, omega - 0.01)
                    print(f"Omega: {omega:.2f}")
                elif key == ord('a'):
                    patch_size = max(7, patch_size - 2)
                    print(f"Patch Size: {patch_size}")
                elif key == ord('d'):
                    patch_size = min(31, patch_size + 2)
                    print(f"Patch Size: {patch_size}")
                elif key == ord('m'):
                    mode = 'rop' if mode == 'original' else 'original'
                    print(f"模式: {mode.upper()}")
                elif key == ord(' '):
                    self._save_image(frame, omega, patch_size, mode)
                elif key == ord('q'):
                    break
                
                # 更新 ROP 參數
                self.rop.set_params(omega, patch_size)
                
                # 處理影像 (並測量時間)
                start_time = time.time()
                if mode == 'rop':
                    frame_out = self.rop.process(frame)
                else:
                    frame_out = frame
                self.processing_time = (time.time() - start_time) * 1000
                
                # 繪製資訊
                frame_display = self._draw_info(frame_out, omega, patch_size, mode)
                
                # 顯示
                cv2.imshow('Phase 2 - LAB Color Space ROP', frame_display)
                
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
    
    def _draw_info(self, frame, omega, patch_size, mode):
        """繪製資訊"""
        info_frame = frame.copy()
        
        # 半透明背景
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, info_frame, 0.4, 0, info_frame)
        
        # 文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 30
        
        cv2.putText(info_frame, f"FPS: {self.fps:.1f}", 
                   (20, y), font, 0.6, (0, 255, 0), 2)
        y += 25
        
        cv2.putText(info_frame, f"Processing: {self.processing_time:.1f}ms", 
                   (20, y), font, 0.6, (0, 255, 0), 2)
        y += 25
        
        cv2.putText(info_frame, f"Omega: {omega:.2f}", 
                   (20, y), font, 0.6, (255, 255, 0), 2)
        y += 25
        
        cv2.putText(info_frame, f"Patch: {patch_size}", 
                   (20, y), font, 0.6, (255, 255, 0), 2)
        y += 25
        
        mode_text = "ROP (LAB)" if mode == 'rop' else "ORIGINAL"
        color = (0, 255, 255) if mode == 'rop' else (255, 0, 0)
        cv2.putText(info_frame, f"Mode: {mode_text}", 
                   (20, y), font, 0.6, color, 2)
        
        return info_frame
    
    def _save_image(self, frame, omega, patch_size, mode):
        """儲存影像 (包含對比)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 儲存原始
        original_path = self.output_dir / f"original_{timestamp}.png"
        cv2.imwrite(str(original_path), frame)
        
        if mode == 'rop':
            # 處理
            enhanced = self.rop.process(frame)
            
            # 儲存增強
            enhanced_path = self.output_dir / f"enhanced_LAB_{timestamp}_w{omega:.2f}_p{patch_size}.png"
            cv2.imwrite(str(enhanced_path), enhanced)
            
            # 對比圖 (左右 + 標籤)
            h, w = frame.shape[:2]
            comparison = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)
            comparison[40:, :w] = frame
            comparison[40:, w:] = enhanced
            
            # 標籤
            cv2.putText(comparison, "Original", (20, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(comparison, "ROP Enhanced (LAB)", (w + 20, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            comparison_path = self.output_dir / f"comparison_LAB_{timestamp}.png"
            cv2.imwrite(str(comparison_path), comparison)
            
            print(f"\n✓ 影像已儲存 (LAB 模式):")
            print(f"  {original_path.name}")
            print(f"  {enhanced_path.name}")
            print(f"  {comparison_path.name}\n")
        else:
            print(f"\n✓ 原始影像已儲存: {original_path.name}\n")
    
    def _cleanup(self):
        """清理"""
        print("\n" + "="*60)
        print("Phase 2 統計:")
        print(f"  總處理幀數: {self.frame_count}")
        print(f"  平均 FPS: {self.fps:.1f}")
        print(f"  平均處理時間: {self.processing_time:.1f}ms")
        print("="*60)
        
        self.camera.release()
        cv2.destroyAllWindows()
        print("\n✓ 清理完成")


# ============================================================
# 主程式
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2 - LAB 色彩空間 ROP 系統')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640, help='寬度')
    parser.add_argument('--height', type=int, default=480, help='高度')
    
    args = parser.parse_args()
    
    try:
        system = Phase2System(
            camera_id=args.camera,
            width=args.width,
            height=args.height
        )
        system.run()
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
