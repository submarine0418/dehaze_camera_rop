#!/usr/bin/env python3
"""
Phase 1: 基礎驗證版本
目標: 驗證整個流程可以運作（使用 OpenCV，不用 kernel driver）
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2



# ============================================================
# Phase 1: 簡化的 ROP 除霧
# ============================================================

class SimpleROPDehaze:
    """簡化版 ROP 除霧（先驗證流程）"""

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
        """
        處理 BGR 影像（OpenCV 格式）

        Phase 1: 先在 RGB 空間處理（簡單版本）
        Phase 2 再改成 LAB 空間
        """
        # BGR → RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # 正規化到 [0, 1]
        img = rgb.astype(np.float32) / 255.0
        
        # 簡化版除霧
        # 1. Dark Channel
        dark = self._dark_channel(img)
        
        # 2. 大氣光估計
        A = self._estimate_atmospheric_light(img, dark)
        
        # 3. 透射率
        t = self._estimate_transmission(img, A, dark)
        
        # 4. 恢復
        J = self._recover(img, A, t)
        
        # 轉回 BGR
        result = (np.clip(J, 0, 1) * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result_bgr
    
    def _dark_channel(self, img):
        """Dark channel"""
        # 取 RGB 三通道最小值
        min_channel = np.min(img, axis=2)
        
        # 最小值濾波
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                          (self.patch_size, self.patch_size))
        dark = cv2.erode(min_channel, kernel)
        return dark
    
    def _estimate_atmospheric_light(self, img, dark):
        """估計大氣光（簡化版）"""
        h, w = dark.shape
        num_pixels = h * w
        num_brightest = int(num_pixels * 0.001)
        
        # 找最亮的 0.1% 像素
        dark_vec = dark.reshape(num_pixels)
        indices = np.argsort(dark_vec)[-num_brightest:]
        
        # 在原圖中找這些位置的最大值
        img_vec = img.reshape(num_pixels, 3)
        brightest = img_vec[indices].max(axis=0)
        
        return brightest
    
    def _estimate_transmission(self, img, A, dark):
        """估計透射率"""
        # 正規化
        norm_img = np.zeros_like(img)
        for i in range(3):
            norm_img[:, :, i] = img[:, :, i] / (A[i] + 1e-6)
        
        # Dark channel
        dark_norm = self._dark_channel(norm_img)
        
        # 透射率
        t = 1 - self.omega * dark_norm
        return t
    
    def _recover(self, img, A, t):
        """恢復場景輻射"""
        t0 = 0.1
        t = np.maximum(t, t0)
        
        J = np.zeros_like(img)
        for i in range(3):
            J[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]
        
        return J


# ============================================================
# Phase 1: 主程式
# ============================================================

class Phase1System:
    """Phase 1 - 基礎驗證系統"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        print("\n" + "="*60)
        print("  Phase 1: 基礎驗證")
        print("="*60)
        
        # 初始化 Camera
        print("\n初始化 Camera...")
        pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
        "videoconvert ! "
        "appsink drop=true"
          )
        self.camera =  cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.camera.isOpened():
            raise RuntimeError("無法開啟 Camera")
        
        # 測試讀取
        ret, test_frame = self.camera.read()
        if not ret:
            raise RuntimeError("無法讀取影像")
        
        print(f"✓ Camera 初始化成功")
        print(f"  解析度: {test_frame.shape[1]}×{test_frame.shape[0]}")
        
        # 初始化 ROP
        print("\n初始化 ROP 演算法...")
        self.rop = SimpleROPDehaze(omega=0.90, patch_size=15)
        print("✓ ROP 初始化完成")
        
        # 輸出目錄
        self.output_dir = Path("./phase1_output")
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ 輸出目錄: {self.output_dir}")
        
        # 統計
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        
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
        mode = 'rop'  # 'original' or 'rop'
        
        # FPS 計算
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                # 讀取影像
                ret, frame = self.camera.read()
                if not ret or frame is None:
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
                
                # 處理影像
                start_time = time.time()
                if mode == 'rop':
                    frame_out = self.rop.process(frame)
                else:
                    frame_out = frame
                self.processing_time = (time.time() - start_time) * 1000
                
                # 繪製資訊
                frame_display = self._draw_info(frame_out, omega, patch_size, mode)
                
                # 顯示
                cv2.imshow('Phase 1 - Basic ROP System', frame_display)
                
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
        cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
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
        
        mode_text = "ROP" if mode == 'rop' else "ORIGINAL"
        color = (0, 255, 255) if mode == 'rop' else (255, 0, 0)
        cv2.putText(info_frame, f"Mode: {mode_text}", 
                   (20, y), font, 0.6, color, 2)
        
        return info_frame
    
    def _save_image(self, frame, omega, patch_size, mode):
        """儲存影像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 儲存原始
        original_path = self.output_dir / f"original_{timestamp}.png"
        cv2.imwrite(str(original_path), frame)
        
        # 儲存處理後
        if mode == 'rop':
            enhanced = self.rop.process(frame)
            enhanced_path = self.output_dir / f"enhanced_{timestamp}_w{omega:.2f}_p{patch_size}.png"
            cv2.imwrite(str(enhanced_path), enhanced)
            
            # 對比圖
            comparison = np.hstack([frame, enhanced])
            comparison_path = self.output_dir / f"comparison_{timestamp}.png"
            cv2.imwrite(str(comparison_path), comparison)
            
            print(f"\n✓ 影像已儲存:")
            print(f"  {original_path.name}")
            print(f"  {enhanced_path.name}")
            print(f"  {comparison_path.name}\n")
        else:
            print(f"\n✓ 原始影像已儲存: {original_path.name}\n")
    
    def _cleanup(self):
        """清理"""
        print("\n清理資源...")
        print(f"總處理幀數: {self.frame_count}")
        self.camera.release()
        cv2.destroyAllWindows()
        print("✓ 完成")


# ============================================================
# 主程式
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 1 - 基礎 ROP 系統')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640, help='寬度')
    parser.add_argument('--height', type=int, default=480, help='高度')
    
    args = parser.parse_args()
    
    try:
        system = Phase1System(
            camera_id=args.camera,
            width=args.width,
            height=args.height
        )
        system.run()
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
