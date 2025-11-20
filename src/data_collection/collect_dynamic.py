#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import cv2 as cv

# === 可配置更多标签 ===
SAVE_ROOT = "data/raw_dynamic"
LABELS = {
    ord('1'): "1_WaveLeft",
    ord('2'): "2_WaveRight",
    ord('3'): "3_Push",
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if not cap.isOpened(): sys.exit("Fail to open Camera")
    
    # 获取相机内参
    fx = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_FX)
    fy = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_FY)
    cx = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_CX)
    cy = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_CY)
    intrinsics = np.array([fx, fy, cx, cy])

    current_label_key = ord('1')
    label_name = LABELS[current_label_key]
    ensure_dir(os.path.join(SAVE_ROOT, label_name))

    is_recording = False
    frame_buffer_rgb = []
    frame_buffer_depth = []
    
    print("动态采集模式")
    print("按 'r' 开始/停止录制, '1'-'3' 切换标签")

    while True:
        if not cap.grab(): continue
        ret_bgr, bgr = cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
        ret_depth, depth = cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)

        if ret_bgr and ret_depth:
            display_img = bgr.copy()
            
            # 录制状态逻辑
            if is_recording:
                frame_buffer_rgb.append(bgr)
                frame_buffer_depth.append(depth)
                # 录制中显示红点
                cv.circle(display_img, (30, 30), 10, (0, 0, 255), -1)
                cv.putText(display_img, f"REC: {len(frame_buffer_rgb)} frames", (50, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv.putText(display_img, f"Label: {label_name}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv.putText(display_img, "Press 'r' to toggle record", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv.imshow("Dynamic Collector", display_img)
            key = cv.waitKey(1)

            # 切换标签
            if key in LABELS and not is_recording:
                current_label_key = key
                label_name = LABELS[key]
                ensure_dir(os.path.join(SAVE_ROOT, label_name))
                print(f"切换动作: {label_name}")

            # 录制控制
            if key == ord('r'):
                if not is_recording:
                    # 开始录制
                    is_recording = True
                    frame_buffer_rgb = []
                    frame_buffer_depth = []
                    print("开始录制...")
                else:
                    # 停止录制并保存
                    is_recording = False
                    if len(frame_buffer_rgb) > 10: # 过滤太短的误触
                        timestamp = int(time.time() * 1000)
                        save_path = os.path.join(SAVE_ROOT, label_name, f"{timestamp}.npz")
                        
                        # 堆叠数据：(T, H, W, C)
                        np_rgb = np.stack(frame_buffer_rgb)
                        np_depth = np.stack(frame_buffer_depth)
                        
                        np.savez_compressed(save_path, 
                                            rgb=np_rgb, 
                                            depth=np_depth, 
                                            intrinsics=intrinsics,
                                            label=label_name)
                        print(f"序列已保存 ({len(np_rgb)} 帧): {save_path}")
                    else:
                        print("序列过短。")

            if key == ord('q') or key == 27:
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()