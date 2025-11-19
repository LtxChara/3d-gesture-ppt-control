#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import cv2 as cv

# === 可配置更多标签 ===
SAVE_ROOT = "data/raw_static"
# 定义标签：
LABELS = {
    ord('1'): (1, "1_OpenHand"),
    ord('2'): (2, "2_Fist"),
    ord('3'): (3, "3_OK"),
    ord('4'): (4, "4_Yeah"),
    ord('0'): (0, "0_Background")
}
# ===============

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # 初始化相机
    cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if not cap.isOpened():
        sys.exit("Fail to open Orbbec camera.")

    # 获取相机内参
    fx = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_FX)
    fy = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_FY)
    cx = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_CX)
    cy = cap.get(cv.CAP_PROP_OBSENSOR_INTRINSIC_CY)
    intrinsics = np.array([fx, fy, cx, cy])
    print(f"Camera Intrinsics: {intrinsics}")

    # 默认标签
    current_label_key = ord('1')
    label_id, label_name = LABELS[current_label_key]
    
    # 确保目录存在
    ensure_dir(os.path.join(SAVE_ROOT, label_name))

    print("静态采集开始！")
    print("按 '1'-'4' 切换标签, 按 's' 保存, 'q' 退出")

    count = 0

    while True:
        if not cap.grab():
            continue

        ret_bgr, bgr = cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
        ret_depth, depth = cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)

        if ret_bgr and ret_depth:
            # 画面处理
            display_img = bgr.copy()
            
            # 深度图伪彩色显示
            d8 = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            d_color = cv.applyColorMap(d8, cv.COLORMAP_JET)

            # UI 信息显示
            info_text = f"Label: {label_name} | Count: {count}"
            cv.putText(display_img, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.putText(display_img, "Press 's' to save", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # 拼接显示
            cv.imshow("Collector", np.hstack((display_img, d_color)))

            key = cv.waitKey(1)

            # 切换标签
            if key in LABELS:
                current_label_key = key
                label_id, label_name = LABELS[key]
                ensure_dir(os.path.join(SAVE_ROOT, label_name))
                print(f"切换标签为: {label_name}")

            # 保存数据
            if key == ord('s'):
                timestamp = int(time.time() * 1000)
                save_path = os.path.join(SAVE_ROOT, label_name, f"{timestamp}.npz")
                
                # 保存压缩格式
                np.savez_compressed(save_path, 
                                    rgb=bgr, 
                                    depth=depth, 
                                    intrinsics=intrinsics,
                                    label=label_name)
                print(f"已保存: {save_path}")
                count += 1
                # 简单的闪烁反馈
                cv.rectangle(display_img, (0,0), (display_img.shape[1], display_img.shape[0]), (255,255,255), 10)
                cv.imshow("Collector", np.hstack((display_img, d_color)))
                cv.waitKey(50)

            if key == ord('q') or key == 27:
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()