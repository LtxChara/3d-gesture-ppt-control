import numpy as np
import open3d as o3d
import os
import glob
import argparse
import matplotlib.pyplot as plt
import cv2 as cv

# 简单的点云生成函数（复用你的核心逻辑）
def create_pcd(rgb, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    # 降采样一下，否则显示太慢
    factor = 2 
    depth = depth[::factor, ::factor]
    rgb = rgb[::factor, ::factor]
    
    v, u = np.indices(depth.shape)
    v = v * factor
    u = u * factor
    
    Z = depth.astype(np.float32) / 1000.0
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    
    valid = (Z > 0.1) & (Z < 2.0) # 仅显示 0.1m 到 2m 范围内
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid.reshape(-1)])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid.reshape(-1)])
    
    # 翻转一下让朝向正确
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def show_stats(root="data"):
    print(f"{'TYPE':<10} | {'LABEL':<20} | {'COUNT':<10} | {'NOTE'}")
    print("-" * 60)
    
    for dtype in ['raw_static', 'raw_dynamic']:
        base = os.path.join(root, dtype)
        if not os.path.exists(base): continue
        
        dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        for d in dirs:
            files = glob.glob(os.path.join(base, d, "*.npz"))
            count = len(files)
            
            # 简单检查第一个文件
            note = "OK"
            if count > 0:
                try:
                    data = np.load(files[0])
                    if dtype == 'raw_dynamic':
                         note = f"Frames ~{data['depth'].shape[0]}"
                except:
                    note = "Read Error"
            
            print(f"{dtype:<10} | {d:<20} | {count:<10} | {note}")

def view_random_sample(root="data"):
    # 查找所有 npz
    all_files = glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)
    if not all_files:
        print("没有找到数据文件哦！")
        return

    # 随机挑一个
    target = np.random.choice(all_files)
    print(f"正在查看样本: {target}")
    
    data = np.load(target)
    rgb = data['rgb']
    depth = data['depth']
    intrinsics = data['intrinsics']
    
    # 如果是动态序列，取中间一帧查看
    if len(rgb.shape) == 4: # (T, H, W, 3)
        mid = rgb.shape[0] // 2
        print(f"这是一个动态序列，共 {rgb.shape[0]} 帧，展示第 {mid} 帧。")
        rgb = rgb[mid]
        depth = depth[mid]
    
    # 2D 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("RGB")
    plt.imshow(cv.cvtColor(rgb, cv.COLOR_BGR2RGB)) # 你的采集存的是 BGR
    plt.subplot(1, 2, 2)
    plt.title("Depth (Colorized)")
    plt.imshow(depth, cmap='jet')
    plt.show()
    
    # 3D 显示
    print("正在生成点云预览...")
    # 这里需要把 BGR 转回 RGB 供 Open3D 使用
    pcd = create_pcd(cv.cvtColor(rgb, cv.COLOR_BGR2RGB), depth, intrinsics)
    o3d.visualization.draw_geometries([pcd], window_name="Check Sample")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', action='store_true', help='Show data statistics')
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    else:
        view_random_sample()