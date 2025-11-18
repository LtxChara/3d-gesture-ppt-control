### 一、项目目标

*   基于 Gemini 2 / Astra 深度相机，构建一个：
    *   支持 **静态 3D 手势识别** 的 PPT 控制系统；
    *   支持 **基于 3D 轨迹的动态手势识别**（左右挥手翻页）；
    *   框架可扩展，能通过配置映射到 **视频播放器控制** 等其他应用。



### 二、队员分工

1. **组长、撰写 Final report、录制演示视频**：**李天翔**

2. **手势数据采集、标注**：**惠志文**

   - 采集脚本开发
     *   基于 `ptcloud.py` 等代码，实现：
         *   `collect_static.py`：单帧静态手势采集，支持：
             *   RGB + 深度实时显示；
             *   按键切换当前标签（如 `1=OPEN_HAND`, `2=FIST`, …）；
             *   按 `s` 保存一帧为 `data/raw_static/<gesture>/<id>.npz`。
         *   `collect_dynamic.py`：多帧序列采集（用于左右挥手等），保存为 `data/raw_dynamic/<gesture>/<id>.npz`，包含 `T` 帧的深度/点云数据。
   - 数据规范制定与管理
     *   设计统一数据格式（文件命名、目录结构、标签文件）。
     *   编写统计脚本：
         *   统计每类样本数量；
         *   统计样本的深度范围、是否存在缺失帧。
     *   编写 `view_data.py`：
         *   随机展示若干样本的 RGB、深度图和点云可视化，用于排查采集问题。
   - 数据采集组织
     *   每类静态手势样本量：≥200–500（两个人弄）；
     *   动态手势每类序列样本量：≥50–100（两个人弄）；
   
   预期输出：

   *   `collect_static.py`, `collect_dynamic.py`, `view_data.py`;
   *   规范化的数据集目录 `data/raw_static/`, `data/raw_dynamic/`
   
3. **手部分割、点云预处理**：**缪健铭**

   - 手部分割（基于深度）
     *   基于 `segmentation.py` 实现：
         *   深度范围阈值（如 0.3–1.0m）；
         *   最大连通域提取（假设最近的大连通域为手部）；
         *   形态学操作（`erode/dilate`）去噪。
     *   输出：二值 `hand_mask`。
   - 点云生成与归一化
     *   集成 `ptcloud.py` 投影逻辑：
         *   利用相机内参 `fx, fy, cx, cy` 将 `hand_mask` 中像素转为 3D 点；
         *   过滤掉深度为 0 或噪声异常值。
     *   实现预处理函数：
         *   计算质心并居中；
         *   尺度归一（单位球）；
         *   随机/体素下采样到固定点数（如 `N=1024`）。
   - 数据增强
     *   设计并实现：
         *   小幅随机旋转（绕 Z 轴）；
         *   轻微缩放、抖动噪声；
         *   可通过 PyTorch `Dataset` 中 on-the-fly 完成。
   - 离线预处理脚本
     *   编写 `preprocess_static.py`、`preprocess_dynamic.py`：
         *   将 `data/raw_*` 转为 `data/processed_static`, `data/processed_dynamic`；
         *   输出格式为 `N×3` 或 `(T, N, 3)` 的 `npy/npz`。

   预期输出：

   *   `hand_segmentation.py`, `depth_to_pointcloud.py`;
   *   `preprocess_static.py`, `preprocess_dynamic.py`;
   *   `data/processed_static/`, `data/processed_dynamic/` 数据集；
   *   示例点云可视化图若干，用于报告插图。

4. **静态手势识别模型**：**江一航**

   - Dataset 与 DataLoader
     *   编写 `dataset_static.py`：
         *   从 `data/processed_static/` 读取点云 + 标签；
         *   支持 train/val/test 划分（可读取一个 split 配置或按比例随机划分）；
         *   支持调用数据增强模块。
   - PointNet 类模型实现
     *   编写 `models/pointnet_static.py`：
         *   输入 `B×N×3`；
         *   一系列 `Conv1d + BN + ReLU` 提取点特征；
         *   `max pooling` 得到全局特征；
         *   `FC` 分类输出 K 类。
     *   可参考公开 PointNet 实现，但需自己精简重写。
   - 训练与验证脚本
     *   编写 `train_static.py`：
         *   损失函数：`CrossEntropyLoss`；
         *   优化器：`Adam`；
         *   记录训练/验证 accuracy、loss 曲线；
         *   实现 early stopping 或 best model 保存。
     *   编写 `eval_static.py`：
         *   输出混淆矩阵；
         *   统计各类别 precision / recall。

   预期输出：

   *   `dataset_static.py`, `models/pointnet_static.py`;
   *   `train_static.py`, `eval_static.py`;
   *   最佳模型 `static_pointnet.pth`；
   *   静态手势混淆矩阵图、训练曲线图。

5. **动态手势识别模型，手势数据采集**：**方炜轩**

   - 动态数据格式与序列管理
     *   确定动态序列格式：
         *   每个样本 `(T, N, 3)` 点云序列；
         *   或 `(T, 3)` 的质心轨迹 + 少量统计特征；
     *   编写 `dataset_dynamic.py`：
         *   支持加载动态序列数据；
         *   支持 shuffle / batch / collate。
   - 3D 质心轨迹算法
     *   每帧计算手部点云质心 `c_t = (x_t, y_t, z_t)`；
     *   用滑动窗口 `{c_{t-T+1} ... c_t}`：
         *   计算总位移 `Δ = c_t - c_{t-T+1}`；
         *   计算轨迹直线性（例如每帧方向与整体方向的夹角方差）；
     *   设计判定规则：
         *   `Δx > 阈值 & 直线性好 → SWIPE_RIGHT`；
         *   `Δx < -阈值 & 直线性好 → SWIPE_LEFT`；
         *   否则视为普通晃动。
   - 控制模式状态机与防误触发逻辑
     *   设计简单状态机：
         *   `IDLE`：仅识别“OPEN\_HAND”进入控制模式；
         *   `CONTROL`：识别 `SWIPE_LEFT/RIGHT` 等动态手势进行翻页；
         *   条件（如长时间无手 / 手势为 FIST）返回 `IDLE`。
     *   设计防抖策略：
         *   使用最近 K 帧预测投票；
         *   限制动作触发时间间隔（cooldown）。
   - 动态模块评估与可视化
     *   编写 `eval_dynamic.py`：
         *   离线评估左右挥手识别准确率、混淆矩阵；
         *   统计误触发率（no-swipe 情况下错误识别为 swipe 的比例）。
     *   可视化：
         *   绘制若干典型样本的 3D 质心轨迹图；
         *   对比成功/失败样例。

   预期输出：

   *   `dataset_dynamic.py`, `gesture_trajectory.py`（质心与规则实现）；
   *   `state_machine.py`（控制模式逻辑）；
   *   `eval_dynamic.py` 及对应结果图。

6. **跨应用控制框架，实现手势到按键映射**：**张靖琳**

   - 三层抽象的控制框架实现
     *   手势层（由模型/规则输出）：如 `OPEN_HAND`, `FIST`, `SWIPE_LEFT` 等；
     *   语义命令层：如 `NEXT_SLIDE`, `PREV_SLIDE`, `PLAY_PAUSE`, `VOLUME_UP` 等；
     *   应用映射层（配置驱动）：
         *   `configs/ppt.yaml`：定义命令到按键（`NEXT_SLIDE -> right` 等）；
         *   `configs/video.yaml`：定义用于 VLC/PotPlayer 的按键映射。
   - 按键模拟与应用适配
     *   基于 `PyAutoGUI`（或 `keyboard` 等库）实现：
         *   单按键：`press('right')`；
         *   组合键：`hotkey('shift','f5')` 等。
     *   对接：
         *   PowerPoint 放映模式；
         *   至少一个常用视频播放器（VLC、PotPlayer 等）。
   - 统一 demo 脚本与参数化运行
     *   编写 `run_demo.py`：
         *   支持命令行参数：`--mode ppt` / `--mode video`；
         *   支持选择静态/动态/组合控制策略；
   - 端到端调试与优化
     *   完成从“相机 → 识别 → 控制”的整体调试。

   预期输出：

   *   `gesture_router.py`（手势→命令）；
   *   `app_adapter.py`（命令→按键）；
   *   `configs/ppt.yaml`, `configs/video.yaml`；
   *   `run_demo.py` 可执行脚本（最终答辩主要展示入口）。



### 三、项目框架

1.  **基础模块：基于点云的静态 3D 手势识别 + PPT 控制**
    *   深度相机采集 RGB-D
    *   手部分割（可参考已有的 segmentation.py 思路）
    *   点云生成与预处理（可参考已有的 ptcloud.py 的相机模型 &投影）
    *   PointNet 类网络进行静态手势分类
    *   使用 PyAutoGUI 映射到 PPT 快捷键（上一页/下一页/播放/退出）
2.  **创新点 1：基于 3D 轨迹的动态手势（左右挥手翻页）**
    *   不再只看“一帧姿势”，而是利用**连续多帧手部点云轨迹**判断“左划/右划”，减小误触发概率。
    *   技术上可以有两种方案（二选一，工作量可控）：
        *   方案 A：**规则 + 特征工程**（容易落地）
            *   每帧计算手部质心 3D 坐标 `(x_t, y_t, z_t)`，在 T 帧内看整体位移向量：
                *   `Δx > 阈值 & |Δy|、|Δz| 较小` → 右划
                *   `Δx < -阈值 & |Δy|、|Δz| 较小` → 左划
            *   结合速度、轨迹平滑程度判断是否为“有效划动”。
        *   方案 B：**PointNet + GRU 的简单时序模型**（更学术、复杂）。
    *   最终与静态手势结合：
        *   静态手势负责模式控制（如“举手张开 = 进入手势控制模式”），
        *   动态挥手负责翻页。
3.  **创新点 2：可扩展的“应用无关”控制框架（PPT + 视频播放器）**
    *   在代码结构上，把“**手势 → 语义命令 → 具体按键**”拆开：
        *   手势层：`OPEN_HAND`, `FIST`, `SWIPE_LEFT`, `SWIPE_RIGHT` …
        *   语义命令层：`NEXT_SLIDE`, `PREV_SLIDE`, `PLAY_PAUSE`, `VOLUME_UP`, `VOLUME_DOWN` …
        *   应用映射层：
            *   对 PPT：`NEXT_SLIDE → Right`, `PREV_SLIDE → Left` …
            *   对视频播放器（如 PotPlayer/VLC）：`PLAY_PAUSE → Space`, `VOLUME_UP → Up` …