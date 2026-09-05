# FaCT-GS SIBR 迁移、现状与实时监测设计

本文记录 2026-09-05 仓库中的实际实现，目标是在一台没有安装
SIBR viewer 的 Ubuntu/NVIDIA 主机上复现离线查看功能，并为后续实现实时
训练监测保留准确的设计依据。

> 迁移应以本仓库提交的源码为准。不要复制 `build/`、`install/` 或旧
> `CMakeCache.txt`；这些产物绑定目标机的编译器、CUDA 和系统动态库。

## 1. 当前已经实现的功能

当前桥接方式是把 FaCT-GS 的 CT Gaussian 转成 Graphdeco SIBR
`gaussianViewer` 接受的 PLY：

```text
FaCT-GS Gaussian/checkpoint
  -> activated xyz/density/scale/rotation
  -> SIBR degree-0 PLY
  -> model/sibr/point_cloud/iteration_<step>/point_cloud.ply
  -> SIBR_gaussianViewer_app
```

相关文件：

- `fact_gs/utils/sibr_export.py`：转换内存模型或 checkpoint 数据。
- `train_recon.py`：训练过程中按间隔直接导出 SIBR 快照。
- `config/eval/eval_default.yaml`：`sibr_export` 和
  `sibr_export_interval` 配置。
- `tools/sibr_viewer/export_fact_gs.py`：批量转换已经保存的
  `point_cloud/step_*/point_cloud.pickle`。
- `tools/sibr_viewer/run_viewer.sh`：选择本地或指定的 viewer 二进制并启动。
- `tools/sibr_viewer/build_viewer.sh`：构建经过兼容性修改的 SIBR 源码。
- `tools/sibr_viewer/cuda11_glibc_compat.h`：CUDA 11.8 与新 glibc 的声明冲突
  兼容头。
- `tests/test_sibr_export.py`：验证 PLY 属性顺序、几何、scale 和 quaternion。
- `third_party/SIBR_viewers/source/src/projects/gaussianviewer`：viewer 主程序、
  splat/椭球 renderer、X-ray/crop 控件和 shader。

### PLY 语义（必须保持）

SIBR PLY 每个 Gaussian 有 17 个 little-endian float32 字段：

```text
x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity
scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
```

转换规则：

- `xyz` 原样保存。
- activated positive scale 转成 `log(scale)`；SIBR 加载时会激活。
- quaternion 保持 FaCT-GS/SIBR 使用的 `wxyz` 顺序并重新归一化。
- raw checkpoint density 用 `softplus` 激活；内存模型读取 `get_density`。
- density 或 densification gradient 的 1%/99% 分位用于伪彩色归一化。
- degree-0 SH 使用蓝—青—黄—红编码所选诊断量；PLY opacity 使用独立的固定
  显示值，不把 CT density/gradient 解释成物理 opacity。
- 每个 PLY 旁边写 `point_cloud.json`，保存 Gaussian 数量、颜色量分位范围和
  scale 范围。

非摄影 CT 数据还需要导出器生成以下最小场景，供 SIBR 场景解析器使用：

```text
model/sibr/
  cfg_args
  viewer_scene/cameras.json
  viewer_scene/images/reference.ppm
  viewer_scene/sparse/0/{cameras.txt,images.txt,points3D.txt}
```

## 2. 训练和离线使用

训练时导出：

```bash
python train_recon.py <原有 Hydra overrides> \
  eval.sibr_export=true \
  eval.sibr_export_interval=1000
```

最终 step 无论是否整除 interval 都会导出。每个 Gaussian 的 PLY payload 是
`17 * 4 = 68` bytes，另有少量 header/JSON 开销，不建议百万级模型每一步保存。

转换已有 checkpoint：

```bash
conda activate fact-gs
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model
```

为已有 CT 模型生成覆盖完整 volume 的 circular-cone 查看相机（不改 PLY）：

```bash
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model \
  --data /absolute/path/to/r2gs/dataset --cameras-only
```

默认 `orbit` 模式根据 scanner 的归一化 `offOrigin/sVoxel/DSO` 生成 64 个方形
1024x1024 查看相机：全部位于 volume 中心 Z 平面，只改变方位角，朝向 volume 中心，
并用包围球半径加 10% margin 自动计算水平/垂直 FOV。若原 DSO 太小，相机半径会自动
外移，保证在 margin 包围球之外。

若要检查真实窄扇束采集轨迹，可加 `--camera-mode spiral`；训练时对应
`eval.sibr_camera_mode=spiral`。该模式复用 `SceneRecon.getTrainCameras()` 或
FaCT-GS `readCTameras()` 以及 `MODEL/geometry_used.yml`。只有没有 `--data`/训练几何
时才保留 `[0,0,-5]` fallback。

打开最大或指定 iteration：

```bash
tools/sibr_viewer/run_viewer.sh /absolute/path/to/model
tools/sibr_viewer/run_viewer.sh /absolute/path/to/model 5000
```

当前 viewer 只在启动时加载一个 PLY，切换 iteration 必须重启。它尚无训练时间轴、
播放、热重载或训练端网络连接。

## 3. 在全新 Ubuntu 主机上构建

### 3.1 前置条件

- x86-64 Ubuntu 桌面系统，能创建 OpenGL 窗口；纯 SSH/headless 环境不能直接
  运行当前交互 GUI。
- NVIDIA 驱动正常，`nvidia-smi` 可用。
- CUDA toolkit。RTX 50 系显卡使用 CUDA 12.8，并以 `sm_120` 构建。
- GPU compute capability 至少 7.0；`GaussianView` 启动时会检查。

Ubuntu 系统依赖：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
  libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
  libavformat-dev libavutil-dev libswscale-dev libeigen3-dev \
  libxxf86vm-dev libembree-dev
```

包名会随发行版变化。2026-09-05 本机验证环境是 Ubuntu 22.04.5、CMake 3.22.1、
GCC/G++ 11、CUDA 12.8、Boost 1.74、OpenCV 4.5 和 Embree 3.12；已安装二进制链接
`libcudart.so.12` 和 `libembree3.so.3`。不要跨 Ubuntu/CUDA 环境搬运二进制，应在
目标机重新编译。

### 3.2 获取准确的 SIBR 源码

当前 SIBR 基线：

```text
https://gitlab.inria.fr/sibr/sibr_core.git
commit d8856f60c5384cc1975439193bb627d77d917d77
```

修改版完整源码已作为普通目录保存在 `third_party/SIBR_viewers/source`，其中不包含
嵌套 `.git`。目标机 clone 本仓库即可得到构建所需源码，不需要另行 clone SIBR。
上述 commit 用于追溯 upstream；如果重新从 upstream 制作源码树，必须重放第 4 节
兼容补丁并加入 `basic` 与 `gaussianviewer` project。

SIBR CMake 首次配置还会下载 imgui、CudaRasterizer 等 extlibs，需要网络。不要提交
`source/extlibs/*/build`、`build*`、`install/` 或旧 `CMakeCache.txt`。

### 3.3 编译

```bash
tools/sibr_viewer/build_viewer.sh
```

默认 source/build/install 分别为：

```text
third_party/SIBR_viewers/source
third_party/SIBR_viewers/build
third_party/SIBR_viewers/install
```

可用 `SIBR_SOURCE_DIR`、`SIBR_BUILD_DIR`、`SIBR_INSTALL_DIR` 覆盖；CUDA 通过
`CUDA_HOME` 选择，架构通过 `SIBR_CUDA_ARCHITECTURES` 选择。当前机器默认使用
CUDA 12.8、GCC/G++ 11 和 `sm_120`。只有显式选择 CUDA 11.8 时才预包含
`cuda11_glibc_compat.h`。编译器或 CUDA 改变时必须换一个全新 build 目录。

关键 CMake 参数：

```text
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
-DBUILD_IBR_REMOTE=OFF
-DSIBR_USE_EGL=OFF
-DBoost_NO_BOOST_CMAKE=ON
```

`BUILD_IBR_REMOTE=OFF` 表示当前安装不会生成 Network Viewer。实现实时协议时需要
重新启用 remote target，并处理目标系统的编译兼容性。

## 4. 当前 SIBR 源码兼容修改

基于上述 commit，当前源码包含以下本地修改；重新 clone 时必须重放：

1. `CMakeLists.txt`：仅在未定义时设置 `CMAKE_INSTALL_ROOT`，允许仓库内安装。
2. `cmake/linux/dependencies.cmake`：新增默认关闭的 `SIBR_USE_EGL`；Boost 只请求
   `filesystem;date_time`，避免新版本中已 header-only 的 System/Chrono 组件问题。
3. Boost Filesystem API 兼容：
   - `CameraRecorder.cpp`、`ProxyMesh.cpp`、`InteractiveCameraHandler.cpp` 改用
     `boost::filesystem::path(...).extension()`；
   - `Utils.cpp` 改用 `copy_options::overwrite_existing`。
4. Embree 3.12：保持 `embree3` headers/ABI，raycaster 显式创建
   `RTCIntersectContext`，并使用当前 `rtcIntersect*`/`rtcOccluded*` 调用签名。
5. 新 C++ 编译器：`CommandLineArgs.hpp` 的变量模板改成 `inline constexpr`。
6. FFmpeg API 兼容：删除弃用的 `av_register_all`/`avcodec_encode_video2`，改用独立
   `AVCodecContext`、send/receive packet 和 `avcodec_parameters_from_context`。
7. Wayland/GLEW：`Window.cpp` 将“不支持窗口定位”降为 warning，并在已有有效
   context 时容忍 `GLEW_ERROR_NO_GLX_DISPLAY`。
8. CUDA 11.8 + glibc 2.41+：`cuda11_glibc_compat.h` 临时重命名
   `cospi/sinpi/rsqrt` 等声明，规避 exception specifier 冲突。

9. `projects/gaussianviewer`：加入 CT 伪彩色 PLY、真实椭球、X-ray 累加、抽样、
   opacity/scale 和 Crop Box 控件；`projects/basic` 提供 viewer 依赖的基础 renderer。

这些修改已在 Ubuntu 22.04、GCC 11、CUDA 12.8、Boost 1.74、Embree 3 和
OpenCV 4.5 组合完成编译。其他发行版上 Embree/FFmpeg headers 与动态库的 major
version 必须一致；CUDA architecture 必须由目标 GPU 决定。

## 5. 换机后的分层验证

先验证 Python 导出：

```bash
conda activate fact-gs
pytest -q tests/test_sibr_export.py
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model --step 5000
test -f /absolute/path/to/model/sibr/point_cloud/iteration_5000/point_cloud.ply
```

检查二进制和动态库：

```bash
test -x third_party/SIBR_viewers/install/bin/SIBR_gaussianViewer_app
ldd third_party/SIBR_viewers/install/bin/SIBR_gaussianViewer_app | grep 'not found'
```

最后在图形桌面运行：

```bash
tools/sibr_viewer/run_viewer.sh /absolute/path/to/model 5000
```

成功标准：窗口打开、模型可见、相机可交互、`Splats`/`Ellipsoids` 可切换，终端无
CUDA/OpenGL fatal error。所选诊断量表现为伪彩色，透明度仅用于显示。

## 6. 常见失败与诊断

- `SIBR viewer not found`：尚未构建，或 `SIBR_VIEWER_BIN` 不可执行。
- `No snapshots found`：没有 `point_cloud/step_*/point_cloud.pickle`，且训练时未开启
  `eval.sibr_export`。
- iteration 不存在：确认路径严格为
  `sibr/point_cloud/iteration_<整数>/point_cloud.ply`。
- `lib*.so => not found`：缺运行库或复制了别的发行版产物；优先目标机重编译。
- `No CUDA devices detected`：检查驱动、容器 GPU passthrough 和 capability >= 7.0。
- GLEW/`DISPLAY` 错误：确认桌面会话、X11/Wayland/XWayland 和 NVIDIA OpenGL 驱动。
- CMake 找不到 Embree：确认安装 major version与 include/API/link target 一致。
- CUDA/glibc 出现 `cospi`/`rsqrt` 声明冲突：确认 NVCC 命令预包含兼容头；长期
  方案是升级 CUDA。
- 修改 compiler 后仍显示旧路径：换新 build 目录，避免 CMake cache。
- Wayland 报不支持 window position：有当前补丁时应只是 warning。
- 若旧快照仍然全白，请重新运行导出器；新版 PLY 用伪彩色表示所选诊断量。
- 尺度/方向错：确认 activated scale 没被重复 `exp`，quaternion 是 `wxyz`。

## 7. 原版 3DGS Network Viewer 的工作方式

原版 Network Viewer **不传输完整 Gaussian 模型**：

```text
remoteGaussianUI (C++)
  --TCP JSON: camera/FoV/resolution/controls-->
Python train.py（持有当前 GPU Gaussians）
  --当前模型渲染的 RGB8 图像 + source path-->
remoteGaussianUI
```

参考代码：3DGS 的 `gaussian_renderer/network_gui.py`、`train.py` 中每个 iteration
开头的 network loop，以及 SIBR 的 `src/projects/remote/apps/remoteGaussianUI` 和
`RemotePointView.*`。默认 Python 端监听 `127.0.0.1:6009`。

viewer 发送分辨率、FoV、near/far、view/view-projection matrix、train、keep-alive、
SH/covariance 开关和 scaling modifier。训练端构造 camera，用当前内存模型渲染一帧
并返回 uint8 RGB。断连后训练继续。因此它只能看“当前”，不能回放过去，也不能在
viewer 端逐个检查全部 Gaussian。

## 8. FaCT-GS 实时渲染方案

### A. 原版兼容的二维帧协议（首选 MVP）

1. 实现 length-prefixed JSON TCP server，并在训练 step 安全边界非阻塞 poll。
2. 没有 viewer、断连或 viewer 崩溃时训练必须继续。
3. 收到 camera 后，用当前 `gaussians` 生成可视化帧并返回 RGB8。
4. CT 无 RGB SH，应定义固定 density range 的伪彩色 + alpha，避免每帧分位变化。
5. 支持 pause/resume、keep-alive、scaling modifier，并显示实际 step。
6. viewer 时间不计入 training-time 指标；限制分辨率/频率。

关键难点：FaCT-GS 的 `rasterize_proj` 是 X-ray projection renderer，不是自由视角
RGB splat renderer。不能直接替换原版 `render(custom_cam, ...)`；要么适配可视化
Gaussian CUDA rasterizer，要么明确 viewer 展示交互 DRR/X-ray 投影。

### B. 低频推送完整 Gaussian（Ellipsoids/增密检查）

训练端每 N step 推送 `xyz/density/scale/rotation`，viewer 热更新 GPU buffers。
协议必须有 version、step、count、dtype、byte lengths 和 density range；用后台线程、
有界 latest-only 队列和 staging buffer，完整校验后再原子替换。替换时同步 CUDA/GL，
释放所有依赖旧 count 的 buffers，失败则保留旧模型。

百万 Gaussian 的核心原始数据至少约 44 MB/次
(`xyz 12 + density 4 + scale 12 + rotation 16` bytes/点)，只适合低频传输。

### C. 文件快照 + 时间轴（全过程回看）

在 `GaussianView` 增加 iteration 扫描/刷新、slider、前后帧、播放/暂停/FPS/循环，
后台读 PLY 后在渲染线程安全替换 GPU 数据，同时保持相机和 GUI 状态。导出器应先写
临时文件再 atomic rename，避免热加载半写文件。

推荐组合：A 高频看当前二维渲染，B 低频检查 Gaussian/Ellipsoids，C 保存关键节点并
回看全过程。三者必须共享固定 density colormap/range 元数据。

## 9. 实时功能安全与验收

- 无连接、断连、viewer 崩溃时训练继续；pause 只暂停 optimizer。
- 固定相机能看到 step 单调增长和 densification/pruning 变化。
- 单独报告训练计算时间与包含 viewer 的 wall time。
- 默认只监听 loopback；跨主机使用 SSH tunnel，不暴露无认证控制端口。
- 不通过网络反序列化 pickle。
- 限制 resolution、payload length 和 Gaussian count；socket 使用 `recv_exact` 处理
  partial read（原版单次 `recv(length)` 不够健壮）。
- GPU 更新失败时保留上一份可用状态。

## 10. 已修复的查看相机问题

旧实现的 `ensure_minimal_sibr_scene()` 只有 `[0,0,-5]`、identity rotation 的占位
相机，导致初始视角呈轴向俯视。第一版修复曾直接导出真实 spiral acquisition
camera，但其 detector 是窄 fan 长条，只能覆盖 volume 的很小区域。

当前默认修复是与采集几何分离的 circular-cone inspection orbit：同一中心 Z、只转
角度、方形画幅、FOV 自动覆盖完整 volume。真实 spiral 轨迹仍作为可选诊断模式。
两种模式均使用 Graphdeco `camera_to_JSON` 的 c2w rotation/position 约定。

当前 ldctc002/ntrain1000 的 orbit 使用 64 个相机，volume 中心约
`(0,0,-1.121)`，相机 XY 半径为 scanner DSO `3.59593`，所有相机 z 均为
`-1.121`。FOV 根据归一化 2x2x2 bbox 的包围球和 10% margin 自动计算。

## 11. 当前未解决的问题

1. 离线 viewer 一次只加载一个 iteration，没有时间轴/热重载。
2. 当前构建关闭 `BUILD_IBR_REMOTE`，没有 Network Viewer。
3. `train_recon.py` 没有 network server。
4. CT 自由视角实时渲染的视觉定义及 renderer 未实现。
5. density/gradient colormap 已接线；历史 PLY 需重新导出才能获得伪彩色。
6. 每个 snapshot 独立按分位映射，跨时刻颜色不是同一绝对标尺。
7. PLY 尚未用临时文件 + atomic rename，不可安全热加载。
8. build 脚本默认 CUDA 12.8/GCC 11/sm_120，迁移时需显式覆盖；尚无 GPU 自动探测。
9. 只有 PLY/相机单测，没有自动 GUI/OpenGL/CUDA smoke test。
10. 本轮检查中当前机器 `nvidia-smi` 无法连接驱动，所以只确认 binary 存在且
    `ldd` 无缺库，不能视为本轮 GPU 运行验证。

## 12. 迁移交付清单

版本库至少应包含：

```text
config/eval/eval_default.yaml
train_recon.py
fact_gs/utils/sibr_export.py
fact_gs/utils/densify_gradient.py
tests/test_sibr_export.py
tools/sibr_viewer/{README.md,MIGRATION.md,export_fact_gs.py,run_viewer.sh,
                   build_viewer.sh,cuda11_glibc_compat.h}
third_party/SIBR_viewers/source（含 basic/gaussianviewer，排除 build/install）
```

不要依赖或搬运 `build*`、`install/`、extlibs build 目录或 CMake cache。

目标机执行顺序：驱动/CUDA验证 → 系统依赖 → 固定 SIBR 基线和补丁 → 干净 build
→ `ldd` → Python exporter test → 单 checkpoint GUI smoke test → 再开启训练快照或
实时协议开发。
