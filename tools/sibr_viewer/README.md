# FaCT-GS SIBR CT Viewer 使用说明

本目录提供 FaCT-GS 与 Graphdeco SIBR `gaussianViewer` 的离线桥接。它将训练中的
CT Gaussian 或已保存 checkpoint 转成 SIBR PLY，并使用仓库内的修改版 viewer
交互查看几何、密度和增密梯度。

完整换机安装、源码补丁、分层验证和故障排查见 [MIGRATION.md](MIGRATION.md)。

## 1. 已实现功能

- 保留 `xyz`、激活后的各向异性 `scale` 和 `wxyz` quaternion；
- density 或 densification gradient 的 1%/99% 分位伪彩色显示；
- density/gradient 只决定颜色，PLY opacity 是独立的固定显示参数；
- `Splats`、`Initial Points` 和 `Ellipsoids` 三种渲染模式；
- 椭球 X-ray 累加模式，可显示被外层遮挡的内部 Gaussian；
- `Display every Nth`、椭球尺度、透明度范围和透明度强度控制；
- 三维 Crop Box 同时裁剪 splat/椭球，便于剖开内部结构；
- 默认生成覆盖完整 reconstruction volume 的 64 个 circular-orbit 相机；
- 可选保留真实 spiral acquisition cameras，用于扫描几何诊断；
- 训练时按 step 定期导出，或训练后批量转换 checkpoint；
- 自动补齐非摄影 CT 场景所需的最小 COLMAP scaffold 和 proxy mesh。

当前边界：viewer 启动时只加载一个 iteration，不支持窗口内时间轴、热重载或训练端
网络推流。要查看另一 step，需要重新启动。此处的 X-ray 模式是椭球的顺序无关加法
显示，不是 FaCT-GS 物理 X-ray projector 输出的 DRR。

## 2. 软件架构与数据流

```text
训练中 GaussianModel ── export_gaussian_model() ─┐
                                                  ├─> SIBR PLY + JSON
point_cloud.pickle ── export_fact_gs.py ──────────┘       │
                                                          ▼
scanner/相机 ── cameras.json + minimal COLMAP scene ─> GaussianView
                                                          │
                         ┌──────────────┬──────────────────┤
                         ▼              ▼                  ▼
                       Splats     Initial Points       Ellipsoids
                                                        X-ray/Crop
```

核心文件：

| 文件 | 作用 |
| --- | --- |
| `fact_gs/utils/sibr_export.py` | PLY/JSON、orbit/spiral 相机和最小场景导出 |
| `fact_gs/utils/densify_gradient.py` | 在全部训练视角上重算 view-space 增密梯度 |
| `tools/sibr_viewer/export_fact_gs.py` | 已有 checkpoint 的命令行转换器 |
| `tools/sibr_viewer/run_viewer.sh` | 定位二进制、补齐 CT 场景并启动指定 iteration |
| `tools/sibr_viewer/build_viewer.sh` | 配置、编译并安装仓库内 SIBR viewer |
| `third_party/SIBR_viewers/source/src/projects/gaussianviewer/` | CT viewer、椭球 renderer 和 shader |
| `train_recon.py` | 训练期 SIBR snapshot hook |
| `config/eval/eval_default.yaml` | `sibr_export*` 配置默认值 |

## 3. 构建 Viewer

Python 训练环境和 SIBR 编译环境彼此独立。当前默认构建配置为 CUDA 12.8、GCC/G++ 11
和 CUDA architecture 120；迁移到其他显卡时必须设置正确的 toolkit 与 architecture。

Ubuntu 22.04 参考依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
  libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
  libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
  libavformat-dev libavutil-dev libswscale-dev libeigen3-dev \
  libxxf86vm-dev libembree-dev
```

使用默认值构建：

```bash
tools/sibr_viewer/build_viewer.sh
```

覆盖本机配置示例：

```bash
CUDA_HOME=/usr/local/cuda-12.1 \
SIBR_CUDA_ARCHITECTURES=86 \
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
SIBR_BUILD_JOBS=8 \
tools/sibr_viewer/build_viewer.sh
```

还可设置 `SIBR_SOURCE_DIR`、`SIBR_BUILD_DIR`、`SIBR_INSTALL_DIR`。CUDA、编译器或
architecture 发生变化后，请使用新的 `SIBR_BUILD_DIR`，不要复用旧 CMake cache。

## 4. 查看已有模型

转换一个 step：

```bash
conda activate fact-gs
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model --step 30000
```

省略 `--step` 会转换 `MODEL/point_cloud/step_*` 下的所有 checkpoint：

```bash
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model
```

启动最大 iteration 或指定 iteration：

```bash
tools/sibr_viewer/run_viewer.sh /absolute/path/to/model
tools/sibr_viewer/run_viewer.sh /absolute/path/to/model 30000
```

如使用外部 viewer 二进制：

```bash
SIBR_VIEWER_BIN=/absolute/path/to/SIBR_gaussianViewer_app \
  tools/sibr_viewer/run_viewer.sh /absolute/path/to/model 30000
```

## 5. Density 与增密梯度着色

默认按激活后的 CT density 着色：低值蓝色，高值红色。导出器还可以使用和
densification 决策一致的 view-space gradient：

```bash
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model \
  --step 30000 \
  --color-by densify-gradient \
  --training-config /absolute/path/to/hydra-output/.hydra/config.yaml
```

该统计量对 Gaussian `i` 定义为：

```text
sum_visible_views mean(abs(dL/du_i), abs(dL/dv_i)) / visible_view_count_i
```

新 checkpoint 会保存当前训练增密窗口的 `densify_grad`，可直接着色。旧 checkpoint
没有该字段时，命令会用训练配置遍历全部训练相机重算，并缓存为同目录下的
`densify_grads_view_average.npz`。重算需要 CUDA、完整训练数据和可用的 FaCT-GS
rasterizer，目前要求训练配置 `optim.use_fused_ssim=false`。

不同 snapshot 默认分别按自身 1%/99% 分位归一化，因此颜色适合观察单帧空间分布，
不代表跨 step 的同一绝对数值标尺。精确范围见相邻 `point_cloud.json`。
梯度导出还会生成原始 float32 sidecar `point_cloud.filter.bin`；它不会改变标准 PLY，
但更新后的 viewer 会读取它并按绝对梯度筛选。默认下限取训练配置中的
`optim.densify_grad_threshold`，也可用 `--gradient-threshold` 显式覆盖。

## 6. 查看相机

默认 `orbit` 相机根据 scanner 的 `offOrigin`、`sVoxel` 和 `DSO` 生成方形画幅的完整
体积环绕视角。FOV 自动覆盖带 10% margin 的体积包围球；DSO 太小时相机会外移。

为已有模型重新生成相机且不改 PLY：

```bash
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model \
  --data /absolute/path/to/r2gs/dataset \
  --cameras-only
```

检查真实采集轨迹时使用 `spiral`：

```bash
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model \
  --data /absolute/path/to/r2gs/dataset \
  --geometry /absolute/path/to/geometry_used.yml \
  --camera-mode spiral \
  --cameras-only
```

只有既没有 `--data` 又没有训练期几何信息时，导出器才使用 `[0, 0, -5]` fallback。

## 7. 训练期间导出

在任意重建命令追加：

```text
eval.sibr_export=true
eval.sibr_export_interval=1000
eval.sibr_camera_mode=orbit
```

例如：

```bash
python train_recon.py \
  model.data_source_path=/absolute/path/to/data \
  model.model_path=/absolute/path/to/model \
  eval.sibr_export=true \
  eval.sibr_export_interval=1000
```

最终 step 即使不能整除 interval 也会导出。每个 Gaussian 的 PLY payload 为 68 bytes，
百万 Gaussian 每帧约 68 MB（另有 header/JSON）；不要以 densification interval 为
快照间隔，否则可能快速占满磁盘。

## 8. 界面操作

在浮动控制面板中：

- `Render Mode`：切换 `Splats`、`Initial Points`、`Ellipsoids`；
- `X-ray (show interior)`：关闭深度遮挡并累加椭球颜色；
- `Display every Nth`：只画每 N 个椭球，默认 10，用于控制交互帧率；
- `Ellipsoid Scale`：仅改变显示尺寸，不修改模型；
- `Opacity Min/Max`、`Opacity Strength`：调整椭球显示透明度；
- `Filter by diagnostic value`、`Gradient Min/Max`：按原始 densification 梯度值筛选；
  梯度下限默认是训练时的 densification 阈值；
- `Crop Box`：启用后分别调整 XYZ min/max，裁出关注区域；
- `Scaling Modifier`：控制 splat 的可视尺度。

若模型看似全白，请确认 PLY 是用当前导出器重新生成的；旧版本曾固定输出白色。
梯度筛选、opacity 筛选和 Crop Box 是三套独立条件，同时启用时取交集，不会互相
改写参数。取消 `Filter by diagnostic value` 即可显示全部梯度范围，而 opacity 控件
仍按原行为工作。

## 9. 输出格式

```text
MODEL/sibr/
├── cfg_args
├── cameras.json
├── point_cloud/
│   └── iteration_<N>/
│       ├── point_cloud.ply
│       ├── point_cloud.filter.bin  # 可选：原始梯度/诊断值和推荐阈值
│       └── point_cloud.json
└── viewer_scene/
    ├── cameras.json
    ├── input.ply
    ├── images/reference.ppm
    └── sparse/0/{cameras.txt,images.txt,points3D.txt}
```

PLY 使用 17 个 little-endian float32 字段：

```text
x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity
scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
```

activated scale 写为 `log(scale)`，SIBR 加载时再 `exp`；quaternion 为归一化 `wxyz`；
伪彩色写入 degree-0 SH；opacity 写为 display opacity 的 logit。

## 10. ldctl004 迁移烟雾测试

仓库提供了本机真实 DICOM 数据的配置模板：

```bash
python data_preprocess/norm_pipeline.py \
  --config data_preprocess/configs/real_ldctl004_spiral_ntrain1000_sibr_test.yml
python train_recon.py --config-name ldctl004_sibr_test
tools/sibr_viewer/run_viewer.sh \
  models/real/ldctl004/spiral/ntrain1000/factgs_sibr_cold_g4_outside
```

该预处理配置中的原始 DICOM 路径是主机本地路径，迁移后必须修改。它保持原生
`64x736` detector，并从 GT DICOM sampling grid 计算重建范围和轴向中心。训练 preset
使用预生成 FDK 点云冷启动、improved densification、LAS、0.1 框外 Gaussian budget
以及每 1000 step 的 SIBR snapshot。

## 11. 本次修改摘要

- CT density 从“白色 + opacity”改为伪彩色，opacity 与物理衰减值解耦；
- checkpoint 新增 densification gradient/denominator，旧模型支持离线重算与缓存；
- viewer 新增椭球 X-ray、抽样、透明度、尺度和 Crop Box 控件；
- 修复 CT 最小场景、proxy mesh、完整体积 orbit 相机和 off-origin volume 查看；
- 构建脚本支持 CUDA/toolchain/architecture/job 数覆盖，并兼容 CUDA 11 新 glibc；
- SIBR 工程源码作为普通目录纳入仓库，构建/安装/extlib 产物继续忽略；
- 真实数据预处理支持从 GT DICOM grid 推导 `sVoxel` 和 `offOrigin.z`。
