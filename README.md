# FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting

<div align="center">

![cover_figure](assets/cover_figure.png)

###  [Paper](https://arxiv.org/pdf/2604.01844) | [Project Page](https://papieta.github.io/fact-gs/) | [Data](https://doi.org/10.11583/DTU.32054451)

</div>

## Introduction

Official repository for our paper titled FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting. 

#### Related repositories (included in the installation):

[Fast Gaussian Splatting CT Rasterizer](https://github.com/PaPieta/gs-ct-rasterizer) | [Fast Gaussian Splatting Voxelizer](https://github.com/PaPieta/gs-voxelizer) | [Fused SSIM](https://github.com/rahul-goel/fused-ssim) (2D and 3D) | [Fused 3D TV](https://github.com/PaPieta/fused-3D-tv)

## 软件架构与功能概览

FaCT-GS 以各向异性 3D Gaussian 表示 CT 衰减场，同时支持投影域重建、体数据拟合、
模型评估、实验复现和 SIBR 交互可视化。主数据流如下：

```text
DICOM / TIFF / NPY / r2_gaussian 数据
                  │
                  ▼
       data_preprocess + dataset readers
     （归一化、扫描几何、相机、FDK 初始化）
                  │
          ┌───────┴────────┐
          ▼                ▼
  train_recon.py      train_volume.py
  投影域 CT 重建       体数据拟合/压缩
          │                │
          └───────┬────────┘
                  ▼
   GaussianModel (xyz/density/scale/rotation)
          ┌───────┼───────────────┐
          ▼       ▼               ▼
   rasterize   voxelize       checkpoint/eval
   X-ray 投影   3D 体素化       指标、图像、点云
                                  │
                                  ▼
                         SIBR PLY + CT Viewer
```

主要功能模块：

| 模块 | 入口或目录 | 作用 |
| --- | --- | --- |
| 数据预处理 | `data_preprocess/` | 合成/真实投影归一化、DICOM 几何读取、螺旋轨迹、FDK 初始化数据生成 |
| 数据与相机 | `fact_gs/r2_gaussian/dataset/` | 读取 `meta_data.json`、训练/测试相机和重建体积边界 |
| Gaussian 模型 | `fact_gs/r2_gaussian/gaussian/` | 参数激活、优化器、增密、分裂、剪枝及 checkpoint |
| CUDA 渲染核心 | `fact_gs/rasterize.py`, `fact_gs/voxelize.py` | X-ray Gaussian 投影和三维体素化 |
| CT 重建 | `train_recon.py` | 冷启动/先验启动重建、legacy/improved densification、评估和快照导出 |
| 体拟合与压缩 | `train_volume.py` | 拟合 `vol_prior`/`vol_gt`，支持量化训练 |
| 评估 | `test_model.py`, `tests/` | PSNR/SSIM、几何/初始化/导出回归测试 |
| 实验 | `experiments/`, `spiral_tools/` | 论文实验、真实螺旋 CT 流程、结果收集与绘图 |
| 配置 | `config/` | Hydra 的 model/optim/eval/profile/tensorboard 配置组 |
| SIBR Viewer | `tools/sibr_viewer/`, `third_party/SIBR_viewers/` | CT Gaussian 导出、伪彩色、椭球/X-ray/crop 交互查看 |

训练输出通常位于 `model.model_path`：

```text
MODEL/
├── point_cloud/step_<N>/point_cloud.pickle    # FaCT-GS checkpoint
├── eval/                                      # 指标与可视化结果
├── tensorboard/                               # TensorBoard event
└── sibr/point_cloud/iteration_<N>/             # 可选 SIBR PLY/JSON
```

Hydra 的解析后配置默认另存于 `outputs/<date>/<time>/.hydra/config.yaml`；如需让配置
随模型一起迁移，可同时保存该文件，或显式设置 `hydra.run.dir`。

## Installation

You need to have an NVIDIA GPU with CUDA installed (the Python training
environment is tested with CUDA 12.1). The interactive SIBR viewer is a separate
C++/CUDA/OpenGL build and can use another system CUDA toolkit; see
[SIBR viewer environment and migration](tools/sibr_viewer/MIGRATION.md).

Create a dedicated Python environment and install the dependencies:

```sh
git clone --recurse-submodules https://github.com/leggage/fact-gs.git
cd fact-gs
conda env create -f environment.yml
conda activate fact-gs
export GLM_HOME="$(pwd)/fact_gs/submodules/glm"
pip install --no-build-isolation -r submodules.txt
```

The GLM_HOME variable is necessary to automatically link it to the voxelizer and rasterizer submodules.

If the repository was cloned without submodules, repair it before installing:

```sh
git submodule update --init --recursive
```

The Conda environment contains Python 3.10, PyTorch 2.4.1, CUDA 12.1 runtime,
NumPy 2.0, Hydra/OmegaConf, TIGRE, the CT rasterizer/voxelizer, fused SSIM and
fused 3D TV. A quick installation check is:

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
PYTHONPATH=. pytest -q tests/test_sibr_export.py tests/test_norm_pipeline.py
```

The CUDA toolkit used to compile extensions must be ABI-compatible with the
PyTorch environment. Set `CUDA_HOME` explicitly when multiple toolkits are
installed. To limit TIGRE compilation to one GPU architecture, set for example
`CUDA_ARCHITECTURES=86` (Ampere) before installing it.

:exclamation: Consider [using mamba instead of conda](https://iamdamilare13.medium.com/mamba-vs-conda-know-the-differences-and-similarities-be3ae94d2542) for much faster installation. When available, it should be enough to replace the env creation command with `mamba env create -f environment.yml`

## Data -> [get it here](https://doi.org/10.11583/DTU.32054451)

The data used in this project is a collection of volumes and projections generated from various publicly available datasets. A major fraction is a copy from the [r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian/tree/main/r2_gaussian/) project. Check the [dataset description](https://doi.org/10.11583/DTU.32054451) for information on sources and licensing.

<details>
<summary><span style="font-weight: bold;">Details on the contents of each data folder</span></summary>

1. ```real_dataset``` and ```synthetic_dataset``` are direct copies from the [r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian/tree/main/r2_gaussian/) project, used in the main performance analysis. Details on the dataset preparation can be found [here](fact_gs/r2_gaussian/data_generator/synthetic_dataset/README.md) for the synthetic dataset, and [here](fact_gs/r2_gaussian/data_generator/real_dataset/README.md) for the real one.
2. ```coral_dataset``` contains multiple resolutions of the same coral scan, used in the scaling study.
3. ```init_dataset``` contains scan pairs used in the warmstart vs coldstart initialization study.
4. ```warmstart_dataset``` contains lung CT scan pairs used in the warmstart impact study in supplementary material.
5. ```teaser_figure``` holds scans used in generating the main teaser figure of the paper.

</details>

## Running the code

All scripts are parameterized with a [Hydra](https://hydra.cc/) config, located in the ```config``` folder. It can be modified directly or when invoking the script. 

Both reconstruction and volume fitting expect the r2_gaussian data layout (```meta_data.json```).

配置分为五组：`model` 决定数据、输出和初始化；`optim` 决定损失、学习率、增密与
停止条件；`eval` 决定评估、可视化和 SIBR 快照；`profile` 控制 PyTorch profiler；
`tensorboard` 控制训练日志。建议为可复现实验新建顶层 preset，并用命令行 override
修改少量参数。Hydra 会把最终解析配置保存到输出目录的 `.hydra/config.yaml`。

<details>
<summary><span style="font-weight: bold;">Parameters shared by all training scripts</span></summary>

(overwrite with *category.parameter=new_value*)

**Model:**
* **num_gaussians** - Number of Gaussians the optimization is initialized with
* **data_source_path** - Path to the source data folder used during optimization
* **model_path** - Directory where checkpoints, evaluation dumps and exported point clouds are saved
* **init_mode** - How Gaussians are initialized:
    * ```auto``` - load `init_<dataset>.npy` when present; otherwise reproduce the R2-Gaussian FDK initialization (`intensity`) and cache it under that name
    * ```gradient``` - Gaussian locations sampled using the Sobel-gradient magnitude of an FDK-reconstructed volume as the probability distribution
    * ```intensity``` - R2-Gaussian-compatible uniform random sampling over FDK voxels above `density_thresh` (the historical mode name is retained for compatibility)
    * ```precomputed/prior``` - only applicable for reconstruction (see below)
* *density_thresh* - Minimum voxel density used to discard background voxels during initialization
* *density_rescale* - Empirical scaling applied to the sampled densities to compensate for multi-Gaussian occlusion
* *density_init_scale* - Additional initialization-only density multiplier; `1.0` matches R2-Gaussian, while `0.25` reproduces the older FaCT-GS online behavior
* *init_seed* - Random seed used by online voxel sampling
* *save_generated_init* - Cache an auto-generated R2-style initialization as `init_<dataset>.npy` for reproducible future runs
* *scale_min* / *scale_max* - Lower and upper scale bounds expressed as a fraction of the target volume, converted to world units at runtime
* *eval* - When False only the training cameras are loaded from ```meta_data.json```

**Optimization:**
* **steps** - Number of optimization steps to run. One step equals one rasterization/voxelization pass with backward propagation. In CT recon, an ```iteration``` consists of processing all available projections. For volume fitting, since there is only one volume used, ```step==iteration```
* *position_lr_init/final/max_steps* - Learning rate schedule for Gaussian centers; ```*_max_steps``` is expressed as a fraction of ```steps```
* *density_lr_init/final/max_steps* - Learning rate schedule for Gaussian densities/opacity
* *scaling_lr_init/final/max_steps* - Learning rate schedule for per-axis scales
* *rotation_lr_init/final/max_steps* - Learning rate schedule for spherical harmonics rotations
* *lambda_dssim* - Weight of the DSSIM loss component (set to 0 to disable)
* *lambda_tv* - Weight of the 3D total-variation regularizer
* *lambda_frequency* - Optional log-Fourier-magnitude loss on high-frequency projection content; disabled by default so the R2 baseline remains unchanged
* *frequency_highpass_cutoff* - Radial frequency cutoff relative to Nyquist used by the optional frequency loss
* *ssim3d_early_stop / ssim3d_early_stop_threshold* - Optional early-stopping guard. When enabled, training exits as soon as the reported 3D SSIM reaches or exceeds the provided threshold. Useful for time-to-SSIM studies (defaults to disabled).
* *training_time_limit_seconds* - Optional wall-clock limit (seconds). When >0, stops training once the accumulated training time crosses the threshold.
* **densify_gaussians** - Enables/disables periodic Gaussian densification and pruning
* *density_min_threshold* - Minimum density allowed during pruning; Gaussians below the threshold get removed
* *densification_interval/densify_from_step/densify_until_step_percent* - Controls when densification starts, how often it is triggered, and up to what portion of training it remains active
* *densify_grad_threshold* - Gradient magnitude threshold a Gaussian must exceed to be duplicated during densification
* *densify_scale_threshold* - Largest acceptable Gaussian scale (in % of the volume size) before it gets split during densification
* *max_screen_size* / *max_scale* / *max_num_gaussians* - Optional guards that clamp 2D footprint, 3D scale, or total Gaussian count; `max_num_gaussians` is a multiplier of the initial `num_gaussians` (default `1.1` = 110%)

**Evaluation:**
* *eval_in_training* - Run quantitative evaluation in the middle of training
* *every_n_steps* - Evaluation frequency measured in optimization steps (ignored when ```eval_in_training=False```)
* *eval_start* - If True, evaluate the initialization before any optimization step
* *eval_end* - If True, evaluate after the final step (used for leaderboard metrics)
* *extra_eval_iter_num* - (optional) specific iteration to force an additional evaluation and metric export; useful for catching early progress snapshots such as the 150-iteration dumps used in the paper (`eval_default_extra_150`). For reconstruction runs, iterations equal full sweeps over the training cameras, while for volume fitting `iteration == step`.
* *visualize_at_eval* - When enabled, dumps reconstructed volumes (tiff + preview) at evaluation checkpoints
* *visualize_gaussians* - Exports Gaussian position/footprint visualizations and error maps for debugging
* *sibr_export / sibr_export_interval* - Periodically export CT Gaussian snapshots for the bundled SIBR viewer; the final step is always exported
* *sibr_camera_mode* - `orbit` creates full-volume inspection cameras; `spiral` preserves acquisition cameras for geometry diagnosis

The reported 3D SSIM follows the original R2-Gaussian protocol: 2D SSIM is
averaged over non-empty ground-truth slices along X, Y and Z, followed by an
average over the three axes. It is intentionally not replaced by a volumetric
SSIM kernel because the two metrics are not numerically comparable.

TensorBoard logging is enabled by default for reconstruction. Event files are
written to `<model_path>/tensorboard`; disable it with
`tensorboard.enabled=false` when desired.

**Profiling:**
* *profile* - Turns PyTorch's profiler on/off
* *profile_wait* - How many steps to wait before collecting a trace
* *profile_active* - Number of active steps that are recorded in the trace

</details>

<details>
<summary><span style="font-weight: bold;">Some more details on using Hydra</span></summary>

Parameters can either be changed directly in the ```config``` folder or when invoking the script.

To change a specific parameter, use:

```python script.py category.parameter=new_parameter_val```

e.g., ```python train_recon.py optim.steps=10000```

To change a whole config preset, use: 

```python script.py category=new_category_preset```

e.g., ```python train_recon.py eval=eval_silent```

For major adjustments, it is recommended to create your own config presets. 

</details>

### CT Reconstruction

The default reconstruction can be run with:

```sh
python train_recon.py \
    model.data_source_path=your/path/to/scan/data \
    model.model_path=path/where/trained/model/should/be/saved 
```

<details>
<summary><span style="font-weight: bold;">Parameters unique to reconstruction</span></summary>

**Model:**
* **init_mode**:
    * ```prior``` - Warm-start the optimization from a volume prior fitted with ```train_volume```. Requires setting **prior_path** to a valid model (see below)
    * ```precomputed``` - Load initialized Gaussians from a precomputed point cloud. Legacy from the separate r2_gaussian init procedure (expects an ```init_[data_name].npy``` file in the data folder).
* **prior_path** - Absolute path to the ```point_cloud.pickle``` model file that should be loaded when ```init_mode=prior```


**Optimization:**
* *tv_vol_size* - Side length (in voxels) of the cube that is randomly sampled for the 3D TV loss during reconstruction

</details>

### Volume fitting

Volume fitting can be split into two categories:

1. Fitting for warm-starting CT reconstruction with a volumetric prior. The target volume is assumed to be named ```vol_prior.[npy/tiff]``` (controlled with ```model.vol_name```):

    ```sh
    python train_volume.py \
        model.data_source_path=your/path/to/volume/data \
        model.model_path=path/where/trained/model/should/be/saved 
    ```

2. Fitting for volume compression, or simply for creating a Gaussian-based representation. Target volume is assumed to be named ```vol_gt.[npy/tiff]```. Here we train for longer to get a closer match:
    ```sh
    python train_volume.py \
        --config-name compress_volume \
        model.data_source_path=your/path/to/volume/data \
        model.model_path=path/where/trained/model/should/be/saved 
    ```

<details>
<summary><span style="font-weight: bold;">Parameters unique to volume fitting</span></summary>

**Model:**
* **vol_name** - Name (without extension) of the ground-truth volume inside the dataset directory that should be fitted, e.g., ```vol_prior``` or ```vol_gt```

**Optimization:**
* *quantize* - Enables straight-through estimator quantization of Gaussian positions/scales/rotations/densities during training for model-size control. Useful for GS-based compression
* *pos_bits* / *scale_bits* / *rot_bits* / *feat_bits* - Bit precision allocated to xyz, scale, rotation and density/features when ```quantize=True``` (also used by the model size reporter)

</details>
    
### CT reconstruction from a volumetric prior

First fit the Gaussian representation to a volume (see above). Find path to the trained point cloud (ends with ```point_cloud.pickle```). Then call:
```
python train_recon.py \
    --config-name fromPrior_recon \
    model.data_source_path=your/path/to/scan/data \
    model.model_path=path/where/trained/model/should/be/saved \
    model.prior_path=path/to/trained/point_cloud.pickle \
```

### Testing existing models

Use `test_model.py` to load a finished checkpoint and report PSNR/SSIM scores without re-running optimization. The script reuses the Hydra config system (defaults to `config/default_test.yaml`), so point it to your model/data pair just like the training scripts:

```sh
python test_model.py \
    model.data_source_path=your/path/to/scan/or/volume \
    model.model_path=path/to/trained/model \
    model.target=recon  # or "vol" for volume compression checkpoints
    model.vol_name=vol_gt # Target volume name used in volume fitting, ignore for testing recon models
```

## Prepare your own data

>Data preparation/generation is largely copied from [r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian/tree/main/r2_gaussian/).

Our code supports both cone beam and parallel beam configurations.

If you have ground-truth volumes but do not have X-ray projections, follow [these instructions](fact_gs/r2_gaussian/data_generator/synthetic_dataset/README.md) to generate your own dataset.

If you have (more than 100) X-ray projections but do not have ground-truth volumes, follow [these instructions](fact_gs/r2_gaussian/data_generator/real_dataset/README.md).

If you want to test your own data, please first convert it to the r2_gaussian format (```meta_data.json```).

:exclamation: Gaussian initialization is integrated into the reconstruction step for a smoother experience. The separate init capability from r2_gaussian is still retained, but not recommended.

真实 DICOM 螺旋数据可使用 `data_preprocess/norm_pipeline.py` 的 YAML 流程。启用
`real.auto_svoxel_from_gt=true` 时，重建范围由 GT DICOM 的 Rows/Columns、
PixelSpacing 和切片位置计算，避免直接沿用配置文件中的占位 `sVoxel/offOrigin`。
详细字段见 [data_preprocess/README_norm.md](data_preprocess/README_norm.md)。

## SIBR CT Gaussian Viewer

仓库内置修改版 Graphdeco SIBR viewer 和 FaCT-GS 转换桥。它保留 Gaussian 的位置、
各向异性尺度和四元数，将 CT density 或 densification gradient 映射为蓝—青—黄—红
伪彩色，并提供以下查看能力：

- `Splats`、初始点和真实各向异性 `Ellipsoids` 三种模式；
- 椭球 X-ray 累加显示、抽样步长、尺度和透明度控制；
- 三维 Crop Box 裁剪，便于查看内部结构；
- 完整体积 circular orbit 相机，或原始 spiral acquisition 相机；
- 训练期间定期导出，以及旧 checkpoint 的离线批量转换；
- density 与增密梯度着色，邻接 JSON 保存颜色/尺度/数量元数据。

快速使用已有模型：

```bash
conda activate fact-gs
python tools/sibr_viewer/export_fact_gs.py /absolute/path/to/model --step 30000
tools/sibr_viewer/run_viewer.sh /absolute/path/to/model 30000
```

训练期间保存 viewer 快照：

```bash
python train_recon.py \
  model.data_source_path=/absolute/path/to/data \
  model.model_path=/absolute/path/to/model \
  eval.sibr_export=true \
  eval.sibr_export_interval=1000 \
  eval.sibr_camera_mode=orbit
```

viewer 需要桌面 OpenGL 环境和单独的系统依赖。完整的构建命令、界面说明、输出格式、
densification-gradient 重算方法和限制见 [SIBR 使用说明](tools/sibr_viewer/README.md)；
换机安装、源码补丁、验证和故障排查见
[SIBR 迁移教程](tools/sibr_viewer/MIGRATION.md)。当前实现是离线 snapshot viewer，
尚不支持训练端网络实时推流或在窗口内热切换 iteration。

## Running experiments from the paper

The `experiments/` folder contains ready-made shells for each study discussed in the paper:

- `main_experiment.sh` – Reconstructions for all scans in the r2_gaussian dataset (Tab. 1).
- `time_limit_teaser_experiment.sh` – Chest/walnut/coral reconstructions capped at fixed training-time budgets (30 s / 60 s / 120 s) (Fig 1).
- `scaling_study_coral_100it.sh` – Scaling capability - 100-iteration reconstruction time for coral scans with increasing volume size (Fig 4a).
- `scaling_study_coral_090ssim.sh` – Scaling capability - above, but stop reconstruction once 3D SSIM reaches 0.90 (Fig 4b).
- `init_comparison_experiment.sh` – Comparing the impact of various initialization strategies (Fig. 5)
- `compression_experiment.sh` – Quality of volume fitting and compression (Fig. 6, Tab. 2)
- `warm_start_experiment.sh` – Impact of warm-start on reconstruction speed and quality (Supplementary material, Tab. 3)
- `gaussian_study_r2data_experiment.sh` and `gaussian_study_coraldata_experiment.sh` – Ablation study on the impact of the initial number of Gaussians on reconstruction quality (Supplementary material, Fig. 8). :exclamation: Warning - slow.


All scripts default to `$(pwd)` as the root for data and model outputs. To run them on a different storage location, export `MAIN_ROOT=/path/to/root` before launching, e.g.

```bash
export MAIN_ROOT=/my/data/and/models/path
source experiments/main_experiment.sh
```

Each shell wraps the corresponding helper in `experiments/helpers/` to collect metrics into CSV files right after the training finishes.


## Citation

If this repository has helped your research, please consider citing our work:
```
@misc{pieta2026,
      title={FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting}, 
      author={Pawel Tomasz Pieta and Rasmus Juul Pedersen and Sina Borgi and Jakob Sauer Jørgensen and Jens Wenzel Andreasen and Vedrana Andersen Dahl},
      year={2026},
      eprint={2604.01844},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01844}, 
}
```

## Acknowledgements

Based on [r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian/tree/main/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization). Inspired by [image-gs](https://github.com/NYU-ICL/image-gs), [taming-3dgs](https://github.com/humansensinglab/taming-3dgs), [StopThePop](https://github.com/r4dl/StopThePop).

## License

MIT License, excluding the contents of folders ```fact_gs/r2_gaussian``` and ```fact_gs/submodules```. See LICENSE file for details.
