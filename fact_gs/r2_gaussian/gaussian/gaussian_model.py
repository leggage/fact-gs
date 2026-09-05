#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import math
import torch
from torch import nn
import pickle

sys.path.append("./")

from simple_knn._C import distCUDA2
from fact_gs.r2_gaussian.utils.general_utils import t2a
from fact_gs.r2_gaussian.utils.system_utils import mkdir_p
from fact_gs.r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)

EPS = 1e-5


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        # 人口记账（研究用，eval.record_population 开启后由 train_recon 置 True）：
        # densify/prune 事件把出生（clone/split，含 parent 位置）与剪枝
        # （含原因码）追加到 event_log；train_recon 在每个事件后取走并清空。
        self.event_log_enabled = False
        self.event_log = []
        # 干预实验（研究用）：
        # _densify_exclude：本次事件 densify 选择要排除的高斯（2a 框外预算）；
        # clamp_birth_box：出生子代 clamp 到的框 (lo, hi)（2b）；None=不干预。
        self._densify_exclude = None
        self._born_this_event = 0
        self.clamp_birth_box = None
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # return self.rotation_activation(self._rotation)
        return self._rotation
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=int(training_args.position_lr_max_steps * training_args.steps),
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=int(training_args.density_lr_max_steps * training_args.steps),
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=int(training_args.scaling_lr_max_steps * training_args.steps),
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=int(training_args.rotation_lr_max_steps * training_args.steps),
        )

    def update_learning_rate(self, step):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(step)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(step)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(step)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(step)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        # We save pickle files to store more information

        mkdir_p(os.path.dirname(path))

        xyz = t2a(self._xyz)
        densities = t2a(self._density)
        scale = t2a(self._scaling)
        rotation = t2a(self._rotation)

        # Keep the exact view-space statistic used by densification.  Older
        # snapshots omitted it, which made post-training diagnostic colouring
        # impossible without rendering the training views again.
        denom = self.denom
        if denom.numel() == self.get_xyz.shape[0]:
            safe_denom = torch.clamp_min(denom, 1.0)
            densify_grad = self.xyz_gradient_accum / safe_denom
            densify_grad = torch.where(denom > 0, densify_grad, torch.zeros_like(densify_grad))
            densify_grad = t2a(densify_grad)
            densify_denom = t2a(denom)
        else:
            densify_grad = None
            densify_denom = None

        out = {
            "xyz": xyz,
            "density": densities,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self.scale_bound,
            "densify_grad": densify_grad,
            "densify_denom": densify_denom,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def load_ply(self, path, spatial_lr_scale=None):
        if spatial_lr_scale is not None:
            self.spatial_lr_scale = spatial_lr_scale
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.scale_bound = data["scale_bound"]
        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, tag="prune", reasons=None):
        """删除 mask 命中的高斯。tag 区分剪枝来源；reasons 为逐行原因码。"""
        if self.event_log_enabled and bool(mask.any()):
            self.event_log.append({
                "kind": "prune",
                "tag": tag,  # "prune" | "split_parent"
                "reason": None if reasons is None
                else reasons[mask].detach().cpu().numpy(),
                "xyz": self.get_xyz[mask].detach().cpu().numpy(),
            })
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
    ):
        self._born_this_event += len(new_xyz)
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )
        if self._densify_exclude is not None:
            # clone 在前已扩容：exclude 是 pre-clone 的行号，pad 到当前 N
            exclude = torch.zeros(n_init_points, dtype=torch.bool,
                                  device=selected_pts_mask.device)
            exclude[: self._densify_exclude.shape[0]] = self._densify_exclude
            selected_pts_mask = selected_pts_mask & ~exclude

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        if self.event_log_enabled and bool(selected_pts_mask.any()):
            self.event_log.append({
                "kind": "birth",
                "type": "split",
                "xyz": new_xyz.detach().cpu().numpy(),
                "parent_xyz": self.get_xyz[selected_pts_mask].repeat(N, 1)
                .detach().cpu().numpy(),
            })

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter, tag="split_parent")

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )
        if self._densify_exclude is not None:
            selected_pts_mask = selected_pts_mask & ~self._densify_exclude

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        if self.event_log_enabled and bool(selected_pts_mask.any()):
            self.event_log.append({
                "kind": "birth",
                "type": "clone",
                "xyz": new_xyz.detach().cpu().numpy(),
                "parent_xyz": new_xyz.detach().cpu().numpy(),
            })

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
        outside_box=None,
        outside_budget=0.0,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._densify_exclude = None
        self._born_this_event = 0

        # 2a 干预（研究用）：框外高斯数已达预算比例后，本次 densify 不再选择
        # 框外高斯（打断框外 clone/split 放大，同时保留优化器迁移出框的自由）。
        if outside_budget > 0 and outside_box is not None:
            xyz = self.get_xyz
            outside = (
                (xyz[:, 0] < outside_box[0, 0])
                | (xyz[:, 0] > outside_box[1, 0])
                | (xyz[:, 1] < outside_box[0, 1])
                | (xyz[:, 1] > outside_box[1, 1])
                | (xyz[:, 2] < outside_box[0, 2])
                | (xyz[:, 2] > outside_box[1, 2])
            )
            if outside.sum() >= outside_budget * xyz.shape[0]:
                self._densify_exclude = outside

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density；原因码: 1=density, 2=bbox, 3=screen, 4=scale
        prune_mask = (self.get_density < min_density).squeeze()
        reason = torch.zeros(prune_mask.shape, dtype=torch.int8,
                             device=prune_mask.device)
        reason[prune_mask] = 1
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )
            reason = torch.where(prune_mask_xyz & (reason == 0),
                                 torch.tensor(2, dtype=reason.dtype,
                                              device=reason.device), reason)
            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            reason = torch.where(big_points_vs & (reason == 0),
                                 torch.tensor(3, dtype=reason.dtype,
                                              device=reason.device), reason)
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            reason = torch.where(big_points_ws & (reason == 0),
                                 torch.tensor(4, dtype=reason.dtype,
                                              device=reason.device), reason)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask, tag="prune", reasons=reason)

        # 2b 干预（研究用）：本次事件出生的子代（clone/split，位于张量末尾）
        # clamp 回重建框内。与全位置 clamp（Experiment F）不同：优化器仍可把
        # 任何高斯（含这些子代）移出框，只打断"出生即框外"的放大通道。
        if self.clamp_birth_box is not None and self._born_this_event > 0:
            lo, hi = self.clamp_birth_box
            with torch.no_grad():
                self._xyz[-self._born_this_event:].clamp_(min=lo, max=hi)

        # torch.cuda.empty_cache()

        return grads

    def long_axis_split(
        self,
        scores,
        budget,
        filter_mask,
        split_distance,
        density_reduction,
    ):
        """ImprovedGS 长轴分裂（LAS，移植自 XiaoBin2001/Improved-GS）。

        从 filter_mask 命中的候选中按 scores 做重要性抽样（multinomial）选出
        budget 个 parent，每个 parent 生成 2 个子代，沿其最长轴
        ±3·d·σ_long 确定性放置（不再随机三维散射）；长轴尺度 ×(1-d)、
        其余轴 ×√(1-d²)、密度 ×density_reduction。原版 3DGS split 无
        尺度资格（候选由调用方给定 filter_mask）。
        scores: [N] 1D 非负得分；filter_mask: [N] bool 候选资格。
        返回实际分裂的 parent 数。
        """
        if budget <= 0 or scores.numel() == 0 or not torch.any(filter_mask):
            return 0

        padded_importance = scores.detach().float().clamp_min(0)
        padded_importance[~filter_mask] = 0
        positive_count = int((padded_importance > 0).sum().item())
        if positive_count == 0:
            return 0

        budget = min(int(budget), positive_count)
        selected_indices = torch.multinomial(
            padded_importance, budget, replacement=False
        )
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=torch.bool)
        selected_pts_mask[selected_indices] = True

        stds = self.get_scaling[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        axis_mask = torch.zeros_like(stds, dtype=torch.bool).scatter(
            1, max_indices, True
        )
        axis_offsets = stds * axis_mask * 3.0 * float(split_distance)
        axis_offsets = torch.cat([axis_offsets, -axis_offsets], dim=0)

        rotation_mats = build_rotation(self._rotation[selected_pts_mask]).repeat(
            2, 1, 1
        )
        parent_xyz = self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_xyz = (
            torch.bmm(rotation_mats, axis_offsets.unsqueeze(-1)).squeeze(-1)
            + parent_xyz
        )

        split_distance_sq = float(split_distance) * float(split_distance)
        rate_w = max(1.0 - float(split_distance), 1e-6)
        rate_h = math.sqrt(max(1.0 - split_distance_sq, 1e-6))
        new_scales = (
            stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1)
            * rate_h
        )
        new_scaling = self.scaling_inverse_activation(new_scales)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * float(density_reduction)
        ).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(2)

        if self.event_log_enabled:
            self.event_log.append({
                "kind": "birth",
                "type": "split",
                "xyz": new_xyz.detach().cpu().numpy(),
                "parent_xyz": parent_xyz.detach().cpu().numpy(),
            })

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    2 * int(selected_pts_mask.sum().item()),
                    device=selected_pts_mask.device,
                    dtype=torch.bool,
                ),
            )
        )
        self.prune_points(prune_filter, tag="split_parent")
        return budget

    def densify_and_prune_improved(
        self,
        scores,
        grad_threshold,
        min_density,
        budget,
        iteration,
        densify_until_step,
        max_screen_size,
        max_scale,
        densify_scale_threshold,
        bbox=None,
        outside_box=None,
        outside_budget=0.0,
        use_las=True,
        split_distance=0.45,
        density_reduction=0.6,
        late_threshold_relax=1.5,
        eas_qualify_mask=None,
    ):
        """ImprovedGS 移植的预算式 densify（Growth Control + 去 clone）。

        与 legacy 的差异：
        1. 无 clone 分支（框外 clone 放大通道关闭；split 无尺度资格，小高斯
           也有分裂入口）；
        2. 每事件只新增 split_budget = min(budget, 候选数+N) − N 个（预算
           由调用方按 √progress 爬坡给出）；
        3. 候选 = 梯度≥阈值，可 OR 边缘资格 eas_qualify_mask（EAS qualify
           模式，打冷启动边缘带的低梯度缺口）；
        4. 得分 scores 用于候选重要性抽样（LAS 时）；None/窗口末尾回退为
           决策梯度，窗口末尾 100 步内若未满预算阈值 ÷1.5 放宽；
        5. 2a（框外预算）与 2b（出生 clamp）干预保持可用。
        返回决策梯度（xyz_gradient_accum/denom，postfix 重置前），与
        legacy densify_and_prune 一致。
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._densify_exclude = None
        self._born_this_event = 0

        # 2a 干预（研究用）：框外高斯数达预算比例后，本次 densify 不再选择
        # 框外高斯（打断框外 clone/split 放大，同时保留优化器迁移出框的自由）。
        if outside_budget > 0 and outside_box is not None:
            xyz = self.get_xyz
            outside = (
                (xyz[:, 0] < outside_box[0, 0])
                | (xyz[:, 0] > outside_box[1, 0])
                | (xyz[:, 1] < outside_box[0, 1])
                | (xyz[:, 1] > outside_box[1, 1])
                | (xyz[:, 2] < outside_box[0, 2])
                | (xyz[:, 2] > outside_box[1, 2])
            )
            if outside.sum() >= outside_budget * xyz.shape[0]:
                self._densify_exclude = outside

        late_densify = int(iteration) >= int(densify_until_step) - 100
        if scores is None:
            # 无 EAS 得分时回退决策梯度；EAS 模式下得分保持为 EAS
            # （论文式 7 的概率权重），晚段只放宽梯度阈值、不替换得分来源。
            scores = grads.squeeze(-1)
        if late_densify and self.get_xyz.shape[0] < budget:
            grad_threshold = grad_threshold / late_threshold_relax

        grad_qualifiers = grads.squeeze(-1) >= grad_threshold
        if eas_qualify_mask is not None:
            # 兼容保留（旧 qualify-mask 语义已废弃，调用方不再传入）
            grad_qualifiers = grad_qualifiers | eas_qualify_mask
        if self._densify_exclude is not None:
            grad_qualifiers = grad_qualifiers & ~self._densify_exclude

        total_candidates = int(grad_qualifiers.sum().item())
        current_points = int(self.get_xyz.shape[0])
        current_budget = min(int(budget), total_candidates + current_points)
        split_budget = current_budget - current_points
        if split_budget > 0:
            if use_las:
                self.long_axis_split(
                    scores,
                    split_budget,
                    grad_qualifiers,
                    split_distance,
                    density_reduction,
                )
            elif densify_scale_threshold:
                # 无 LAS 的对照：原版随机 split（仍为预算式选择）
                self.densify_and_split(
                    grads, grad_threshold, densify_scale_threshold
                )

        # Prune gaussians with too small density；原因码: 1=density, 2=bbox, 3=screen, 4=scale
        prune_mask = (self.get_density < min_density).squeeze()
        reason = torch.zeros(prune_mask.shape, dtype=torch.int8,
                             device=prune_mask.device)
        reason[prune_mask] = 1
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )
            reason = torch.where(prune_mask_xyz & (reason == 0),
                                 torch.tensor(2, dtype=reason.dtype,
                                              device=reason.device), reason)
            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            reason = torch.where(big_points_vs & (reason == 0),
                                 torch.tensor(3, dtype=reason.dtype,
                                              device=reason.device), reason)
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            reason = torch.where(big_points_ws & (reason == 0),
                                 torch.tensor(4, dtype=reason.dtype,
                                              device=reason.device), reason)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask, tag="prune", reasons=reason)

        # 2b 干预（研究用）：本次事件出生的子代（位于张量末尾）clamp 回重建框内。
        if self.clamp_birth_box is not None and self._born_this_event > 0:
            lo, hi = self.clamp_birth_box
            with torch.no_grad():
                self._xyz[-self._born_this_event:].clamp_(min=lo, max=hi)

        return grads


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # ImprovedGS absolute-gradient accumulation: per-axis |dL/du| instead of
        # the L2 norm, so each coordinate axis contributes equally (a dominant
        # gradient on one axis is no longer diluted by a squared-norm).
        self.xyz_gradient_accum[update_filter] += torch.abs(
            viewspace_point_tensor.grad[update_filter, :2]
        ).mean(dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_3d(self, worldspace_point_tensor, update_filter):
        # Same absolute-gradient convention for the 3D path (unused by the
        # current renderer, kept consistent).
        self.xyz_gradient_accum[update_filter] += torch.abs(
            worldspace_point_tensor.grad[update_filter, :3]
        ).mean(dim=-1, keepdim=True)
        self.denom[update_filter] += 1
