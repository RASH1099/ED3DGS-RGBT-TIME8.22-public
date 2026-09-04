import torch
import torch.nn as nn
# from .utils.lie_group_helper import make_c2w
from scipy.spatial.transform import Rotation as RotLib


def SO3_to_quat(R):
    """
    :param R:  (N, 3, 3) or (3, 3) np
    :return:   (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat):
    """
    :param quat:    (N, 4, ) or (4, ) np
    :return:        (N, 3, 3) or (3, 3) np
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w

def pack_colmajor12(M34: torch.Tensor) -> torch.Tensor:
    """
    (3,4) -> (12,) 列优先打平，匹配 CUDA 的 dL_dTcw[0..11] 索引：
    idx:  0  1  2  | 3  4  5 | 6  7  8 | 9 10 11
        W00 W10 W20 W01 W11 W21 W02 W12 W22  tx ty tz
    """
    return M34.t().reshape(-1).contiguous()

def unpack_colmajor12(T12: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    仅供调试：把 (12,) 还原为 (R(3,3), t(3,))
    """
    M34 = T12.view(4, 3).t().contiguous()  # (3,4)
    R = M34[:, :3].contiguous()
    t = M34[:, 3].contiguous()
    return R, t
class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, learn_s=False,init_c2w=None, compose="right",device: str = "cuda",
        dtype: torch.dtype = torch.float32,):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        # self.init_c2w = None
        # if init_c2w is not None:
        #     self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.compose = compose
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), device=device,dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), device=device,dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

        if learn_s:
            # 初始化为1.0，形状为(num_cams, 1)，保证每个相机有单独缩放
            self.log_s = nn.Parameter(torch.ones(size=(num_cams, 1), device=device,dtype=torch.float32), requires_grad=True)
        else:
            self.log_s = None

        if init_c2w is not None:
            init_c2w = init_c2w.to(device=device, dtype=dtype)
            self.register_buffer("init_c2w", init_c2w)                  # (N,4,4)
            self.register_buffer("init_Tcw", torch.linalg.inv(init_c2w))# (N,4,4)
        else:
            self.init_c2w = None
            self.init_Tcw = None

    # def forward(self, cam_id):
    #     r = self.r[cam_id] # (3, ) axis-angle
    #     t = self.t[cam_id]  # (3, )
    #     if self.s is not None:
    #         t = self.s[cam_id] * t
    #     c2w = make_c2w(r, t)  # (4, 4)

    #     # learn a delta pose between init pose and target pose, if a init pose is provided
    #     if self.init_c2w is not None:
    #         c2w = c2w @ self.init_c2w[cam_id]
    #     # if cam_id == 1:
    #     #     print(c2w)
    #     return c2w
    
    def _delta_R_t(self, cam_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.r[cam_id]                  # (3,)
        t = self.t[cam_id] if self.t is not None else torch.zeros(3, device=r.device, dtype=r.dtype)
        if self.log_s is not None:
            t = torch.exp(self.log_s[cam_id, 0]) * t
        R = Exp(r)                          # (3,3) 正交
        return R, t
    def reset_delta(self, cam_id: int):
        """ 重置坏视角的增量位姿 (delta) """
        self.r.data[cam_id] = torch.zeros(3, device=self.r.device)  # 重置旋转
        self.t.data[cam_id] = torch.zeros(3, device=self.t.device)  # 重置平移
        if self.log_s is not None:
            self.log_s.data[cam_id] = torch.ones(1, device=self.log_s.device)  # 重置缩放
    def forward(self, cam_id: int, out: str = "Tcw12") -> torch.Tensor:
        """
        out:
          - "Tcw12": (12,) 列优先 Tcw，直接给 CUDA/rasterizer
          - "c2w12": (12,) 列优先 c2w（如需要）
          - "matrix": (4,4) c2w 矩阵
        """
        R, t = self._delta_R_t(cam_id)

        if self.init_c2w is None:
            # 无初值：直接输出 delta
            if out == "matrix":
                M = torch.eye(4, dtype=R.dtype, device=R.device)
                M[:3, :3] = R
                M[:3, 3]  = t
                return M

            if out == "c2w12":
                M34 = torch.cat([R, t.unsqueeze(1)], dim=1)    # (3,4)
                return pack_colmajor12(M34)

            # out == "Tcw12"
            M34 = self._delta34_inv(R, t)                       # (3,4)
            return pack_colmajor12(M34)
        # 有初值
        if self.compose == "right":
            # c2w = init @ delta  =>  Tcw = delta^{-1} @ Tcw0
            # R0 = self.init_Tcw[cam_id][:3, :3]
            # t0 = self.init_Tcw[cam_id][:3, 3]
            R0 = self.init_c2w[cam_id][:3, :3]
            t0 = self.init_c2w[cam_id][3, :3]
            RT = R.t()
            Rp = RT @ R0
            tp = RT @ (t0 - t)
            if out == "matrix":
                # 构 c2w 以防你需要
                c2w = torch.eye(4, dtype=R.dtype, device=R.device)
                c2w[:3, :3] = self.init_c2w[cam_id][:3, :3] @ R
                c2w[:3, 3]  = self.init_c2w[cam_id][:3, :3] @ t + self.init_c2w[cam_id][:3, 3]
                return c2w
            if out == "c2w12":
                # c2w 的 (3,4)
                M34 = torch.cat([self.init_c2w[cam_id][:3, :3] @ R,
                                 (self.init_c2w[cam_id][:3, :3] @ t + self.init_c2w[cam_id][:3, 3]).unsqueeze(1)], dim=1)
                return pack_colmajor12(M34)
            # 默认：Tcw12（推荐给 CUDA）
            M34 = torch.cat([Rp, tp.unsqueeze(1)], dim=1)  # (3,4)
            return pack_colmajor12(M34)
        
      
    # def param_groups(self, base_lr: float):
    #     groups = []
    #     if self.r.requires_grad: groups.append({"params": [self.r], "lr": base_lr*0.1, "name":"pose_r"})
    #     if self.t.requires_grad: groups.append({"params": [self.t], "lr": base_lr*1.0, "name":"pose_t"})
    #     if self.log_s is not None and self.log_s.requires_grad:
    #         groups.append({"params": [self.log_s], "lr": base_lr*0.3, "name":"pose_log_s"})
    #     return groups
    
    def pose_param_groups(self, lr_r: float | None, lr_t: float | None, lr_s: float | None,
                          prefix: str = ""):
        """
        返回若干优化组：
        - lr_* 为 None：不返回对应组（彻底冻结）
        - lr_* 为 0：返回组但学习率为 0（也等于冻结，但方便你后面 scheduler 打开）
        - prefix: 组名前缀（如 "thermal_" → "thermal_pose_r"）
        """
        groups = []

        # 旋转：6D 优先，否则轴角
        rot_param = self.r
        if lr_r is not None and rot_param is not None and rot_param.requires_grad:
            groups.append({"params": [rot_param], "lr": lr_r, "name": f"{prefix}pose_r"})

        # 平移
        if lr_t is not None and self.t.requires_grad:
            groups.append({"params": [self.t], "lr": lr_t, "name": f"{prefix}pose_t"})

        # 可选尺度
        if (lr_s is not None) and (self.log_s is not None) and isinstance(self.log_s, torch.nn.Parameter) and self.log_s.requires_grad:
            groups.append({"params": [self.log_s], "lr": lr_s, "name": f"{prefix}pose_log_s"})

        return groups
    
    @torch.no_grad()
    def update_init(self, cam_id: int, new_c2w: torch.Tensor, reset_delta: bool = False):
        """
        用 recenter 后的 RGB c2w 作为新的基准：
        - 更新 init_c2w[cam_id] 和 init_Tcw[cam_id]
        - 可选：清零该相机的 ΔR, Δt, log_s
        """
        assert new_c2w.shape == (4, 4)
        new_c2w = new_c2w.to(device=self.init_c2w.device, dtype=self.init_c2w.dtype)

        # 覆盖 buffer
        self.init_c2w[cam_id].copy_(new_c2w)
        self.init_Tcw[cam_id].copy_(torch.linalg.inv(new_c2w))

        if reset_delta:
            self.reset_delta(cam_id)

    @torch.no_grad()
    def rebase_inits(self, cam_ids: list[int], new_c2w_batch: torch.Tensor, reset_delta: bool = True):
        """
        批量更新基准 init，并可选清零 delta
        - cam_ids: 要更新的相机 id 列表
        - new_c2w_batch: (K, 4, 4) 与 cam_ids 对齐
        """
        assert new_c2w_batch.shape[0] == len(cam_ids) and new_c2w_batch.shape[1:] == (4, 4)
        new_c2w_batch = new_c2w_batch.to(device=self.init_c2w.device, dtype=self.init_c2w.dtype)

        for i, cam_id in enumerate(cam_ids):
            self.init_c2w[cam_id].copy_(new_c2w_batch[i])
            self.init_Tcw[cam_id].copy_(torch.linalg.inv(new_c2w_batch[i]))
            if reset_delta:
                self.reset_delta(cam_id)