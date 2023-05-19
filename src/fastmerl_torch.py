"""
A MERL BRDF class
modified from the codebase of 
[Sztrajman, A., Rainer, G., Ritschel, T., and Weyrich, T. 2021. Neural BRDF Representation and Importance Sampling. Computer Graphics Forum 40, 6, 332–346.]

Re-implemented by PyTorch in order to:
1. support gradient computations
2. support 3D interpolation in the BRDF tensor. 
"""

import struct
import torch
import torch.nn.functional as F
from utils import PI
from utils import grid_sample_3d
from functools import partial


class Merl:
    sampling_theta_h = 90
    sampling_theta_d = 90
    sampling_phi_d = 180

    scale = torch.tensor([1.0 / 1500, 1.15 / 1500, 1.66 / 1500])

    def __init__(self, merl_file, device="cpu"):
        """
        Initialize and load a MERL BRDF file

        :param merl_file: The path of the file to load
        """
        with open(merl_file, "rb") as f:
            data = f.read()
            n = struct.unpack_from("3i", data)
            Merl.sampling_phi_d = n[2]
            length = Merl.sampling_theta_h * Merl.sampling_theta_d * Merl.sampling_phi_d
            if n[0] * n[1] * n[2] != length:
                raise IOError("Dimensions do not match")
            brdf = struct.unpack_from(
                str(3 * length) + "d", data, offset=struct.calcsize("3i")
            )

            self.brdf_tensor = torch.tensor(brdf, device=device).reshape(3, -1)
            # convert all invalid entries into 0
            self.mask = ~(self.brdf_tensor[0] < 0)
            self.brdf_tensor[self.brdf_tensor < 0] = 0.0
            Merl.scale = Merl.scale.to(device)

    def _filter_theta_h(theta_h):
        angle_range = PI / 2

        theta_h = torch.where(theta_h < 0, theta_h + angle_range, theta_h)
        theta_h = torch.where(theta_h > angle_range, theta_h - angle_range, theta_h)
        return theta_h

    def _filter_theta_d(theta_d):
        angle_range = PI / 2

        theta_d = torch.where(theta_d < 0, theta_d + angle_range, theta_d)
        theta_d = torch.where(theta_d > angle_range, theta_d - angle_range, theta_d)
        return theta_d

    def _filter_phi_d(phi_d):
        angle_range = 2 * PI

        phi_d = torch.where(phi_d < 0, phi_d + angle_range, phi_d)
        phi_d = torch.where(phi_d > angle_range, phi_d - angle_range, phi_d)

        phi_d = torch.where(phi_d >= PI, phi_d - PI, phi_d)
        return phi_d

    def eval_raw(self, theta_h, theta_d, phi_d):
        """
        Lookup the BRDF value for given half diff coordinates

        :param theta_h: half vector elevation angle in radians
        :param theta_d: diff vector elevation angle in radians
        :param phi_d: diff vector azimuthal angle in radians
        :return: A list of 3 elements giving the BRDF value for R, G, B in
        linear RGB
        """
        theta_h = Merl._filter_theta_h(torch.atleast_1d(theta_h))
        theta_d = Merl._filter_theta_d(torch.atleast_1d(theta_d))
        phi_d = Merl._filter_phi_d(torch.atleast_1d(phi_d))

        return self._eval_idx(
            Merl._theta_h_idx(theta_h),
            Merl._theta_d_idx(theta_d),
            Merl._phi_d_idx(phi_d),
        )

    def merl_lookup(merl_tensor, theta_h, theta_d, phi_d, scaling=True, higher=False):
        """
        lookup (3D interpolation) the BRDF tensor in the position (theta_h, theta_d, phi_d)
            merl_tensor: the BRDF tensor
            theta_h, theta_d, phi_d: the position to lookup
            scaling: do merl tonemapping if needed
            higher: indicate whether the higher gradients are needed, to select different implementations
        """

        theta_h = Merl._filter_theta_h(torch.atleast_1d(theta_h))
        theta_d = Merl._filter_theta_d(torch.atleast_1d(theta_d))
        phi_d = Merl._filter_phi_d(torch.atleast_1d(phi_d))

        # deal with the nonliearity mapping of theta_h
        idx_th = torch.sqrt(theta_h / (PI / 2) + 1e-8) * Merl.sampling_theta_h
        th_prev = Merl._theta_h_from_idx(torch.floor(idx_th))
        th_next = Merl._theta_h_from_idx(torch.floor(idx_th) + 1)
        idx_th_prev_normalized = torch.floor(idx_th) / (Merl.sampling_theta_h - 1)
        idx_th_next_normalized = (torch.floor(idx_th) + 1) / (Merl.sampling_theta_h - 1)
        idx_th_normalized = idx_th_prev_normalized + (theta_h - th_prev) / (
            th_next - th_prev
        ) * (idx_th_next_normalized - idx_th_prev_normalized)

        idx_td_normalized = (
            theta_d / (PI / 2) * Merl.sampling_theta_d / (Merl.sampling_theta_d - 1)
        )
        idx_pd_normalized = phi_d / PI * Merl.sampling_phi_d / (Merl.sampling_phi_d - 1)

        idx = torch.stack(
            [
                2 * (idx_pd_normalized - 0.5),
                2 * (idx_td_normalized - 0.5),
                2 * (idx_th_normalized - 0.5),
            ],
            dim=1,
        )

        if higher:
            interpolator = grid_sample_3d
        else:
            interpolator = partial(
                F.grid_sample,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )

        C = merl_tensor.shape[0]
        interpolated = interpolator(
            merl_tensor.reshape(
                1, C, Merl.sampling_theta_h, Merl.sampling_theta_d, Merl.sampling_phi_d
            ),
            idx.reshape(1, -1, 1, 1, 3),
        ).reshape(C, -1)

        if scaling:
            interpolated *= Merl.scale[..., None]

        return interpolated

    def eval_interp(self, theta_h, theta_d, phi_d):
        """
        Lookup the BRDF value for given half diff coordinates and perform an
        interpolation over theta_h, theta_d and phi_d

        :param theta_h: half vector elevation angle in radians
        :param theta_d: diff vector elevation angle in radians
        :param phi_d: diff vector azimuthal angle in radians
        :return: A list of 3 elements giving the BRDF value for R, G, B in
        linear RGB
        """
        return Merl.merl_lookup(self.brdf_tensor, theta_h, theta_d, phi_d)

    def _eval_idx(self, ith, itd, ipd):
        """
        Lookup the BRDF value for a given set of indexes
        :param ith: theta_h index
        :param itd: theta_d index
        :param ipd: phi_d index
        :return: A list of 3 elements giving the BRDF value for R, G, B in
        linear RGB
        """
        ind = ipd + Merl.sampling_phi_d * (itd + ith * Merl.sampling_theta_d)

        # TODO: type casting operation can be differentiable?
        ind = ind.to(torch.long)

        return Merl.scale[..., None] * self.brdf_tensor[:, ind]

    def _theta_h_from_idx(theta_h_idx):
        """
        Get the theta_h value corresponding to a given index

        :param theta_h_idx: Index for theta_h
        :return: A theta_h value in radians
        """
        ret_val = theta_h_idx / Merl.sampling_theta_h
        return ret_val * ret_val * PI / 2

    def _theta_h_idx(theta_h):
        """
        Get the index corresponding to a given theta_h value

        :param theta_h: Value for theta_h in radians
        :return: The corresponding index for the given theta_h
        """
        th = Merl.sampling_theta_h * torch.sqrt(theta_h / (PI / 2))

        return torch.clip(torch.floor(th), 0, Merl.sampling_theta_h - 1)

    def _theta_d_from_idx(theta_d_idx):
        """
        Get the theta_d value corresponding to a given index

        :param theta_d_idx: Index for theta_d
        :return: A theta_d value in radians
        """
        return theta_d_idx / Merl.sampling_theta_d * PI / 2

    def _theta_d_idx(theta_d):
        """
        Get the index corresponding to a given theta_d value

        :param theta_d: Value for theta_d in radians
        :return: The corresponding index for the given theta_d
        """
        td = Merl.sampling_theta_d * theta_d / (PI / 2)
        return torch.clip(torch.floor(td), 0, Merl.sampling_theta_d - 1)

    def _phi_d_from_idx(phi_d_idx):
        """
        Get the phi_d value corresponding to a given index

        :param phi_d_idx: Index for phi_d
        :return: A phi_d value in radians
        """

        return phi_d_idx / Merl.sampling_phi_d * PI

    def _phi_d_idx(phi_d):
        """
        Get the index corresponding to a given phi_d value

        :param theta_h: Value for phi_d in radians
        :return: The corresponding index for the given phi_d
        """
        pd = Merl.sampling_phi_d * phi_d / PI
        return torch.clip(torch.floor(pd), 0, Merl.sampling_phi_d - 1)

    def to_(self, device):
        self.brdf_tensor = self.brdf_tensor.to(device)
        Merl.scale = Merl.scale.to(device)
        return self
