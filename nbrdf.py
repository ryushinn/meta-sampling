import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import PI
import coords
from os.path import join as pjoin
from fastmerl_torch import Merl

_epsilon = 1e-6


def mean_absolute_logarithmic_error(y_true, y_pred):
    """
    the loss function used in our paper
    note that both y_true and y_pred are already cosine weighted
    """
    return torch.mean(torch.abs(torch.log(1 + y_true) - torch.log(1 + y_pred)))


def mean_cubic_root_error(y_true, y_pred):
    return torch.mean(
        torch.pow(torch.square(y_true - y_pred) + _epsilon, 1 / 3)
    )


def mean_log2_error(y_true, y_pred):
    return torch.mean(
        torch.log((y_true + _epsilon) / (y_pred + _epsilon)) ** 2
    )


def mean_log1_error(y_true, y_pred):
    return torch.mean(
        torch.log((y_true + _epsilon) / (y_pred + _epsilon)).abs()
    )


def brdf_to_rgb(rangles, brdf):
    """
    cosine weight brdf values
    """
    theta_h, theta_d, phi_d = torch.unbind(rangles, dim=1)

    # cos(wi)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
        torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz[:, None], 0, 1)
    return rgb


class MLP(torch.nn.Module):
    """
    Neural BRDF model
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=6, out_features=21, bias=True)
        self.fc2 = torch.nn.Linear(in_features=21, out_features=21, bias=True)
        self.fc3 = torch.nn.Linear(in_features=21, out_features=3, bias=True)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

        self.fc1.weight = torch.nn.Parameter(torch.zeros((6, 21)).uniform_(-0.05, 0.05).T,
                                             requires_grad=True)
        self.fc2.weight = torch.nn.Parameter(torch.zeros((21, 21)).uniform_(-0.05, 0.05).T,
                                             requires_grad=True)
        self.fc3.weight = torch.nn.Parameter(torch.zeros((21, 3)).uniform_(-0.05, 0.05).T,
                                             requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # additional relu is max() op as in code in nn.h
        x = F.relu(torch.exp(self.fc3(x)) - 1.0)
        return x


class phong(torch.nn.Module):
    """
    Phong BRDF model
    """

    def __init__(self):
        super(phong, self).__init__()
        self.factor_sum = torch.nn.Parameter(
            torch.randn(3)
        )
        self.factor_ratio = torch.nn.Parameter(
            torch.randn(3)
        )
        self.factor_q = torch.nn.Parameter(
            torch.randn(1)
        )

        self.register_buffer('_reflect', torch.tensor(
            [-1.0, -1.0, 1.0]), persistent=False)

    def forward(self, x):
        sum = torch.sigmoid(self.factor_sum)
        ratio = torch.sigmoid(self.factor_ratio)
        q = torch.exp(self.factor_q)

        kd = sum * ratio
        ks = sum * (1 - ratio)

        diffuse = kd / PI

        wi, wo = coords.hd_to_io(
            *torch.split(x, [3, 3], dim=1)
        )
        r = wi * self._reflect
        cosine = torch.einsum("nj,nj->n", r, wo)
        cosine = F.relu(cosine) + 1e-5

        specular = torch.outer((cosine ** q), ks * (2 + q) / (2 * PI))

        return diffuse + specular


class cook_torrance(torch.nn.Module):
    """
    Cook Torrance model
    """

    def __init__(self):
        super(cook_torrance, self).__init__()
        self.factor_kd = torch.nn.Parameter(
            torch.randn(3)
        )
        self.factor_ks = torch.nn.Parameter(
            torch.randn(3)
        )
        self.factor_alpha = torch.nn.Parameter(
            torch.randn(1)
        )
        self.factor_f0 = torch.nn.Parameter(
            torch.randn(1)
        )

    def forward(self, x):
        kd = torch.sigmoid(self.factor_kd)
        ks = torch.sigmoid(self.factor_ks)
        alpha = torch.sigmoid(self.factor_alpha)
        f0 = torch.sigmoid(self.factor_f0)

        diffuse = kd / PI

        half, diff = torch.split(x, [3, 3], dim=1)
        wi, wo = coords.hd_to_io(
            half, diff
        )

        cos_theta_h = half[:, 2]
        tan_theta_h2 = (half[:, 0] ** 2 + half[:, 1] **
                        2) / (cos_theta_h ** 2 + _epsilon)
        cos_theta_d = diff[:, 2]
        # torch.clamp avoids numerical issues
        cos_theta_i = torch.clamp(wi[:, 2], min=0, max=1)
        cos_theta_o = torch.clamp(wo[:, 2], min=0, max=1)
        alpha2 = alpha ** 2

        D = torch.exp(- tan_theta_h2 / (alpha2 + _epsilon)) / \
            (alpha2 * cos_theta_h ** 4 + _epsilon)

        G = torch.clamp(
            2 * cos_theta_h *
            torch.minimum(cos_theta_i, cos_theta_o) / (cos_theta_d + _epsilon),
            max=1.0
        )

        F = f0 + (1 - f0) * (1 - cos_theta_d) ** 5

        specular = torch.outer(
            D * G * F / (PI * cos_theta_i * cos_theta_o + _epsilon), ks)

        return diffuse + specular


class _PCA(torch.nn.Module):
    """
    PCA BRDF model (base class)
    The implementation is based on [Nielsen, J.B., Jensen, H.W., and Ramamoorthi, R. 2015. On optimal, minimal BRDF sampling for reflectance acquisition. ACM Transactions on Graphics 34, 6, 1–11.]
    and their codebase https://brdf.compute.dtu.dk/#citation
    """

    def __init__(self, precomputed_path, basis_num=240):
        super(_PCA, self).__init__()
        # register all precomputed components
        # ** Note that components are already MERL tonemapped **
        self.register_buffer(
            "maskMap", torch.tensor(
                np.load(pjoin(precomputed_path, "MaskMap.npy"))),
            persistent=False
        )
        self.register_buffer(
            "cosMap", torch.tensor(
                np.load(pjoin(precomputed_path, "CosineMap.npy")), dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "median", torch.tensor(
                np.load(pjoin(precomputed_path, "Median.npy")), dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "relativeOffset", torch.tensor(
                np.load(pjoin(precomputed_path, "RelativeOffset.npy")), dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "Q", torch.tensor(np.load(pjoin(
                precomputed_path, "ScaledEigenvectors.npy")), dtype=torch.float32)[:, 0:basis_num],
            persistent=False
        )
        # convert all components from nielsen's format to our format
        oldMask = self.maskMap
        self.maskMap = _PCA.reshape(oldMask.reshape(-1, 1)).flatten()
        self.cosMap = _PCA.reshape(_PCA.unmask(self.cosMap, oldMask))[
            self.maskMap, :]
        self.median = _PCA.reshape(_PCA.unmask(self.median, oldMask))[
            self.maskMap, :]
        self.relativeOffset = _PCA.reshape(_PCA.unmask(
            self.relativeOffset, oldMask))[self.maskMap, :]
        self.Q = _PCA.reshape(_PCA.unmask(self.Q, oldMask))[self.maskMap, :]

        # the number of basis
        self.n = basis_num

    def unmap(mappedRecon, median, cosMap):
        eps = 1e-3
        unmappedRecon = (torch.exp(mappedRecon) *
                         (median + eps) - eps) / cosMap
        return unmappedRecon

    def unmask(maskedRecon, maskMap):
        unmaskedRecon = torch.zeros(
            maskMap.shape[0], maskedRecon.shape[1]).to(maskedRecon.device)
        unmaskedRecon[maskMap, :] = maskedRecon
        return unmaskedRecon

    def reshape(BRDFTensor):
        # reshape nielsen convention [180 (pd) x 90 (th) x 90 (td)] x k (channel)
        # to fastmerl convention [90 (th) x 90 (td) x 180 (pd)] x k (channel)
        k = BRDFTensor.shape[1]
        BRDFTensor = BRDFTensor.reshape(180, 90, 90, k)
        BRDFTensor = BRDFTensor.permute(1, 2, 0, 3)
        return BRDFTensor.reshape(-1, k)

    def getBRDFTensor(self, c):
        """
        Given the weights/coefficients "c", reconstruct the BRDF tensor using basis
        """
        # from c to mapped BRDF tensor
        mappedRecon = self.Q @ c + self.relativeOffset

        # from mapped to unmapped
        maskedRecon = _PCA.unmap(mappedRecon, self.median, self.cosMap)

        # unmask
        recon = _PCA.unmask(maskedRecon, self.maskMap)

        return recon


class PCA(_PCA):
    """
    derived PCA model that fits the weights using gradient-based optimization
    """
    def __init__(self, precomputed_path, basis_num=240):
        super(PCA, self).__init__(precomputed_path, basis_num)
        # weights
        self.c = torch.nn.Parameter(torch.zeros(self.n, 3))

    def forward(self, x):
        # Note that for PCA model, x is 3D Rusink angles

        # from c to the BRDF tensor
        recon = super(PCA, self).getBRDFTensor(self.c)

        # lookup
        theta_h, theta_d, phi_d = torch.unbind(x, dim=1)
        return Merl.merl_lookup(recon.T, theta_h, theta_d, phi_d, scaling=False, higher=True).T

    def getBRDFTensor(self):
        BRDFTensor = super(PCA, self).getBRDFTensor(self.c)

        BRDFTensor[~self.maskMap, :] = -1
        return BRDFTensor.cpu().numpy()


class PCARR(_PCA):
    """
    derived PCA model that fits the weights using Ridge Regression (RR) as proposed in [Nielsen, J.B., Jensen, H.W., and Ramamoorthi, R. 2015. On optimal, minimal BRDF sampling for reflectance acquisition. ACM Transactions on Graphics 34, 6, 1–11.]
    """
    def __init__(self, precomputed_path, basis_num=240):
        super(PCARR, self).__init__(precomputed_path, basis_num)

    def forward(self, c, rangles):
        # from c to the BRDF tensor
        recon = self.getBRDFTensor(c)

        # lookup
        theta_h, theta_d, phi_d = torch.unbind(rangles, dim=1)
        return Merl.merl_lookup(recon.T, theta_h, theta_d, phi_d, scaling=False, higher=True).T

    def RR(self, rangles, observations):
        """
        analytically solve the weights/coefficients from observations using RR
        """
        eta = 40
        th, td, pd = torch.unbind(rangles, dim=1)
        stacks = PCA.unmask(
            torch.hstack([self.Q, self.median, self.relativeOffset]),
            self.maskMap
        )

        Q, median, relativeOffset = torch.split(
            Merl.merl_lookup(stacks.T, th, td, pd, scaling=False).T,
            [self.n, 1, 1], dim=1
        )
        ph = torch.zeros_like(th).to(th.device)
        wi, wo = coords.hd_to_io_sph(torch.stack(
            [th, ph], dim=1), torch.stack([td, pd], dim=1))
        cosMap = torch.cos(wi[:, [0]]) * torch.cos(wo[:, [0]])
        cosMap[cosMap < 0.0] = 0.0  # max(cos, 0.0)

        mappedObs = torch.log((observations * cosMap + 1e-4) / (median + 1e-4))
        b = mappedObs - relativeOffset
        U, s, Vt = torch.linalg.svd(Q, full_matrices=False)
        sinv = torch.diag(s / (s * s + eta))

        c = Vt.T @ sinv @ U.T @ b

        return c
