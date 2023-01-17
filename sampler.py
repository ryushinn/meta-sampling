import torch
import torch.nn as nn
import torch.nn.functional as F
import coords
from utils import PI
from fastmerl_torch import Merl


def _sample_on_merl(brdf, theta_h, theta_d, phi_d):
    """
    evaluate the given BRDF at a given position (theta_h, theta_d, phi_d)
        return:
            rangles: Rusinkiewicz angular coordinate
            rvectors: Rusinkiewicz xyz coordiante
            brdf_vals: BRDF values
    """

    hx, hy, hz, dx, dy, dz = coords.rangles_to_rvectors(
        theta_h, theta_d, phi_d)

    # nsamples x 3
    rangles = torch.stack([theta_h, theta_d, phi_d], dim=1)
    # nsamples x 6
    rvectors = torch.stack([hx, hy, hz, dx, dy, dz], dim=1)
    # nsamples x 3
    brdf_vals = brdf.eval_interp(theta_h, theta_d, phi_d).T

    return rangles, rvectors, brdf_vals


def sample_on_merl(brdf, sampler, nsamples):
    """
    generate N samples using the given sampler, and use them to evaluate the given BRDF.
        brdf: the BRDF to be evaluated
        sampler: the specified sampler
        nsamples: the number of generated samples
    """
    theta_h, _, theta_d, phi_d = sampler.generate(nsamples)
    rangles, rvectors, brdf_vals = _sample_on_merl(
        brdf, theta_h, theta_d, phi_d)

    return rangles, rvectors, brdf_vals


def sample_on_merl_with_rejection(brdf, sampler, nsamples):
    """
    generate N valid samples using the given sampler, and use them to evaluate the given BRDF.
    Samples are guaranteed to be valid by rejecting those invalid samples and resampling, iteratively, till all N samples are valid.
        brdf: the BRDF to be evaluated
        sampler: the specified sampler
        nsamples: the number of generated valid samples
    """
    theta_h, _, theta_d, phi_d = sampler.generate(nsamples)
    rangles, rvectors, brdf_vals = _sample_on_merl(
        brdf, theta_h, theta_d, phi_d)

    # filter out invalid directions
    # TODO: detect invalid samples using rangles instead of brdf_vals
    valid_idx = torch.any(brdf_vals != 0., dim=1)
    rangles = rangles[valid_idx, :]
    rvectors = rvectors[valid_idx, :]
    brdf_vals = brdf_vals[valid_idx, :]

    n_invalid = nsamples - valid_idx.sum()
    if n_invalid > 0:
        # print(f"append another {n_invalid} samples")
        a_rangles, a_rvectors, a_brdf_vals = sample_on_merl_with_rejection(
            brdf, sampler, n_invalid)
        rangles = torch.vstack([rangles, a_rangles])
        rvectors = torch.vstack([rvectors, a_rvectors])
        brdf_vals = torch.vstack([brdf_vals, a_brdf_vals])

    return rangles, rvectors, brdf_vals


"""
Following are samplers used in our paper

Each samplers is responsible of producing ** Rusinkiewicz half and diff ** samples
in theta-phi parameterization (4D), based on its own rules (e.g. by some distribution)

theta should be within [0, pi / 2];
phi should be within [0, pi * 2];

"""


class uniform_sampler:
    """
    generate uniform samples, or quasirandom samples (Sobol Sequence) if quasi = True
    """
    def __init__(self, device='cpu', quasi=False):
        self.device = device
        self.quasi = quasi
        if self.quasi:
            self.sobolEngine = torch.quasirandom.SobolEngine(
                dimension=4, scramble=True, seed=4
            )

    def generate(self, nsamples):
        device = self.device

        if self.quasi:
            thphtdpd = self.sobolEngine.draw(n=nsamples)
        else:
            thphtdpd = torch.rand(nsamples, 4)

        thphtdpd *= torch.tensor(
            [[PI / 2, PI * 2, PI / 2, PI * 2]]
        )

        th, ph, td, pd = torch.unbind(thphtdpd, dim=1)

        return th.to(device), ph.to(device), \
            td.to(device), pd.to(device)

    def to(self, device):
        self.device = device
        return self


class uniform_sampler_preloaded:
    """
    generate uniform samples, or quasirandom samples (Sobol Sequence) if quasi == True.
    Note that,
    1. All samples are generated and loaded in memory at once when initialized
    2. If reject == True, all samples are ensured to be valid in Rusinkiewicz space
    """
    def __init__(self, device='cpu', n_loaded=2500000, reject=False, quasi=False):
        self.device = device
        splr = uniform_sampler(quasi=quasi)
        if not reject:
            self._loaded = torch.stack(splr.generate(n_loaded), dim=1)
        else:
            n = 0
            _loaded = []
            while n != n_loaded:
                th, ph, td, pd = splr.generate(n_loaded - n)
                wi, wo = coords.hd_to_io(
                    torch.stack(coords.sph2xyz(1, th, ph), dim=1),
                    torch.stack(coords.sph2xyz(1, td, pd), dim=1)
                )
                valid_idx = torch.logical_and(wi[:, 2] > 0, wo[:, 2] > 0)
                if valid_idx.sum() != 0:
                    n += valid_idx.sum()
                    _loaded.append(torch.stack(
                        [th[valid_idx], ph[valid_idx],
                         td[valid_idx], pd[valid_idx]],
                        dim=1
                    ))
            self._loaded = torch.cat(_loaded, dim=0)
        self._n_loaded = n_loaded

        self.counter = 0

    def shuffle(self):
        p = torch.randperm(self._n_loaded)
        self._loaded = self._loaded[p, :]

    def generate(self, nsamples):
        device = self.device

        if self.counter + nsamples > self._n_loaded:
            self.shuffle()
            self.counter = 0

        # sampled_ind = torch.multinomial(torch.ones(self._n_loaded), nsamples)
        # sampled_ind = torch.randint(high=self._n_loaded, size=(nsamples, ))
        sampled_ind = range(self.counter, self.counter + nsamples)
        theta_h, phi_h, theta_d, phi_d = torch.unbind(
            self._loaded[sampled_ind, :],
            dim=1
        )

        self.counter += nsamples

        return theta_h.to(device), phi_h.to(device), \
            theta_d.to(device), phi_d.to(device)

    def to(self, device):
        self.device = device
        return self


class inverse_transform_sampler:
    """
    generate 3D samples proportional to one given 3D distribution, which is represented by a 3D tensor.
    """
    def __init__(self, target3D, device='cpu'):
        self.target3D = target3D
        # marginal distributions
        self.target1D = torch.sum(self.target3D, dim=(1, 2))
        self.target2D = torch.sum(self.target3D, dim=(2, ))
        self.device = device

    def generate(self, nsamples):
        device = self.device
        x = torch.multinomial(self.target1D, nsamples, replacement=True)
        y = torch.multinomial(self.target2D[x], 1, replacement=True).flatten()
        z = torch.multinomial(
            self.target3D[x, y], 1, replacement=True).flatten()

        x = Merl._theta_h_from_idx(x)
        y = Merl._theta_d_from_idx(y)
        z = Merl._phi_d_from_idx(z)

        return x.to(device), None, y.to(device), z.to(device)

    def to(self, device):
        self.device = device
        return self


class trainable_sampler_det(nn.Module):
    """
    another offline/preloaded sampler, akin to uniform_sampler_preloaded
    But samples are trainable (requires_grad=True)

    Please refer to our paper for implementation details
    """
    def __init__(self, n_fixed, quasi_init=False):
        super(trainable_sampler_det, self).__init__()

        self.n_fixed = n_fixed
        self.register_buffer(
            'scale',
            torch.tensor([PI / 2, PI * 2]),
            persistent=False
        )

        # initialize the sampler with directions uniformly on hd(theta-phi) space with rejection
        # in order to ensure as less as much directions that are initially placed in the invalid area
        preload_sampler = uniform_sampler_preloaded(
            n_loaded=n_fixed, reject=True, quasi=quasi_init
        )
        half, diff = torch.split(preload_sampler._loaded, [2, 2], dim=1)
        half_init, diff_init = half / self.scale, diff / self.scale

        self.factor_h = nn.Parameter(
            torch.logit(half_init, eps=1e-6)
        )
        self.factor_d = nn.Parameter(
            torch.logit(diff_init, eps=1e-6)
        )

        self.counter = 0

    def load_sampler(self, other_splr):
        other_n_fixed = other_splr.n_fixed
        # assert other_n_fixed == self.n_fixed // 2
        with torch.no_grad():
            self.factor_h[:other_n_fixed, :] = other_splr.factor_h
            self.factor_d[:other_n_fixed, :] = other_splr.factor_d

    def generate(self, nsamples):
        if self.counter + nsamples > self.n_fixed:
            self.counter = 0

        fh = self.factor_h[self.counter:self.counter+nsamples, ...]
        fd = self.factor_d[self.counter:self.counter+nsamples, ...]
        half = torch.sigmoid(fh) * self.scale
        diff = torch.sigmoid(fd) * self.scale

        theta_h, phi_h = torch.unbind(half, dim=1)
        theta_d, phi_d = torch.unbind(diff, dim=1)

        self.counter += nsamples
        return theta_h, phi_h, theta_d, phi_d
