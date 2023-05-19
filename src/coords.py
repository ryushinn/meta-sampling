import torch
from utils import PI


def bdot(a, b):
    """
    dot product in batch
    """
    return torch.einsum("bi,bi->b", a, b)


def normalize(v):
    """
    normalize a list of vectors "v"
    """
    return v / torch.norm(v, dim=1, keepdim=True)


def rotate_vector(v, axis, angle):
    """
    rotate "v" by "angle" along "axis" element-wise
        v: a list of vectors
        axis: a list of axes
        angle: a list of angles
    """
    sin_vals = torch.sin(angle).reshape(-1, 1)
    cos_vals = torch.cos(angle).reshape(-1, 1)
    return (
        v * cos_vals
        + axis * bdot(axis, v).reshape(-1, 1) * (1 - cos_vals)
        + torch.cross(axis, v, dim=-1) * sin_vals
    )


def io_to_hd_sph(wi, wo):
    """
    convert <wi, wo> (in spherical-coordinate) to <half, diff> (in spherical-coordinate)
    """
    theta_i, phi_i = torch.unbind(wi, dim=1)
    theta_o, phi_o = torch.unbind(wo, dim=1)

    ix, iy, iz = sph2xyz(1, theta_i, phi_i)
    ox, oy, oz = sph2xyz(1, theta_o, phi_o)

    half, diff = io_to_hd(
        torch.stack([ix, iy, iz], dim=1), torch.stack([ox, oy, oz], dim=1)
    )

    hx, hy, hz = torch.unbind(half, dim=1)
    dx, dy, dz = torch.unbind(diff, dim=1)

    _, theta_h, phi_h = xyz2sph(hx, hy, hz)
    _, theta_d, phi_d = xyz2sph(dx, dy, dz)

    return torch.stack([theta_h, phi_h], dim=1), torch.stack([theta_d, phi_d], dim=1)


def io_to_hd(wi, wo):
    """
    convert <wi, wo> (in xyz-coordinate) to <half, diff> (in xyz-coordinate)
    """
    # compute halfway vector
    half = normalize(wi + wo)
    r_h, theta_h, phi_h = xyz2sph(*torch.unbind(half, dim=1))

    # compute diff vector
    device = wi.device
    # # 1. by rotate computation
    # bi_normal = torch.tile(torch.tensor([0.0, 1.0, 0.0], device=device), (wi.size(0), 1))
    # normal = torch.tile(torch.tensor([0.0, 0.0, 1.0], device=device), (wi.size(0), 1))
    # tmp = rotate_vector(wi, normal, -phi_h)
    # diff = rotate_vector(tmp, bi_normal, -theta_h)

    # 2. by matrix computation
    row1 = torch.stack(
        [
            torch.cos(theta_h) * torch.cos(phi_h),
            torch.cos(theta_h) * torch.sin(phi_h),
            -torch.sin(theta_h),
        ],
        dim=0,
    )
    row2 = torch.stack(
        [-torch.sin(phi_h), torch.cos(phi_h), torch.zeros(wi.size(0), device=device)],
        dim=0,
    )
    row3 = torch.stack(
        [
            torch.sin(theta_h) * torch.cos(phi_h),
            torch.sin(theta_h) * torch.sin(phi_h),
            torch.cos(theta_h),
        ],
        dim=0,
    )
    mat = torch.stack([row1, row2, row3], dim=0)
    mat.to(device)

    diff = torch.einsum("ijn,nj->ni", mat, wi)

    return half, diff


def hd_to_io_sph(half, diff):
    """
    convert <half, diff> (in spherical-coordinate) to <wi, wo> (in spherical-coordinate)
    """
    theta_h, phi_h = torch.unbind(half, dim=1)
    theta_d, phi_d = torch.unbind(diff, dim=1)

    hx, hy, hz = sph2xyz(1, theta_h, phi_h)
    dx, dy, dz = sph2xyz(1, theta_d, phi_d)

    wi, wo = hd_to_io(
        torch.stack([hx, hy, hz], dim=1), torch.stack([dx, dy, dz], dim=1)
    )

    ix, iy, iz = torch.unbind(wi, dim=1)
    ox, oy, oz = torch.unbind(wo, dim=1)

    _, theta_i, phi_i = xyz2sph(ix, iy, iz)
    _, theta_o, phi_o = xyz2sph(ox, oy, oz)

    return torch.stack([theta_i, phi_i], dim=1), torch.stack([theta_o, phi_o], dim=1)


def hd_to_io(half, diff):
    """
    convert <half, diff> (in xyz-coordinate) to <wi, wo> (in xyz-coordinate)
    """
    r_h, theta_h, phi_h = xyz2sph(*torch.unbind(half, dim=1))

    # compute wi vector
    device = half.device
    # # 1. by rotate computations
    # y_axis = torch.tile(torch.tensor([0.0, 1.0, 0.0], device=device), (half.size(0), 1))
    # z_axis = torch.tile(torch.tensor([0.0, 0.0, 1.0], device=device), (half.size(0), 1))
    # tmp = rotate_vector(diff, y_axis, theta_h)
    # wi = normalize(rotate_vector(tmp, z_axis, phi_h))

    # 2. by matrix computations
    row1 = torch.stack(
        [
            torch.cos(phi_h) * torch.cos(theta_h),
            -torch.sin(phi_h),
            torch.cos(phi_h) * torch.sin(theta_h),
        ],
        dim=0,
    )
    row2 = torch.stack(
        [
            torch.sin(phi_h) * torch.cos(theta_h),
            torch.cos(phi_h),
            torch.sin(phi_h) * torch.sin(theta_h),
        ],
        dim=0,
    )
    row3 = torch.stack(
        [
            -torch.sin(theta_h),
            torch.zeros(half.size(0), device=device),
            torch.cos(theta_h),
        ],
        dim=0,
    )
    mat = torch.stack([row1, row2, row3], dim=0)
    mat.to(device)
    wi = torch.einsum("ijn,nj->ni", mat, diff)

    wo = normalize((2 * bdot(wi, half)[..., None] * half - wi))

    return wi, wo


def xyz2sph(x, y, z):
    """
    convert xyz-coordinate to spherical-coordinate
    """
    r2_xy = x**2 + y**2
    r = torch.sqrt(r2_xy + z**2)
    theta = torch.atan2(torch.sqrt(r2_xy), z)
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2 * PI, phi)
    return r, theta, phi


def sph2xyz(r, theta, phi):
    """
    convert spherical-coordinate to xyz-coordinate
    """
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z


def rangles_to_rvectors(theta_h, theta_d, phi_d):
    """
    convert <half, diff> (in spherical-coordinate) to <half, diff> (in xyz-coordinate)
    # assume phi_h = 0
    """

    hx = torch.sin(theta_h) * 1.0  # cos(0.0)
    hy = torch.sin(theta_h) * 0.0  # sin(0.0)
    hz = torch.cos(theta_h)
    dx = torch.sin(theta_d) * torch.cos(phi_d)
    dy = torch.sin(theta_d) * torch.sin(phi_d)
    dz = torch.cos(theta_d)
    return hx, hy, hz, dx, dy, dz


def rvectors_to_rangles(hx, hy, hz, dx, dy, dz):
    """
    convert <half, diff> (in xyz-coordinate) to <half, diff> (in spherical-coordinate)
    # assume phi_h = 0
    """

    theta_h = torch.arctan2(torch.sqrt(hx**2 + hy**2), hz)
    theta_d = torch.arctan2(torch.sqrt(dx**2 + dy**2), dz)
    phi_d = torch.arctan2(dy, dx)
    phi_d = torch.where(phi_d < 0, phi_d + 2 * PI, phi_d)
    return theta_h, theta_d, phi_d


# def rsph_to_rvectors(half_sph, diff_sph):
#     hx, hy, hz = sph2xyz(*half_sph)
#     dx, dy, dz = sph2xyz(*diff_sph)
#     return np.array([hx, hy, hz, dx, dy, dz])

# def rvectors_to_rsph(hx, hy, hz, dx, dy, dz):
#     half_sph = xyz2sph(hx, hy, hz)
#     diff_sph = xyz2sph(dx, dy, dz)
#     return half_sph, diff_sph
