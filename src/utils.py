import os
import numpy as np
import random
import torch

PI = np.pi  # the macro of the pi constant


def split_merl(merlpath, split=0.8):
    """
    split all MERL-format (.binary) BRDFs in a given path with a given ratio
    """
    brdf_names = os.listdir(merlpath)
    brdf_names = [name for name in brdf_names if ".binary" in name]
    # make sure the same order for reproducible experiments in any machine
    brdf_names.sort()

    return split_merl_subset(merlpath, brdf_names, split=split)


# diffuse subset in MERL
_diffuse_subset_merl = [
    "beige-fabric.binary",
    "black-fabric.binary",
    "blue-fabric.binary",
    "green-fabric.binary",
    "light-brown-fabric.binary",
    "pink-fabric.binary",
    "pink-fabric2.binary",
    "red-fabric.binary",
    "red-fabric2.binary",
    "white-fabric.binary",
    "white-fabric2.binary"
]
# specular subset in MERL
_specular_subset_merl = [
    "specular-black-phenolic.binary",
    "specular-blue-phenolic.binary",
    "specular-green-phenolic.binary",
    "specular-maroon-phenolic.binary",
    "specular-orange-phenolic.binary",
    "specular-red-phenolic.binary",
    "specular-violet-phenolic.binary",
    "specular-white-phenolic.binary",
    "specular-yellow-phenolic.binary",
    "yellow-phenolic.binary"
]


def split_merl_subset(merlpath, subset=_specular_subset_merl, split=0.8):
    """
    split the given subset with a given ratio
    """
    brdf_paths = []
    for name in subset:
        brdf_paths.append(os.path.join(merlpath, name))
    subset = np.asarray(subset)
    brdf_paths = np.asarray(brdf_paths)

    n_brdfs = len(subset)
    n_train_brdfs = int(n_brdfs * split)
    n_test_brdfs = n_brdfs - n_train_brdfs

    mask = np.zeros(n_brdfs, dtype=bool)
    mask[np.random.choice(n_brdfs, n_train_brdfs, replace=False)] = 1

    train_brdfs = brdf_paths[mask]
    test_brdfs = brdf_paths[~mask]

    return train_brdfs, test_brdfs


def seed_all(seed):
    """
    provide the seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(model, name, exp_path):
    """
    save the trained model for rendering
        model: the trained model
        name: the BRDF name
        exp_path: the path to save the model
    """
    if hasattr(model, "getBRDFTensor"):  # if the model is PCA, directly save to .binary
        save_binary(model.getBRDFTensor(), name, exp_path)
    else:
        save_npy(model, name, exp_path)


def save_binary(BRDFVals, name, exp_path):
    """
    save a BRDF tensor to a MERL-type .binary file
    """

    # Do MERL tonemapping if needed
    BRDFVals /= (1.00/1500, 1.15/1500, 1.66/1500)

    # Vectorize:
    vec = BRDFVals.T.flatten()

    filename = os.path.join(exp_path, f"{name}.binary")
    try:
        f = open(filename, "wb")
        np.array((90, 90, 180)).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", os.path.basename(filename))
        return


def save_npy(model, name, exp_path):
    """
    save the model's parameters
    """

    for el in model.named_parameters():
        param_name = el[0]   # either fc1.bias or fc1.weight
        weights = el[1]
        segs = param_name.split('.')
        if segs[-1] == 'weight':
            param_name = segs[0]
        else:
            param_name = segs[0].replace('fc', 'b')

        filename = f"{name}_{param_name}.npy"
        filepath = os.path.join(exp_path, filename)
        # transpose bc mitsuba code was developed for TF convention
        curr_weight = weights.detach().cpu().numpy().T
        np.save(filepath, curr_weight)


def freeze(model, freezed_layer_name=''):
    """
    do not compute gradients for some parameters 
    in order to simplify the computation graph
    """
    for name, param in model.named_parameters():
        if freezed_layer_name in name:
            param.requires_grad = False


def unfreeze(model, freezed_layer_name=''):
    """
    undo freeze
    """
    for name, param in model.named_parameters():
        if freezed_layer_name in name:
            param.requires_grad = True


def grid_sample_3d(image, optical):
    """
    this is an unofficial implementation of torch.nn.functional.grid_sample,
    BUT supports higher gradient computations.

    Modified from https://github.com/pytorch/pytorch/issues/34704, thanks for your awesome code :)
    """

    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():

        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw *
                           IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne *
                           IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw *
                           IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse *
                           IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw *
                           IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne *
                           IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw *
                           IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse *
                           IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val
