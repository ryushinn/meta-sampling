import os
from random import uniform
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import argparse

import fastmerl_torch
import nbrdf
import utils
import shutil
from sampler import sample_on_merl_with_rejection, sample_on_merl, uniform_sampler_preloaded, trainable_sampler_det
from utils import freeze
import fastmerl_torch

if __name__ == '__main__':

    """
    ----------------------------------------
    load command arguments
    ----------------------------------------
    """

    parser = argparse.ArgumentParser(
        description='fit model to BRDF with specified configurations')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='the training batch size')
    parser.add_argument('--n_iter', type=int, default=10000,
                        help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='the learning rate')
    parser.add_argument('--data_path', type=str,
                        default='./data', help='the path of data')
    parser.add_argument('--brdf_name', type=str,
                        default='alum-bronze', help='the brdf to be trained on')
    parser.add_argument('--save_path', type=str,
                        default='./outputs', help='the path of saved results')
    parser.add_argument('--model', type=str, default='nbrdf',
                        help='the model being fitted')
    parser.add_argument('--mode', type=str, default='classic',
                        help='classic->few steps + few samples, overfit->unlimited resources')
    parser.add_argument('--save', action='store_true',
                        help='if True, save the results into the workspace in the specified folder')
    args = parser.parse_args()

    batch_size = args.batch_size
    n_iter = args.n_iter
    learning_rate = args.lr
    mode = args.mode

    """
    ----------------------------------------
    general configurations
    ----------------------------------------
    """

    # set seed
    utils.seed_all(42)
    torch.set_default_dtype(torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    ----------------------------------------
    training
    ----------------------------------------
    """

    # initialize model
    if args.model == 'nbrdf':
        model = nbrdf.MLP().to(device)
    elif args.model == 'phong':
        model = nbrdf.phong().to(device)
    elif args.model == 'cooktorrance':
        model = nbrdf.cook_torrance().to(device)
    else:
        raise NotImplementedError(f"{args.model} has not been implemented!")

    loss_fn = nbrdf.mean_absolute_logarithmic_error

    # load samples depending on the mode
    if mode == 'classic':
        splr = uniform_sampler_preloaded(
            device, n_loaded=batch_size, reject=True)
    elif mode == 'overfit':
        splr = uniform_sampler_preloaded(device, reject=True)
    else:
        raise NotImplemented("mode should be either 'overfit' or 'classic'!")

    optim = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             betas=(0.9, 0.999),
                             eps=1e-15,  # eps=None raises error
                             weight_decay=0.0,
                             amsgrad=False)

    # read merl brdf:
    merlpath = os.path.join(args.data_path, f"{args.brdf_name}.binary")
    merl = fastmerl_torch.Merl(merlpath, device)

    train_losses = []

    with tqdm(total=n_iter, desc="iter") as t:
        for it in range(n_iter):
            logs = {}

            # get batch from MERL data
            optim.zero_grad()
            rangles, mlp_input, gt = sample_on_merl(merl, splr, batch_size)

            # feed into model to get prediction
            output = model(mlp_input)

            # convert to RGB data
            rgb_pred = nbrdf.brdf_to_rgb(rangles, output)
            rgb_true = nbrdf.brdf_to_rgb(rangles, gt)

            loss = loss_fn(y_true=rgb_true, y_pred=rgb_pred)
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            logs['train_loss'] = f"{train_losses[-1]:.7f}"
            t.set_postfix(logs)
            t.update()

    """
    ----------------------------------------
    save trained results
    ----------------------------------------
    """

    if args.save:

        # get workspace path
        _now = datetime.now()
        _format = "%Y_%m_%d_%H_%M_%S"
        workspace = _now.strftime(_format)
        ws_path = os.path.join(args.save_path, workspace)

        # get model path
        method_path = f"classic_{args.model}"
        if mode == 'classic':
            model_path = os.path.join(method_path, f"{batch_size}")
        elif mode == 'overfit':
            model_path = os.path.join(method_path, "overfit")

        # make the directory
        os.makedirs(ws_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, model_path), exist_ok=True)

        # save train losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.savefig(os.path.join(ws_path, "loss.png"))

        torch.save(train_losses, os.path.join(ws_path, "train_losses.pth"))

        # save trained model
        utils.save_model(model, args.brdf_name,
                         os.path.join(args.save_path, model_path))
