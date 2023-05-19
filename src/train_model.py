import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import argparse

import fastmerl_torch
import nbrdf
import utils
from sampler import sample_on_merl, uniform_sampler_preloaded
from ruamel.yaml import YAML


def main(config):
    # general setup
    # ----------

    batch_size = config.batch_size
    n_iter = config.n_iter
    learning_rate = config.lr
    mode = config.mode

    # set seed
    utils.seed_all(42)
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training
    # ----------

    # initialize model
    if config.model == "nbrdf":
        model = nbrdf.MLP().to(device)
    elif config.model == "phong":
        model = nbrdf.phong().to(device)
    elif config.model == "cooktorrance":
        model = nbrdf.cook_torrance().to(device)
    else:
        raise NotImplementedError(f"{config.model} has not been implemented!")

    loss_fn = nbrdf.mean_absolute_logarithmic_error

    # load samples depending on the mode
    if mode == "classic":
        splr = uniform_sampler_preloaded(device, n_loaded=batch_size, reject=True)
    elif mode == "overfit":
        splr = uniform_sampler_preloaded(device, reject=True)
    else:
        raise NotImplemented("mode should be either 'overfit' or 'classic'!")

    optim = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-15,  # eps=None raises error
        weight_decay=0.0,
        amsgrad=False,
    )

    # read merl brdf:
    merlpath = os.path.join(config.data_path, f"{config.brdf_name}.binary")
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
            logs["train_loss"] = f"{train_losses[-1]:.7f}"
            t.set_postfix(logs)
            t.update()

    # save trained results
    # ----------

    if config.save:
        # get workspace path
        _now = datetime.now()
        _format = "%Y_%m_%d_%H_%M_%S"
        workspace = _now.strftime(_format)
        ws_path = os.path.join(config.exp_path, workspace)

        # make the directory
        os.makedirs(ws_path, exist_ok=True)

        # save config
        yaml = YAML()
        with open(os.path.join(ws_path, "config.yaml"), "w") as f:
            yaml.dump(vars(config), f)

        # save train losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.savefig(os.path.join(ws_path, "loss.png"))

        torch.save(train_losses, os.path.join(ws_path, "train_losses.pth"))

        # save trained model
        utils.save_model(model, config.brdf_name, ws_path)


if __name__ == "__main__":
    # load command arguments
    # ----------

    parser = argparse.ArgumentParser(
        description="fit model to BRDF with specified configurations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="the training batch size"
    )
    parser.add_argument(
        "--n_iter", type=int, default=10000, help="the number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="the learning rate")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="the path of data"
    )
    parser.add_argument(
        "--brdf_name", type=str, default="alum-bronze", help="the brdf to be trained on"
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./outputs",
        help="the path of saved results of this experiment",
    )
    parser.add_argument(
        "--model", type=str, default="nbrdf", help="the model being fitted"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classic",
        help="classic->few steps + few samples, overfit->unlimited resources",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="if True, save the results into the workspace in the specified folder",
    )
    args = parser.parse_args()

    main(args)
