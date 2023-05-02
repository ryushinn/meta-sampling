from datetime import datetime
import os
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader
import learn2learn as l2l
from ruamel.yaml import YAML

import datasets, sampler, nbrdf, utils
from utils import freeze, unfreeze, split_merl

import argparse


def fast_adapt(learner, task, splr, shots, k, loss_fn):
    task_train, task_test = task

    for step in range(k):
        (
            rangles_adapt,
            mlp_input_adapt,
            groundTruth_adapt,
        ) = sampler.sample_on_merl_with_rejection(task_train, splr, shots)
        output = learner(mlp_input_adapt)
        rgb_pred = nbrdf.brdf_to_rgb(rangles_adapt, output)
        rgb_gt = nbrdf.brdf_to_rgb(rangles_adapt, groundTruth_adapt)
        train_loss = loss_fn(y_true=rgb_gt, y_pred=rgb_pred)
        learner.adapt(train_loss)

    # compute eval_loss for valid samples
    rangles_eval, mlp_input_eval, groundTruth_eval = task_test.next()
    output = learner(mlp_input_eval)
    rgb_pred = nbrdf.brdf_to_rgb(rangles_eval, output)
    rgb_gt = nbrdf.brdf_to_rgb(rangles_eval, groundTruth_eval)
    eval_loss = loss_fn(y_true=rgb_gt, y_pred=rgb_pred)

    return eval_loss


def evaluate(loader, model_GBML, splr, shots, k, loss_fn):
    freeze(model_GBML, "lrs")

    meta_val_loss = 0.0
    for tasks in loader:
        for _, task in enumerate(zip(*tasks)):
            learner = model_GBML.clone()

            eval_loss = fast_adapt(learner, task, splr, shots, k, loss_fn=loss_fn)
            meta_val_loss += eval_loss.item()

    meta_val_loss /= len(loader.dataset)

    unfreeze(model_GBML, "lrs")

    return meta_val_loss


def main(config):
    # general setup
    # ----------

    # FOR DEBUG
    # torch.autograd.set_detect_anomaly(True)

    # torch config & set random seed
    utils.seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)

    # hyperparameters
    shots = config.shots
    k = config.k
    meta_lr = config.meta_lr
    fast_lr = config.fast_lr
    meta_bs = config.meta_bs
    n_epochs = config.n_epochs
    n_display_ep = config.n_disp_ep

    # config path
    exp_path = config.exp_path
    data_path = config.data_path
    model_path = config.model_path

    # prepare datasets
    train_brdfs, test_brdfs = split_merl(data_path, split=0.8)
    # print(f"datasets: {len(train_brdfs)} for training and {len(test_brdfs)} for testing")

    taskset_train = datasets.MerlTaskset(train_brdfs, n_test_samples=512)
    taskset_test = datasets.MerlTaskset(test_brdfs, n_test_samples=512)

    taskloader_train = DataLoader(
        taskset_train, meta_bs, shuffle=True, collate_fn=datasets.custom_collate
    )

    taskloader_test = DataLoader(
        taskset_test, len(test_brdfs), collate_fn=datasets.custom_collate
    )

    # training setting
    # ----------
    if config.model == "nbrdf":
        model = nbrdf.MLP
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    elif config.model == "phong":
        model = nbrdf.phong
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    elif config.model == "cooktorrance":
        model = nbrdf.cook_torrance
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    else:
        raise NotImplementedError(f"{config.model} have not been implemented!")

    model_GBML = l2l.algorithms.MetaSGD(model=model(), lr=fast_lr).to(device)

    # 1e-6 weight decaying comes from
    # [Michael, 2022, Metappearance: Meta-Learning for Visual Appearance Reproduction.]
    model_optimizer = optim.Adam(model_GBML.parameters(), lr=meta_lr, weight_decay=1e-6)

    splr = sampler.uniform_sampler_preloaded(reject=True)
    splr.to(device)

    # misc variables
    val_loss = "N/A"  # for logging

    losses = list()
    val_losses = list()

    # record the reference loss and the initial states
    meta_val_loss = evaluate(taskloader_test, model_GBML, splr, shots, k, loss_fn)
    val_loss = f"{meta_val_loss:.5f}"
    val_losses.append(meta_val_loss)

    # save in the beginning
    if config.save:
        _now = datetime.now()
        _format = "%Y_%m_%d_%H_%M_%S"
        workspace = _now.strftime(_format)
        ws_path = os.path.join(exp_path, workspace)
        os.makedirs(ws_path, exist_ok=True)

        yaml = YAML()
        with open(os.path.join(ws_path, "config.yaml"), "w") as f:
            yaml.dump(vars(config), f)

    def make_checkpoint(counter):
        ckpt = dict()
        ckpt["model"] = copy.deepcopy(model_GBML.state_dict())
        ckpt["model_optimizer"] = copy.deepcopy(model_optimizer.state_dict())
        torch.save(ckpt, os.path.join(ws_path, f"ckpt_{counter:04d}.pth"))

        torch.save(
            copy.deepcopy(model_GBML.state_dict()),
            os.path.join(model_path, f"pretrained_{config.model}_20x512_10000ep.pth")
        )

    ckpt_counter = 0
    if config.save:
        make_checkpoint(ckpt_counter)
        ckpt_counter += 1

    # main loop
    # ----------
    with tqdm(total=n_epochs) as t:
        for ep in range(n_epochs):
            # logging info
            logs = {}

            meta_train_loss = 0.0
            meta_train_rej_loss = 0.0
            for tasks in taskloader_train:
                model_optimizer.zero_grad()
                total_loss = 0.0
                for _, task in enumerate(zip(*tasks)):
                    learner = model_GBML.clone()

                    eval_loss = fast_adapt(learner, task, splr, shots, k, loss_fn)
                    total_loss += eval_loss
                    meta_train_loss += eval_loss.item()

                total_loss = total_loss / taskloader_train.batch_size
                total_loss.backward()
                model_optimizer.step()

            # logging
            meta_train_loss = meta_train_loss / len(taskloader_train.dataset)
            losses.append(meta_train_loss)

            # validate
            if (ep + 1) % n_display_ep == 0:
                meta_val_loss = evaluate(
                    taskloader_test, model_GBML, splr, shots, k, loss_fn
                )

                # logging
                val_loss = f"{meta_val_loss:.5f}"
                val_losses.append(meta_val_loss)

                # record intermediate states
                if config.save:
                    make_checkpoint(ckpt_counter)
                    ckpt_counter += 1

            logs["val_loss"] = val_loss
            logs["train_loss"] = f"{meta_train_loss:.5f}"
            t.set_postfix(logs)
            t.update()

    if config.save:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.savefig(os.path.join(ws_path, "train_losses.pdf"), bbox_inches="tight")
        torch.save(losses, os.path.join(ws_path, "train_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(val_losses)
        plt.savefig(os.path.join(ws_path, "validate_losses.pdf"), bbox_inches="tight")
        torch.save(val_losses, os.path.join(ws_path, "validate_losses.pth"))


if __name__ == "__main__":
    # load command arguments
    # ----------

    parser = argparse.ArgumentParser(
        description="run meta-sampler experiment with specified configurations"
    )
    parser.add_argument(
        "--model", type=str, default="nbrdf", help="the name of model to be trained"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/data/brdfs/",
        help="the path containing brdf binaries",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/content/data/meta-models/",
        help="the path containing those pretrained meta models",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1,
        help="the number of samples per step in the inner loop",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="the number of steps in the inner loop"
    )
    parser.add_argument(
        "--meta_bs", type=int, default=1, help="the batch size of outer loop"
    )
    parser.add_argument(
        "--meta_lr", type=float, default=1e-4, help="the meta learning rate"
    )
    parser.add_argument(
        "--fast_lr", type=float, default=1e-3, help="the learning rate of inner loop"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="the number of epochs"
    )
    parser.add_argument(
        "--n_disp_ep",
        type=int,
        default=10,
        help="the number of epochs to validate the model",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="if True, save the results into the workspace in the specified folder",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="/content/drive/MyDrive/experiments/nbrdf-meta_sampler/",
        help="the experiment folder",
    )

    args = parser.parse_args()

    main(args)
