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

    invalid_samples = list()

    for step in range(k):
        rangles_adapt, mlp_input_adapt, groundTruth_adapt = sampler.sample_on_merl(
            task_train, splr, shots
        )
        valid_idx = torch.any(groundTruth_adapt != 0.0, dim=1)
        n_valid = valid_idx.sum()  # the number of valid samples
        if n_valid != shots:
            invalid_samples.append(rangles_adapt[~valid_idx, :])
            # skip this step if there are not valid samples
            if n_valid == 0:
                continue
            rangles_adapt, mlp_input_adapt, groundTruth_adapt = (
                rangles_adapt[valid_idx, :],
                mlp_input_adapt[valid_idx, :],
                groundTruth_adapt[valid_idx, :],
            )
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

    # compute rejection loss for invalid samples
    if len(invalid_samples) != 0:
        invalid_samples = torch.vstack(invalid_samples)
        loss_w = 1e-2  # to balance 2 loss values
        rejection_loss = (
            loss_w
            * 0.5
            * (invalid_samples[:, 0] ** 2 + invalid_samples[:, 1] ** 2).sum()
        )
    else:
        rejection_loss = 0.0

    return eval_loss, rejection_loss


def evaluate(loader, model_GBML, splr, shots, k, loss_fn):
    freeze(splr)
    freeze(model_GBML, "lrs")

    meta_val_loss = 0.0
    meta_val_rej_loss = 0.0
    for tasks in loader:
        for _, task in enumerate(zip(*tasks)):
            learner = model_GBML.clone()

            eval_loss, rejection_loss = fast_adapt(
                learner, task, splr, shots, k, loss_fn
            )
            meta_val_loss += eval_loss.item()
            meta_val_rej_loss += (
                rejection_loss
                if isinstance(rejection_loss, float)
                else rejection_loss.item()
            )
    meta_val_loss /= len(loader.dataset)
    meta_val_rej_loss /= len(loader.dataset)

    unfreeze(splr)
    unfreeze(model_GBML, "lrs")

    return meta_val_loss, meta_val_rej_loss


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
    n_det = config.n_det
    if n_det == -1:
        n_det = k * shots
    sampler_lr = config.sampler_lr
    fast_lr = config.fast_lr
    meta_bs = config.meta_bs
    n_epochs = config.n_epochs
    n_display_ep = config.n_disp_ep

    # config path
    exp_path = config.exp_path
    data_path = config.data_path
    model_path = config.model_path
    sampler_path = config.sampler_path

    # prepare datasets
    train_brdfs, test_brdfs = split_merl(data_path, split=0.8)
    # print(f"datasets: {len(train_brdfs)} for training and {len(test_brdfs)} for testing")

    taskset_train = datasets.MerlTaskset(train_brdfs, n_test_samples=25000)
    taskset_test = datasets.MerlTaskset(test_brdfs, n_test_samples=25000)

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

    # load the pretrained meta model
    pretrained_model = torch.load(
        os.path.join(model_path, f"pretrained_{config.model}_20x512_10000ep.pth")
    )
    model_GBML.load_state_dict(pretrained_model)

    # prepare sampler
    splr = sampler.trainable_sampler_det(n_det, quasi_init=True)
    if n_det == 1:
        # 50 attempts to select the best initial positions
        best_attempt_loss = float("inf")
        for _ in range(50):
            tmp_splr = sampler.trainable_sampler_det(n_det)
            tmp_splr.to(device)
            attempt_loss, _ = evaluate(
                taskloader_train, model_GBML, tmp_splr, shots, k, loss_fn
            )
            if attempt_loss < best_attempt_loss:
                best_attempt_loss = attempt_loss
                splr = tmp_splr
    else:
        trained_sampler_path = os.path.join(
            sampler_path, f"meta_sampler_{config.model}_{n_det//2}.pth"
        )
        if os.path.exists(trained_sampler_path):
            splr.load_samples(torch.load(trained_sampler_path))

    splr.to(device)

    sampler_optimizer = optim.Adam(splr.parameters(), sampler_lr)
    sampler_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sampler_optimizer, T_max=500, eta_min=sampler_lr / 5
    )

    # misc variables
    val_loss = "N/A"  # for logging

    losses = list()
    rej_losses = list()

    val_losses = list()
    val_rej_losses = list()

    # record the reference loss and the initial states
    meta_val_loss, meta_val_rej_loss = evaluate(
        taskloader_test, model_GBML, splr, shots, k, loss_fn
    )
    val_loss = f"{meta_val_loss:.5f}"
    val_losses.append(meta_val_loss)
    val_rej_losses.append(meta_val_rej_loss)

    # save in the beginning
    if config.save:
        _now = datetime.now()
        _format = "%Y_%m_%d_%H_%M_%S"
        workspace = _now.strftime(_format)
        ws_path = os.path.join(exp_path, workspace)
        os.makedirs(ws_path, exist_ok=True)

        yaml = YAML()
        with open(os.path.join(ws_path, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    def make_checkpoint(counter):
        ckpt = dict()
        ckpt["sampler"] = copy.deepcopy(splr.state_dict())
        ckpt["sampler_optimizer"] = copy.deepcopy(sampler_optimizer.state_dict())
        ckpt["sampler_scheduler"] = copy.deepcopy(sampler_scheduler.state_dict())
        torch.save(ckpt, os.path.join(ws_path, f"ckpt_{counter:04d}.pth"))

    ckpt_counter = 0
    if config.save:
        make_checkpoint(ckpt_counter)
        ckpt_counter += 1

    # for recording the best
    best_loss = float("inf")

    # main loop
    # ----------
    with tqdm(total=n_epochs) as t:
        for ep in range(n_epochs):
            # logging info
            logs = {}

            meta_train_loss = 0.0
            meta_train_rej_loss = 0.0
            for tasks in taskloader_train:
                sampler_optimizer.zero_grad()
                total_loss = 0.0
                for _, task in enumerate(zip(*tasks)):
                    learner = model_GBML.clone()

                    eval_loss, rejection_loss = fast_adapt(
                        learner, task, splr, shots, k, loss_fn
                    )
                    total_loss += eval_loss + rejection_loss
                    meta_train_loss += eval_loss.item()
                    meta_train_rej_loss += (
                        rejection_loss
                        if isinstance(rejection_loss, float)
                        else rejection_loss.item()
                    )

                total_loss = total_loss / taskloader_train.batch_size
                total_loss.backward()

                sampler_optimizer.step()

            sampler_scheduler.step()

            # logging
            meta_train_loss = meta_train_loss / len(taskloader_train.dataset)
            meta_train_rej_loss = meta_train_rej_loss / len(taskloader_train.dataset)
            losses.append(meta_train_loss)
            rej_losses.append(meta_train_rej_loss)

            # record the best
            if meta_train_loss < best_loss:
                best_loss = meta_train_loss
                # save the best splr over training
                if config.save:
                    torch.save(
                        copy.deepcopy(splr.state_dict()),
                        os.path.join(ws_path, f"meta_sampler_{config.model}_{n_det}.pth"),
                    )
                    torch.save(
                        copy.deepcopy(splr.state_dict()),
                        os.path.join(sampler_path, f"meta_sampler_{config.model}_{n_det}.pth"),
                    )

            # validate
            if (ep + 1) % n_display_ep == 0:
                meta_val_loss, meta_val_rej_loss = evaluate(
                    taskloader_test, model_GBML, splr, shots, k, loss_fn
                )

                # logging
                val_loss = f"{meta_val_loss:.5f}"
                val_losses.append(meta_val_loss)
                val_rej_losses.append(meta_val_rej_loss)

                # record intermediate states
                if config.save:
                    make_checkpoint(ckpt_counter)
                    ckpt_counter += 1

            logs["val_loss"] = val_loss
            logs["train_loss"] = f"{meta_train_loss:.5f}"
            logs["best_loss"] = f"{best_loss:.5f}"
            t.set_postfix(logs)
            t.update()

    if config.save:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.savefig(os.path.join(ws_path, "train_losses.pdf"), bbox_inches="tight")
        torch.save(losses, os.path.join(ws_path, "train_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(rej_losses)
        plt.savefig(os.path.join(ws_path, "train_rej_losses.pdf"), bbox_inches="tight")
        torch.save(rej_losses, os.path.join(ws_path, "train_rej_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(val_losses)
        plt.savefig(os.path.join(ws_path, "validate_losses.pdf"), bbox_inches="tight")
        torch.save(val_losses, os.path.join(ws_path, "validate_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(val_rej_losses)
        plt.savefig(
            os.path.join(ws_path, "validate_rej_losses.pdf"), bbox_inches="tight"
        )
        torch.save(val_rej_losses, os.path.join(ws_path, "validate_rej_losses.pth"))


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
        "--sampler_path",
        type=str,
        default="/content/data/meta-samplers/",
        help="the path containing those trained meta samplers",
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
        "--n_det",
        type=int,
        default=-1,
        help="the number of trainable deterministic directions, deafulting to -1, which indicates k*shots",
    )
    parser.add_argument(
        "--meta_bs", type=int, default=1, help="the batch size of outer loop"
    )
    parser.add_argument(
        "--fast_lr", type=float, default=1e-3, help="the learning rate of inner loop"
    )
    parser.add_argument(
        "--sampler_lr", type=float, default=1e-4, help="the learning rate of sampler"
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
