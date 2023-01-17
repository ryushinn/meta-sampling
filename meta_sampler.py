from functools import partial
from tabnanny import verbose
import time
from datetime import datetime
import os
import shutil
import numpy as np
import copy
import random


import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader, dataloader
import learn2learn as l2l
from ruamel.yaml import YAML

import fastmerl_torch, datasets, sampler, nbrdf, utils
from utils import freeze, unfreeze, split_merl, split_merl_subset

import argparse

def fast_adapt(learner, task, splr, trainable, shots, adaptation_steps, loss_fn, model):
    task_train, task_test = task

    if trainable: invalid_samples = list()
    
    for step in range(adaptation_steps):
        if trainable:
            rangles_adapt, mlp_input_adapt, groundTruth_adapt = sampler.sample_on_merl(task_train, splr, shots)
            valid_idx = torch.any(groundTruth_adapt != 0., dim=1)
            n_valid = valid_idx.sum() # the number of valid samples
            if n_valid != shots:
                invalid_samples.append(rangles_adapt[~valid_idx, :])
                # skip this step if there are not valid samples
                if n_valid == 0: continue
                rangles_adapt, mlp_input_adapt, groundTruth_adapt = (
                    rangles_adapt[valid_idx, :], 
                    mlp_input_adapt[valid_idx, :],
                    groundTruth_adapt[valid_idx, :]
                )
        else:
            rangles_adapt, mlp_input_adapt, groundTruth_adapt = sampler.sample_on_merl_with_rejection(task_train, splr, shots)
        if model == 'PCA':
            output = learner(rangles_adapt)
        else:
            output = learner(mlp_input_adapt)
        rgb_pred = nbrdf.brdf_to_rgb(rangles_adapt, output)
        rgb_gt = nbrdf.brdf_to_rgb(rangles_adapt, groundTruth_adapt)
        train_loss = loss_fn(y_true=rgb_gt, y_pred=rgb_pred)
        learner.adapt(train_loss)

    # compute eval_loss for valid samples
    rangles_eval, mlp_input_eval, groundTruth_eval = task_test.next()
    if model == 'PCA':
        output = learner(rangles_eval)
    else:
        output = learner(mlp_input_eval)
    rgb_pred = nbrdf.brdf_to_rgb(rangles_eval, output)
    rgb_gt = nbrdf.brdf_to_rgb(rangles_eval, groundTruth_eval)
    eval_loss = loss_fn(y_true=rgb_gt, y_pred=rgb_pred)

    # compute rejection loss for invalid samples
    if trainable and len(invalid_samples) != 0:
        invalid_samples = torch.vstack(invalid_samples)
        loss_w = 1e-2 # to balance 2 loss values
        rejection_loss = loss_w * 0.5 *  (invalid_samples[:, 0] ** 2 + invalid_samples[:, 1] ** 2).sum()
    else:
        rejection_loss = 0.0
    
    return eval_loss, rejection_loss

def evaluate(loader, model_GBML, splr, trainable, shots, k, loss_fn, model):
    if trainable: freeze(splr)
    freeze(model_GBML, 'lrs')
    meta_val_loss = 0.0
    meta_val_rej_loss = 0.0
    for tasks in loader:
        for _, task in enumerate(zip(*tasks)):
            learner = model_GBML.clone()

            eval_loss, rejection_loss = fast_adapt(
                learner, task, 
                splr,
                trainable, 
                shots, k,
                loss_fn=loss_fn,
                model=model
            )
            meta_val_loss += eval_loss.item()
            meta_val_rej_loss += rejection_loss \
                if isinstance(rejection_loss, float) else rejection_loss.item()
    meta_val_loss /= len(loader.dataset)
    meta_val_rej_loss /= len(loader.dataset)

    if trainable: unfreeze(splr)
    unfreeze(model_GBML, 'lrs')

    return meta_val_loss, meta_val_rej_loss

def main(config):
    # torch.autograd.set_detect_anomaly(True)

    # torch config & set random seed
    RD_SEED = config['RD_SEED']

    utils.seed_all(RD_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float32)
    
    # hyperparameters
    shots = config['shots']
    k = config['k']
    n_det = config['n_det']
    if n_det == -1:
        n_det = k * shots
    meta_lr = config['meta_lr']
    sampler_lr = config['sampler_lr']
    fast_lr = config['fast_lr']
    meta_bs = config['meta_bs']
    n_epochs = config['n_epochs']
    n_display_ep = config['n_disp_ep']

    # config path
    exp_path = config['exp_path']
    brdf_path = config['brdf_path']

    # prepare datasets
    n_test_samples = config['n_test_samples']
    train_brdfs, test_brdfs = split_merl(brdf_path, split=0.8)
    # print(f"datasets: {len(train_brdfs)} for training and {len(test_brdfs)} for testing")

    taskset_train = datasets.MerlTaskset(train_brdfs, n_test_samples)
    taskset_test = datasets.MerlTaskset(test_brdfs, n_test_samples)

    taskloader_train = DataLoader(
        taskset_train, meta_bs, 
        shuffle=True, collate_fn=datasets.custom_collate
    )

    taskloader_test = DataLoader(
        taskset_test, len(test_brdfs), 
        collate_fn=datasets.custom_collate
    )

    train_model = config['train_model']

    # training setting
    if config['model'] == 'nbrdf':
        model = nbrdf.MLP
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    elif config['model'] == 'phong':
        model = nbrdf.phong
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    elif config['model'] == 'cooktorrance':
        model = nbrdf.cook_torrance
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    elif config['model'] == 'PCA':
        model = partial(nbrdf.PCA, precomputed_path=exp_path, basis_num=config['n_basis'])
        loss_fn = nbrdf.mean_absolute_logarithmic_error
    else:
        raise NotImplementedError(f'{config["model"]} have not been implemented!')

    # transform = l2l.optim.transforms.MetaCurvatureTransform
    # transform = nbrdf.ModuleTransform(partial(nbrdf.ScaleLayer, init_scaling=fast_lr))
    # transform = partial(nbrdf.MultiStepSGDTransform, init_scaling=fast_lr, k=k, verbose=0)
    # transform = partial(nbrdf.AdamTransform, alpha=fast_lr, beta1=0.9, beta2=0.999, k=k, verbose=0)
    # use lr=1.0 to make sure gradients w.r.t. transform intact
    # model_GBML = l2l.algorithms.GBML(module=model(), transform=transform, lr=1.0, first_order=True).to(device)
    model_GBML = l2l.algorithms.MetaSGD(model=model(), lr=fast_lr).to(device)


    if not train_model:
        # load the pretrained meta model
        if config['model'] == "PCA":
            pretrained_model_name = f"pretrained_{config['model']}{config['n_basis']}_20x512_1000ep.pth"
        else:
            pretrained_model_name = f"pretrained_{config['model']}_20x512_10000ep.pth"
        pretrained_model = torch.load(os.path.join(exp_path, pretrained_model_name))
        model_GBML.load_state_dict(pretrained_model)

    if train_model:
        # 1e-6 weight decaying comes from [Michael, 2022, Metappearance: Meta-Learning for Visual Appearance Reproduction.]
        model_optimizer = optim.Adam(model_GBML.parameters(), lr=meta_lr, weight_decay=1e-6)

    # prepare sampler
    if config['sampler'] == 'uniform':
        trainable = False
        splr = sampler.uniform_sampler()
    elif config['sampler'] == 'uniform_preloaded':
        trainable = False
        splr = sampler.uniform_sampler_preloaded(reject=True)
    elif config['sampler'] == 'trainable_det':
        trainable = True
        splr = sampler.trainable_sampler_det(quasi_init=True)
        if n_det == 1:
            # N attempts to select the best initial positions
            n_attempts = 50
            best_attempt_loss = 1e5
            for _ in range(n_attempts):
                tmp_splr = sampler.trainable_sampler_det(n_det)
                tmp_splr.to(device)
                attempt_loss, _ = evaluate(taskloader_train, model_GBML, tmp_splr, trainable, shots, k, loss_fn, config['model'])
                if attempt_loss < best_attempt_loss:
                    best_attempt_loss = attempt_loss
                    print(best_attempt_loss)
                    splr = tmp_splr
        else:
            pretrained = os.path.join(exp_path, f"ckpt_best_{n_det//2}.pth")
            if os.path.exists(pretrained):
                pretrained = torch.load(pretrained)
                pretrained_splr = sampler.trainable_sampler_det(n_det // 2)
                pretrained_splr.load_state_dict(pretrained)
                splr.load_sampler(pretrained_splr)
        
    elif config['sampler'] == 'trainable':   
        trainable = True
        N = config['N']
        Km = config['Km']
        Ks = config['Ks']
        splr = sampler.trainable_sampler(N, Km, Ks)
    elif config['sampler'] == 'trainable_MLP':
        trainable = True
        splr = sampler.trainable_sampler_MLP()
        identity_weights = torch.load(os.path.join(exp_path, "identity_weights.pth"))
        splr.load_state_dict(identity_weights)
    else:
        raise NotImplementedError(f'{config["sampler"]} have not been implemented!')
    splr.to(device)

    if trainable:
        sampler_optimizer = optim.Adam(splr.parameters(), sampler_lr)
        # sampler_optimizer = optim.Adam(
        #     [
        #         {'params': splr.parameters()},
        #         {'params': model_metasgd.lrs.parameters(), 'lr':meta_lr}
        #     ], 
        #     sampler_lr
        # )
        sampler_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            sampler_optimizer, T_max=500, eta_min=sampler_lr / 5
        )

    # debug: warning
    if train_model and trainable:
        print("WARNING: the controlling flow is not expected, double check your setting!")
    
    # misc variables
    val_loss='N/A' # for logging

    losses = list()
    rej_losses = list()

    val_losses = list()
    val_rej_losses = list()

    # record the reference loss and the initial states
    meta_val_loss, meta_val_rej_loss = evaluate(taskloader_test, model_GBML, splr, trainable, shots, k, loss_fn, config['model'])
    val_loss = f'{meta_val_loss:.5f}'
    val_losses.append(meta_val_loss)
    val_rej_losses.append(meta_val_rej_loss)

    # save in the begining
    if config['save']:
        _now = datetime.now()
        _format = "%Y_%m_%d_%H_%M_%S"
        workspace = _now.strftime(_format)
        ws_path = os.path.join(exp_path, workspace)
        os.makedirs(ws_path, exist_ok=True)

        with open(os.path.join(ws_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

        with open(os.path.join(ws_path, 'readme.txt'), 'w') as f:
            f.write(config['readme'])

    ckpt_counter = 0
    def make_checkpoint(counter):
        ckpt = dict()
        ckpt['model'] = copy.deepcopy(model_GBML.state_dict())
        if trainable:
            ckpt['sampler'] = copy.deepcopy(splr.state_dict())
            ckpt['sampler_optimizer'] = copy.deepcopy(sampler_optimizer.state_dict())
            ckpt['sampler_scheduler'] = copy.deepcopy(sampler_scheduler.state_dict())
        if train_model:
            ckpt['model_optimizer'] = copy.deepcopy(model_optimizer.state_dict())
        torch.save(ckpt, os.path.join(ws_path, f"ckpt_{counter:04d}.pth"))

    if config['save']: 
        make_checkpoint(ckpt_counter)
        ckpt_counter += 1
    
    if trainable:
        best_loss = 100

    # main loop
    with tqdm(total=n_epochs) as t:
        for ep in range(n_epochs):
            
            # logging info
            logs={}
            
            meta_train_loss = 0.0
            meta_train_rej_loss = 0.0
            for tasks in taskloader_train:
                
                if trainable: sampler_optimizer.zero_grad()
                if train_model: model_optimizer.zero_grad()
                total_loss = 0.0
                for _, task in enumerate(zip(*tasks)):
                    learner = model_GBML.clone()
                    
                    eval_loss, rejection_loss = fast_adapt(
                        learner, task, 
                        splr,
                        trainable, 
                        shots, k,
                        loss_fn,
                        config['model']
                    )
                    total_loss += eval_loss + rejection_loss
                    meta_train_loss += eval_loss.item()
                    meta_train_rej_loss += rejection_loss \
                        if isinstance(rejection_loss, float) else rejection_loss.item()

                total_loss = total_loss / taskloader_train.batch_size
                total_loss.backward()
                if trainable: sampler_optimizer.step()
                if train_model: model_optimizer.step()

            if trainable: sampler_scheduler.step()

            # logging
            meta_train_loss = meta_train_loss / len(taskloader_train.dataset)
            meta_train_rej_loss = meta_train_rej_loss / len(taskloader_train.dataset)
            losses.append(meta_train_loss)
            rej_losses.append(meta_train_rej_loss)

            # record the best
            if trainable:
                if meta_train_loss < best_loss:
                    best_loss = meta_train_loss
                    # save the best splr over training
                    if config['save']: 
                        torch.save(
                            copy.deepcopy(splr.state_dict()), 
                            os.path.join(ws_path, f"ckpt_best.pth")
                        )
                        torch.save(
                            copy.deepcopy(splr.state_dict()), 
                            os.path.join(exp_path, f"ckpt_best_{n_det}.pth")
                        )

            # validate
            if (ep + 1) % n_display_ep == 0:
                meta_val_loss, meta_val_rej_loss = evaluate(
                    taskloader_test, model_GBML, splr, 
                    trainable, shots, k,
                    loss_fn,
                    config['model']
                )

                # logging
                val_loss = f'{meta_val_loss:.5f}'
                val_losses.append(meta_val_loss)
                val_rej_losses.append(meta_val_rej_loss)

                # record intermediate states
                if config['save']: 
                    make_checkpoint(ckpt_counter)
                    ckpt_counter += 1

            logs['val_loss'] = val_loss
            logs['train_loss'] = f"{meta_train_loss:.5f}"
            if trainable: logs['best_loss'] = f"{best_loss:.5f}"
            t.set_postfix(logs)
            t.update()

    if config['save']:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.savefig(os.path.join(ws_path, "train_losses.pdf"), bbox_inches='tight')
        torch.save(losses, os.path.join(ws_path, "train_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(rej_losses)
        plt.savefig(os.path.join(ws_path, "train_rej_losses.pdf"), bbox_inches='tight')
        torch.save(rej_losses, os.path.join(ws_path, "train_rej_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(val_losses)
        plt.savefig(os.path.join(ws_path, "validate_losses.pdf"), bbox_inches='tight')
        torch.save(val_losses, os.path.join(ws_path, "validate_losses.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(val_rej_losses)
        plt.savefig(os.path.join(ws_path, "validate_rej_losses.pdf"), bbox_inches='tight')
        torch.save(val_rej_losses, os.path.join(ws_path, "validate_rej_losses.pth"))


parser = argparse.ArgumentParser(description='run meta-sampler experiment with the specified configurations')
parser.add_argument('--config_path', type=str, help='the path of configuration file. If specified, overwrite all other arguments')
parser.add_argument('--sampler', type=str, default='uniform_hd', help='the name of sampler to be trained')
parser.add_argument('--model', type=str, default='nbrdf', help='the name of model to be trained')
parser.add_argument('--n_basis', type=int, default=240, help='the number of basis used in PCA model')
parser.add_argument('--brdf_path', type=str, default='/content/data/brdfs/', help='the path containing brdf binaries')
parser.add_argument('--shots', type=int, default=1, help='the number of samples per step in the inner loop')
parser.add_argument('--k', type=int, default=1, help='the number of steps in the inner loop')
parser.add_argument('--n_det', type=int, default=-1, help='the number of trainable deterministic directions, deafulting to -1, which indicates k*shots')
parser.add_argument('--N', type=int, default=1, help='the number of stacked flows for trainable distributions by NF')
parser.add_argument('--Km', type=int, default=6, help='the number of centers of mobius transforms')
parser.add_argument('--Ks', type=int, default=12, help='the number of segments of spline transforms')
parser.add_argument('--meta_bs', type=int, default=1, help='the batch size of outer loop')
parser.add_argument('--meta_lr', type=float, default=1e-4, help='the meta learning rate')
parser.add_argument('--fast_lr', type=float, default=1e-3, help='the learning rate of inner loop')
parser.add_argument('--sampler_lr', type=float, default=1e-4, help='the learning rate of sampler')
parser.add_argument('--n_epochs', type=int, default=1000, help='the number of epochs')
parser.add_argument('--n_disp_ep', type=int, default=10, help='the number of epochs to validate the model')
parser.add_argument('--save', action='store_true', help='if True, save the results into the workspace in the specified folder')
parser.add_argument('--exp_path', type=str, default='/content/drive/MyDrive/experiments/nbrdf-meta_sampler/', help='the experiment folder')
parser.add_argument('--readme', type=str, default='This is a readme', help='The readme file that will be written into the workspace')
parser.add_argument('--train_model', action='store_true', help='if True, meta-train model')
parser.add_argument('--n_test_samples', type=int, default=512, help='the number of test samples for every BRDF task')
parser.add_argument('--RD_SEED', type=int, default=np.random.randint(0, 65535), help='the random seed used for reproducible experiments')
# parser.add_argument('--SAMPLER_SEED', type=int, default=np.random.randint(0, 65535), help='the random seed used for reproducible experiments (sampler init part)')
args = parser.parse_args()
yaml = YAML()
if __name__=='__main__':
    if args.config_path != None:
        with open(args.config_path, "r") as stream:
            config = yaml.load(stream)
    else:
        config = vars(args)
        del config['config_path']
        # print(config)

    main(config)