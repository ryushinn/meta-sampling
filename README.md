<h1>
  <a style="color: inherit;" href="https://ryushinn.github.io/metasampling">
    Learning to Learn and Sample BRDFs
  </a>
</h1>

This repo provides the official code implementation and related data for our paper

> [**Learning to Learn and Sample BRDFs**](https://ryushinn.github.io/metasampling)  
> by [Chen Liu](https://ryushinn.github.io/), [Michael Fischer](https://mfischer-ucl.github.io/) and [Tobias Ritschel](http://www.homepages.ucl.ac.uk/~ucactri/)  
> in Eurographics 2023

For more details, please check out \([Paper](https://arxiv.org/pdf/2210.03510.pdf), [Project Page](https://ryushinn.github.io/metasampling)\)!

![repo-illustration](repo.gif)

## Setup

After cloning this repo,

```bash
git clone https://github.com/ryushinn/meta-sampling.git && cd meta-sampling/
```

it would be easy to configure everything by running following scripts.

### Environment

We recommend using [Anaconda](https://www.anaconda.com/) to setup the environment

```bash
conda env create -n meta-sampling -f environment.yaml
conda activate meta-sampling
```

By default, this command installs cpu-only pytorch. **If you are using CUDA machines, please select the correct version of CUDA support manually in `environment.yaml`.**

Or you can download by running commands as instructed [here](https://pytorch.org/get-started/previous-versions/). But please note that we didn't test for this case.

### Data

The necessary data, in the minimal requirement for running our repo, can be downloaded using this script:

```bash
bash scripts/download_data.sh
```

In case that the script failed due to network issues, you can download them manually:

- download [MERL BRDF dataset](https://www.dropbox.com/sh/yjt3bczfy52gb7o/AADvG_FhncJL59HgGOKxbE7Ya/brdfs) into `data/brdfs/`;
- download [pretrained models](https://drive.google.com/file/d/1AkHjQhPSo7QDTBaPhrI9uHdP2s_u7QYo/view?usp=share_link) into `data/meta-models/`;
- download [trained samplers](https://drive.google.com/file/d/1NQ_ZVF5dQnFdFALKlipkYbNRj_MQwa3P/view?usp=share_link) into `data/meta-samplers/`.

Briefly, `data/brdfs` contains 100 isotropic measured BRDFs from MERL dataset and we randomly choose 80 of them as our training dataset.

`data/meta-models` is meta-learned initializations and learning rates for the three nonlinear models `Neural BRDF`, `Cooktorrance`, and `Phong`. Besides, there are 5 precomputed components for `PCA` model, obtained by running [NJR15 codebase](https://brdf.compute.dtu.dk/#navbar-code) over the our training dataset. Note that there are `80 * 3 = 240` PCs but only the first 5 are used in our PCA model.

`data/meta-samplers` is those optimal samples for each model. The number ranges from 1 to 512.

## Run

As illustrated by Algorithm 1 in the paper, our pipeline generally runs in two stages: 1. meta-model and 2. meta-sampler.

Here we provide scripts to easily run in this framework and reproduce the main experiments.

### Meta-train models & samplers

`scripts/meta_model.sh` offers a quick configuration of meta-model experiments, while `scripts/meta_sampler.sh` is the counterpart of meta-sampler experiments.

It is expected to run them in order:

```bash
bash scripts/meta_model.sh
bash scripts/meta_sampler.sh
```

By default, this will run for `Neural BRDF` model. But in the scripts the model being fit is modifiable and can be set to one of `Phong`, `Neural BRDF`, and `Cooktorrance`.

### Classic fitting

There is also a script `scripts/classic.sh` for simply fitting models to BRDF without any "meta training", which is called `classic` method in the paper.

```bash
bash script/classic.sh
```

The `classic` mode only has access to limited resources (1 \~ 512 samples and 20 learning iterations) to fit models.
In the contrary, the `overfit` mode represents the fitting process with sufficient samples and iterations.

## Citation

Please consider citing as follows if you find our paper and repo useful:

```bibtex
@article{liu2022learning,
  title={Learning to Learn and Sample BRDFs},
  author={Liu, Chen and Fischer, Michael and Ritschel, Tobias},
  journal={arXiv preprint arXiv:2210.03510},
  year={2022}
}
```
