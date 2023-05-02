# Learning to Learn and Sample BRDFs

This is the official code implementation for our paper [Liu, C., Fischer, M. & Ritschel, T. Learning to Learn and Sample BRDFs. Preprint at <https://doi.org/10.48550/arXiv.2210.03510> (2022).].

## Setup

After cloning this repo,

```bash
git clone https://github.com/ryushinn/meta-sampling.git && cd meta-sampling/
```

it would be easy to configure everything by running these following scripts.

### Environment

We recommend using [Anaconda](https://www.anaconda.com/) to setup the environment

```bash
conda env create -f environment.yaml
```

### Data

The necessary data, in the minimal requirement for running our repo, can be downloaded using this script:

```bash
bash scripts/download_data.sh
```

In case that the script failed due to network issues, you can download them manually:

- download [MERL BRDF dataset](https://www.dropbox.com/sh/yjt3bczfy52gb7o/AADvG_FhncJL59HgGOKxbE7Ya/brdfs) into `data/brdfs/`;
- download [pretrained models](https://drive.google.com/file/d/1AkHjQhPSo7QDTBaPhrI9uHdP2s_u7QYo/view?usp=share_link) into `data/meta-models/`;
- download [trained samplers](https://drive.google.com/file/d/1NQ_ZVF5dQnFdFALKlipkYbNRj_MQwa3P/view?usp=share_link) into `data/meta-samplers/`.

## Run

As illustrated by Algorithm 1 in the paper, our pipeline generally runs in two stages: 1. meta-model and 2. meta-sampler.

Here we provide scripts to easily run in this framework and reproduce the main experiments.

### Meta-train models & samplers

`scripts/meta_models.sh` offers a quick configuration of meta-model experiments, while `scripts/meta_samplers.sh` is the counterpart of meta-sampler experiments.

It is expected to run them in order:

```bash
bash scripts/meta_models.sh
bash scripts/meta_samplers.sh
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
