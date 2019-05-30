## Learning Undirected Posteriors by Backpropagation through MCMC Updates

This repository offers the Tensorflow implementation of undirected posterior variational autoencoders presented in
[this paper](https://arxiv.org/abs/1901.03440). This repository can be used to reproduce all the results presented in 
the paper (Table 1, 2 and 3) for both binarized MNIST and OMNIGLOT.

This repo is mainly based on our earlier implementation available [here](https://github.com/QuadrantAI/dvae).

For sampling from Boltzmann priors, population annealing (PA) algorithms is used. We rely on the
sampling library QuPA which was released by [Quadrant](http://quadrant.ai). 
You can have access to this library [here](https://try.quadrant.ai/qupa).
For sampling from Boltzmann posteriors, persistent contrastive divergence (PCD) is implemented in Tensorflow.

<br/>

## Running the Training/Evaluation Code
The main train/evaluation script can be run locally using the following command: 

```bash
python run.py \
    --log_dir=${PATH_TO_LOG_DIR} \
    --data_dir=${PATH_TO_DATA_DIR}
```

If you don't have the datasets locally, the scripts will download them automatically to the data directory.

The following flags are introduced in order to run the settings reported in the paper:
1. `--dataset` specifies the dataset used for the experiment. Currently, we support `omniglot` and `binarized_mnist`.
2. `--baseline` sets the type of objective function used for training. This corresponds to different columns 
in Table 1. You can use `dvaes_power` for DVAE# (power), `pwl_relax` for the PWL relaxation, `gsm_relax` for the Concrete 
relaxation, and `rbm_post` for DVAE## (RBM posterior).
3. `--num_latents` sets the number of latent stochastic units. We examined 200 and 400.
4. `--experiment` set the experiment. Use `vae` for the generative models in Table 1 and 2 or use `struct` for the 
structured prediction problem.
5. `--L` sets number of stochastic layers. We used L=1,2,4 for directed posteriors and L=1 for undirected.
6. `--k` specifies the number of samples used for estimating the variational bound in the case of DVAE/DVAE++ 
and the importance weighted bound in the case of DVAE#. 

Example:
```bash
python run.py \
    --log_dir=${PATH_TO_LOG_DIR} \
    --data_dir=${PATH_TO_DATA_DIR} \
    --L=1 \
    --baseline=rbm_post \
    --dateset=binarized_mnist \ 
    --experiment=vae \
    --k=1 \
    --num_latents=200 \
```
<br />

## Running Tensorboard

You can monitor the progress of training and the performance on the validation and test datasets using tensorboard.
Run the following command to start tensorboard on the log directory:

```bash
tensorboard --logdir=${PATH_TO_LOG_DIR}
```

<br />

## Prerequisites
Make sure that you have:
* Python (version 2.7 or higher)
* Numpy
* Scipy
* [QuPA](https://try.quadrant.ai/qupa) 
* Tensorflow (The version should be compatible with QuPA, we tested with Tensorflow 1.12.0)

<br/>

## Change Logs


##### July 13, 2018

First release including DVAE#, PWL, Concrete and DVAE##


<br/>

## Citation

if you use this code in your research, please cite us:
```
@article{vahdat2019undirected,
  title={Learning Undirected Posteriors by Backpropagation through {MCMC} Updates},
  author={Vahdat, Arash and Andriyash, Evgeny and Macready, William G.},
  journal={arXiv preprint arXiv:1901.03440},
  year={2019}
}
```
<br/>

## Contributors

Arash Vahdat, Evgeny Andriyash
