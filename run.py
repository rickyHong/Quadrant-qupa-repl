# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import numpy as np
import tensorflow as tf
import os

from vae import VAE
from thirdparty import input_data
from train_eval import run_training

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data',
                    'Directory to save training/test data.')
flags.DEFINE_string('log_dir', './logs',
                    'Directory to save the checkpoints and summaries.')
flags.DEFINE_string('dataset', 'binarized_mnist',
                    'Dataset to run experiments. We support "omniglot" and "binarized_mnist" for now.')
flags.DEFINE_integer('L', 1,
                     'Number of stochastic layers. We used L=1,2,4 for directed posteriors and L=1 for undirected.')
flags.DEFINE_string('experiment', 'vae',
                    'Sets the type of experiment. Use either "vae" for generative models and '
                    '"struct" for the structured prediction problem.')
flags.DEFINE_string('baseline', 'dvaes_power',
                    'Baseline used in training. Select one from: '
                    'dvaes_power for DVAE# (power function), '
                    'pwl_relax for PWL relaxation, '
                    'gsm_relax for Concrete relaxation, '
                    'rbm_post for DVAE## (RBM posterior).'
                    )
flags.DEFINE_integer('k', 1,
                     'Number of samples used for the importance weighted bound in training.')
flags.DEFINE_integer('num_latents', 200,
                     'Number of latent stochastic units. We examined with 200 and 400.')
flags.DEFINE_integer('num_train_iter', int(1e6),
                     'Number of training iterations (steps).')
flags.DEFINE_integer('eval_iw_samples', 4000,
                     'Number of importance weighted samples used in the final evaluation')

FLAGS = flags.FLAGS


def get_config(mean_x, training_size):
    dataset = FLAGS.dataset
    num_iter = FLAGS.num_train_iter
    eval_iw_samples = FLAGS.eval_iw_samples
    k = FLAGS.k
    L = FLAGS.L
    experiment = FLAGS.experiment
    baseline = FLAGS.baseline
    num_units = FLAGS.num_latents
    data_dir = FLAGS.data_dir

    structure_pred = experiment == 'struct'
    use_iw = k > 1
    batch_size = 100
    # set config train
    if baseline == 'rbm_post':
        dist_type = 'dvaess_con'
        beta = 4. if dataset == 'binarized_mnist' else 3.
        num_det_layers_enc = 1 if structure_pred else 2
    elif baseline == 'dvaes_power':
        dist_type = 'dvaes_power'
        beta = 30.
        num_det_layers_enc = 0
    elif baseline == 'pwl_relax':
        dist_type = 'pwl_relax'
        beta = 4. if dataset == 'binarized_mnist' else 3.
        num_det_layers_enc = 0
    elif baseline == 'gsm_relax':
        dist_type = 'gsm_relax'
        beta = 10.
        num_det_layers_enc = 0
    else:
        raise NotImplementedError

    num_det_layers_recon = 1 if structure_pred else 2
    lr = 3e-4 if structure_pred else 3e-3

    num_units_per_layer = num_units // L
    config = {'dist_type': dist_type, 'weight_decay': 1e-4, 'num_latent_layers': L,
              'num_latent_units': num_units_per_layer, 'name': 'lay0', 'num_det_layers_enc': num_det_layers_enc,
              'num_det_units_enc': 200, 'weight_decay_enc': 1e-4,
              'beta': beta, 'use_qupa': True, 'batch_norm': True, 'rank': 0}

    config_recon = {'num_det_layers': num_det_layers_recon, 'num_det_units': 200,
                    'weight_decay_dec': 1e-4, 'name': 'recon', 'batch_norm': True}

    expr_id = '%s/%s/%d/%s/%d/%d' % (dataset, experiment, num_units, baseline, L, k)
    config_train = {'dataset': dataset, 'batch_size': batch_size, 'eval_batch_size': 1, 'k': k, 'lr': lr,
                    'k_iw': eval_iw_samples, 'num_iter': num_iter, 'use_iw': use_iw, 'data_dir': data_dir,
                    'subset_eval': False, 'is_struct_pred': structure_pred, 'expr_id': expr_id}

    config_train['mean_x'] = mean_x
    config_train['training_size'] = training_size

    print config_train
    print config
    print config_recon

    return config_train, config, config_recon


def main(argv):
    data_sets = input_data.read_data_set(FLAGS.data_dir, dataset=FLAGS.dataset)

    # we need per pixel mean for normalizing input.
    mean_x = np.mean(data_sets.train._images, axis=0)
    training_size = data_sets.train.num_examples

    # get configurations
    config_train, config, config_recon = get_config(mean_x, training_size)
    log_dir = os.path.join(FLAGS.log_dir, config_train['expr_id'])

    vae = VAE(num_input=784, config=config, config_recon=config_recon, config_train=config_train)
    run_training(vae, cont_train=False, config_train=config_train, log_dir=log_dir)


if __name__ == '__main__':
    tf.app.run()

