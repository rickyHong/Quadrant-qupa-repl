# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from autoencoder import SimpleDecoder, SimpleEncoder, RBMEncoder
from dist_util import FactorialBernoulliUtil, MixtureGeneric, RelaxDist
from dist_util import sample_through_pwlinear, sample_concrete
from smoothing_util import PowerLaw
from util import get_global_step_var, Print, repeat_input_iw, get_structure_mask
from rbm import RBM, MarginalRBMType1Generic
from rbm_posterior import ConcreteRBM
from nets import FeedForwardNetwork


class VAE:
    def __init__(self, num_input, config, config_recon, config_train):
        """  This function initializes an instance of the VAE class. 
        Args:
            num_input: the length of observed random variable (x).
            config: a dictionary containing config. for the (hierarchical) posterior distribution and prior over z. 
            config_recon: a dictionary containing config. for the reconstruct function in the decoder p(x | z).
            config_train: a dictionary containing config. training (hyperparameters).
        """
        np.set_printoptions(threshold=10)
        Print(str(config))
        Print(str(config_recon))
        Print(str(config_train))

        self.num_input = num_input
        self.config = config              # configuration dictionary for approx post and prior on z
        self.config_recon = config_recon  # configuration dictionary for reconstruct function p(x | z)
        self.config_train = config_train  # configuration dictionary for training hyper-parameters

        # bias term on the visible node
        self.train_bias = -np.log(1. / np.clip(self.config_train['mean_x'], 0.001, 0.999) - 1.).astype(np.float32)
        self.entropy_lower_bound = 0.05

        self.dist_type = config['dist_type']  # flag indicating whether we have rbm prior.
        tf.summary.scalar('beta', config['beta'])
        self.encoder_type = 'hierarchical'
        self.is_struct_pred = config_train['is_struct_pred']

        # define DistUtil classes that will be used in posterior and prior.
        if self.dist_type == "dvaes_power":                                            # DVAE# (power)
            dist_util = MixtureGeneric
            self.smoothing_dist = PowerLaw(params={'beta': self.config['beta']})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/lambda', self.smoothing_dist._lambda)
        elif self.dist_type == "pwl_relax":                                            # PWL relaxtion
            dist_util = RelaxDist
            dist_util_param = {'beta': self.config['beta'], 'smoothing_fun': sample_through_pwlinear}
            tf.summary.scalar('posterior/beta', dist_util_param['beta'])
        elif self.dist_type == "gsm_relax":                                             # Gumbel-Softmax relaxtion
            dist_util = RelaxDist
            dist_util_param = {'beta': self.config['beta'], 'smoothing_fun': sample_concrete}
            tf.summary.scalar('posterior/beta', dist_util_param['beta'])
        elif self.dist_type == "dvaess_con":
            self.encoder_type = 'rbm'
            num_var = self.config['num_latent_units'] // 2
            self.posterior = ConcreteRBM(training_size=config_train['training_size'], num_var1=num_var,
                                         num_var2=num_var, num_gibbs_iter=10, beta=self.config['beta'],
                                         num_eval_k=self.config_train['k_iw'], num_train_k=self.config_train['k'])
        else:
            raise ValueError('self.dist_type=%s is unknown' % self.dist_type)

        # define p(z)
        self.prior = self.define_prior()

        if self.encoder_type == 'hierarchical':
            # create encoder for the first level.
            num_hidden_pre = [200] * 1 if self.is_struct_pred else [200] * 2
            self.pre_process_net = FeedForwardNetwork(
                num_input, num_hiddens=num_hidden_pre, num_output=None, name='pre_proc', weight_decay_coeff=1e-4,
                output_split=1, use_batch_norm=True, collections='q_collections')
            self.encoder = SimpleEncoder(num_input=200, config=config, dist_util=dist_util,
                                         dist_util_param=dist_util_param)
        else:
            self.pre_process_net = None
            self.encoder = RBMEncoder(num_input=num_input, config=config, posterior_rbm=self.posterior)

        # create encoder and decoder for lower layers.
        num_latent_units = self.config['num_latent_units'] * self.config['num_latent_layers']
        self.decoder = SimpleDecoder(num_latent_units=num_latent_units, num_output=num_input, config_recon=config_recon)

    def should_compute_log_z(self):
        return isinstance(self.prior, RBM)

    def define_prior(self):
        """ Defines the prior distribution over z. The prior will be an RBM or Normal prior based on self.dist_type.
         
        Returns:
            a DistUtil object representing the prior distribution.
        """
        # set up the rbm
        num_var1 = self.config['num_latent_units'] * self.config['num_latent_layers'] // 2
        wd = self.config['weight_decay']
        if self.dist_type in {'pwl_relax', 'gsm_relax'}:
            rbm_prior = RBM(num_var1=num_var1, num_var2=num_var1, num_samples=1000, weight_decay=wd,
                            kld_term='kld_as_function', use_qupa=self.config['use_qupa'])
        elif self.dist_type in {'dvaess_con'}:
            rbm_prior = RBM(num_var1=num_var1, num_var2=num_var1, num_samples=1000, weight_decay=wd,
                            kld_term=self.dist_type, use_qupa=self.config['use_qupa'])
        elif self.dist_type in {'dvaes_power'}:
            rbm_prior = MarginalRBMType1Generic(num_var1=num_var1, num_var2=num_var1, num_samples=1000, weight_decay=wd,
                                                use_qupa=self.config['use_qupa'], smoothing_dist=self.smoothing_dist)
        else:
            raise NotImplementedError

        return rbm_prior

    def generate_samples(self, num_samples):
        """ It will randomly sample from the model using ancestral sampling. It first generates samples from p(z_0).
        Then, it generates samples from the hierarchical distributions p(z_j|z_{i < j}). Finally, it forms p(x | z_i).  
        
         Args:
             num_samples: an integer value representing the number of samples that will be generated by the model.
        """
        if isinstance(self.prior, RBM):
            prior_samples = self.prior.samples
            prior_samples = tf.slice(prior_samples, [0, 0], [num_samples, -1])
        else:
            raise NotImplementedError

        output_activations = self.decoder.generator(prior_samples)
        output_activations[0] = output_activations[0] + self.train_bias
        output_dist = FactorialBernoulliUtil(output_activations)
        output_samples = tf.nn.sigmoid(output_dist.logit_mu)
        return output_samples

    def elbo_terms(self, input, posterior, post_samples, log_z, k, is_training, batch_norm_update,
                   post_samples_mf=None):
        # create features for the likelihood p(x|z)
        output_activations = self.decoder.reconstruct(post_samples, is_training, batch_norm_update)
        # add data bias
        output_activations[0] = output_activations[0] + self.train_bias
        # form the output dist util.
        output_dist = FactorialBernoulliUtil(output_activations)
        # create the final output
        output = tf.nn.sigmoid(output_dist.logit_mu)
        output = self.mix_output_with_input(input, output)

        # concat all the samples
        post_samples_concat = tf.concat(axis=-1, values=post_samples)
        # post_samples_concat = post_samples_mf  # remove this, it uses MF instead of samples

        kl, log_q, log_p = 0., 0., 0.
        if self.config_train['use_iw'] and is_training and k > 1:
            Print('Using IW Obj.')
            if self.encoder_type == 'hierarchical':
                for samples, factorial in zip(post_samples, posterior):
                    log_q += factorial.log_prob(samples, stop_grad=True)

                log_p = self.prior.log_prob(post_samples_concat, stop_grad=False, is_training=is_training)
            else:
                log_q = self.posterior.log_prob(posterior, post_samples_concat, log_z, stop_grad=True)
                log_p = - self.prior.energy_tf(post_samples_concat) - self.prior.log_z_train

            if self.is_struct_pred:
                kl = log_q
                log_p, log_q = 0., 0.
        else:
            Print('Using VAE Obj.')
            if self.is_struct_pred:  # add only the entropy loss to the objective function
                log_q = 0.
                if self.encoder_type == 'hierarchical':
                    for samples, factorial in zip(post_samples, posterior):
                        log_q += factorial.log_prob(samples, stop_grad=True, is_training=True)
                else:
                    log_q = self.posterior.log_prob(posterior, post_samples_concat, log_z, stop_grad=True)
                kl = log_q
            else:
                # compute KL only for VAE case
                if self.encoder_type == 'hierarchical':
                    kl = self.prior.kl_dist_from(posterior, post_samples, is_training)
                elif self.encoder_type == 'rbm':
                    kl = self.posterior.kl_from_this(posterior, self.prior, post_samples_mf, log_z, is_training)

        # expected log prob p(x| z)
        cost = - output_dist.log_prob_per_var(input, stop_grad=False)
        cost = self.process_decoder_target(cost)
        cost = tf.reduce_sum(cost, axis=1)

        return kl, cost, output, log_p, log_q

    def neg_elbo(self, input, is_training, input_index=None, k=1):
        """ Defines the core operations that are used for both training and evaluation.
        
        Args:
            input: a 2D tensor containing current batch. 
            is_training: a boolean representing whether we are building the train or test computation graph.
            input_index: indices of current batch in the whole training dataset
            k: number of samples used for evaluating the objective function
             
        Returns:
            neg_elbo: is a scalar tensor containing negative EBLO computed for the batch. For training batch the KL
              coeff is applied.   
            output: a tensor representing p(x|z) that is created by a single sample z~q(z|x).  
            wd_loss: a scalar tensor containing weight decay loss for all the networks.
            log_p: a tensor of length batch size, representing the importance weights log p(x, z) / q(z|x). This
             will be used in the test batch for evaluating Log Likelihood.
        """
        # apply pre-processing
        masked_input = self.process_encoder_input(input)
        if is_training:
            tf.summary.image('masked input', tf.reshape(masked_input, [-1, 28, 28, 1]))
        if self.pre_process_net is not None:
            pre_process_input = masked_input - self.config_train['mean_x']
            encoder_input = self.pre_process_net.build_network(pre_process_input, is_training, batchnorm_update=is_training)
            encoder_input = encoder_input[0]
        else:
            # subtract mean from input
            encoder_input = masked_input - self.config_train['mean_x']

        input = repeat_input_iw(input, k)

        log_z = None
        post_samples_mf = None
        # form the encoder for z
        if self.encoder_type == 'hierarchical':
            # repeat the input if K > 1
            encoder_input = repeat_input_iw(encoder_input, k)
            posterior, post_samples = self.encoder.hierarchical_posterior(encoder_input, is_training)
            # convert list of samples to single tensor
            post_samples_concat = tf.concat(axis=-1, values=post_samples)

        elif self.encoder_type == 'rbm':
            if is_training:
                encoder_input = repeat_input_iw(encoder_input, k)
            posterior, post_samples, log_z = self.encoder.posterior(encoder_input, input_index, is_training)
            post_samples, post_samples_mf = post_samples
            post_samples_concat = post_samples

        # form the objective
        kl_coeff = self.kl_coeff_annealing(is_training)
        if self.config_train['use_iw'] and is_training and k > 1:
            kl, cost, output, log_p, log_q = self.elbo_terms(input, posterior, post_samples, log_z, k,
                                                             is_training, batch_norm_update=is_training,
                                                             post_samples_mf=post_samples_mf)

            log_iw = kl_coeff * (log_p - log_q) - cost
            log_iw_k = tf.reshape(log_iw, [-1, k])
            norm_w = tf.nn.softmax(log_iw_k)
            norm_w_squared = tf.square(norm_w)
            iw_loss_p = tf.reduce_sum(tf.stop_gradient(norm_w) * log_iw_k, axis=1)
            iw_loss_p = - tf.reduce_mean(iw_loss_p)
            if self.is_struct_pred:
                iw_loss_q = tf.reduce_sum(tf.stop_gradient(norm_w) * log_iw_k, axis=1)
            else:
                iw_loss_q = tf.reduce_sum(tf.stop_gradient(kl_coeff * norm_w_squared +
                                                           (1. - kl_coeff) * norm_w) * log_iw_k, axis=1)
            iw_loss_q = - tf.reduce_mean(iw_loss_q)

            # additional entropy term for structure prediction
            if self.is_struct_pred:
                kl = tf.reduce_mean(tf.reshape(kl, [-1, k]))
                iw_loss_q += kl_coeff * kl

            neg_elbo = (iw_loss_p, iw_loss_q)
        else:
            kl, cost, output, _, _ = self.elbo_terms(input, posterior, post_samples, log_z, k, is_training,
                                                     batch_norm_update=is_training, post_samples_mf=post_samples_mf)
            neg_elbo_per_sample = kl_coeff * kl + cost
            neg_elbo_per_sample = tf.reshape(neg_elbo_per_sample, [-1, k])
            neg_elbo_per_sample = tf.reduce_mean(neg_elbo_per_sample, axis=1)
            neg_elbo = tf.reduce_mean(neg_elbo_per_sample)

        # weight decay loss
        pre_process_wd_loss = self.pre_process_net.get_weight_decay_loss() if self.pre_process_net is not None else 0.
        enc_wd_loss = self.encoder.get_weight_decay()
        dec_wd_loss = self.decoder.get_weight_decay()
        prior_wd_loss = self.prior.get_weight_decay() if isinstance(self.prior, RBM) else 0
        wd_loss = enc_wd_loss + dec_wd_loss + prior_wd_loss + pre_process_wd_loss
        if is_training:
            tf.summary.scalar('weigh decay/encoder', enc_wd_loss)
            tf.summary.scalar('weigh decay/decoder', dec_wd_loss)
            tf.summary.scalar('obj/recon_loss', tf.reduce_mean(cost))
            tf.summary.scalar('obj/kl', tf.reduce_mean(kl))
            tf.summary.scalar('weigh decay/total', wd_loss)

        # compute importance weights
        if not is_training:
            if self.is_struct_pred:
                log_iw = 0.
            else:
                # log importance weight log p(z) - log q(z|x)
                log_iw = self.prior.log_prob(post_samples_concat, stop_grad=False, is_training=is_training)
                if self.encoder_type == 'hierarchical':
                    for i in range(len(posterior)):
                        log_iw -= posterior[i].log_prob(post_samples[i], stop_grad=False, is_training=is_training)
                elif self.encoder_type == 'rbm':
                    log_iw -= self.posterior.log_prob(posterior, post_samples_concat, log_z, stop_grad=False)

            # add p(x|z)
            log_iw -= cost
            log_p = tf.reduce_logsumexp(log_iw) - tf.log(tf.to_float(k))
        else:
            log_p = None

        return neg_elbo, output, wd_loss, log_p

    def kl_coeff_annealing(self, is_training):
        """ defines the coefficient used for annealing the KL term. It return 1 for the test graph but, a value
        between 0 and 1 for the training graph.
        
        Args:
            is_training: a boolean flag indicating whether the network is part of train or test graph. 

        Returns:
            kl_coeff: a scalar (non-trainable) tensor containing the kl coefficient.
        """
        global_step = get_global_step_var()
        if is_training:
            if self.is_struct_pred:
                # anneal the entropy coefficient in 60% iterations.
                max_epochs = 0.5 * self.config_train['num_iter']
                kl_coeff = tf.maximum(1. - tf.to_float(global_step) / max_epochs, self.entropy_lower_bound)
            else:
                # anneal the KL coefficient in 30% iterations.
                max_epochs = 0.3 * self.config_train['num_iter']
                kl_coeff = tf.minimum(tf.to_float(global_step) / max_epochs, 1.)

            tf.summary.scalar('kl_coeff', kl_coeff)
        else:
            kl_coeff = 1.

        return kl_coeff

    def training(self, neg_elbo, wd_loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.

        Args:
            neg_elbo: neg_elbo tensor, from neg_elbo().
            wd_loss: weight decay loss.

        Returns:
            train_op: The Op for training.
        """
        global_step = get_global_step_var()
        base_lr = self.config_train['lr']
        lr_values = [base_lr / 10, base_lr, base_lr / 3, base_lr / 10, base_lr / 33]
        boundaries = np.array([0.02, 0.6, 0.75, 0.95]) * self.config_train['num_iter']
        boundaries = [int(b) for b in boundaries]
        lr = tf.train.piecewise_constant(global_step, boundaries, lr_values)

        tf.summary.scalar('learning_rate', lr)
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.config_train['use_iw'] and self.config_train['k'] > 1:
                iw_loss_p, iw_loss_q = neg_elbo
                grads_vars_q = optimizer.compute_gradients(iw_loss_q + wd_loss, var_list=tf.get_collection('q_collections'))
                grads_vars_p = optimizer.compute_gradients(iw_loss_p + wd_loss, var_list=tf.get_collection('p_collections'))
                grads_vars = grads_vars_p + grads_vars_q
                train_op = optimizer.apply_gradients(grads_vars, global_step=global_step)
            else:
                loss = neg_elbo + wd_loss
                train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def process_encoder_input(self, encoder_input):
        if self.is_struct_pred:
            mask = get_structure_mask(self.num_input)
            return encoder_input * mask
        else:
            return encoder_input

    def process_decoder_target(self, target):
        if self.is_struct_pred:
            mask = get_structure_mask(self.num_input)
            return target * (1. - mask)
        else:
            return target

    def mix_output_with_input(self, input, output):
        if self.is_struct_pred:
            mask = get_structure_mask(self.num_input)
            return output * (1. - mask) + input * mask
        else:
            return output
