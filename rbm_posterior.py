# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

from __future__ import print_function

import tensorflow as tf
import numpy as np

import qupa
from dist_util import sample_concrete, sample_through_pwlinear
from nets import add_vars_to_collection


class RBMParam:
    def __init__(self):
        self.num_var1 = None
        self.num_var2 = None
        self.num_var = None
        self.b1 = None
        self.b2 = None
        self.w = None

    @staticmethod
    def get_rbm_param_nn(param, posterior, rank, is_training, collections):
        """
        Create a RBMParam object from the output of neural network
        
        Args:
            param: a tensor containing all the parameters
            posterior: posterior object
            rank: unused (set to 0)
            is_training: a flag indicating if we are building the training graph
            collections: a string 
        """
        assert isinstance(posterior, PosteriorRBM)
        rbm_param = RBMParam()
        rbm_param.num_var1 = posterior.num_var1
        rbm_param.num_var2 = posterior.num_var2
        rbm_param.num_var = posterior.num_var

        rbm_param.b1 = param[0][:, :rbm_param.num_var1]
        rbm_param.b2 = param[0][:, rbm_param.num_var1:rbm_param.num_var]

        w = param[0][:, rbm_param.num_var:]
        if rank == 0:
            with tf.variable_scope('rbm_post', reuse=not is_training):
                J = tf.layers.batch_normalization(w, momentum=0.99, epsilon=1e-2, training=is_training, name='bn_w')
                t = 2.  # fixed

            J = 1.5 * J / np.sqrt(posterior.num_var1)
            J = tf.reshape(J, (-1, rbm_param.num_var1, rbm_param.num_var2))
            rbm_param.w = 2. * t * J
            rbm_param.b1 = t * (rbm_param.b1 - tf.reduce_sum(J, axis=2))
            rbm_param.b2 = t * (rbm_param.b2 - tf.reduce_sum(J, axis=1))

        if is_training:
            add_vars_to_collection('rbm_post', collections)

        return rbm_param

    @staticmethod
    def get_rbm_param_from_param(b1, b2, w):
        """
         Creates the RBMParam object from b1, b2, w directly.
         
         Args:
            b1: bias1
            b2: bias2
            w: pairwise terms
        """
        rbm_param = RBMParam()
        rbm_param.num_var1 = b1.get_shape().as_list()[1]
        rbm_param.num_var2 = b2.get_shape().as_list()[1]
        rbm_param.num_var = rbm_param.num_var1 + rbm_param.num_var2

        rbm_param.b1 = b1
        rbm_param.b2 = b2
        rbm_param.w = w
        return rbm_param


class PosteriorRBM:
    def __init__(self, training_size, num_var1, num_var2, num_gibbs_iter, num_eval_k, num_train_k, name='post_rbm'):
        """
            This class is an abstract class implementing the super class for undirected posteriors.
            
            Args:
                training_size: an integer indicating the size of training dataset.
                num_var1: number of variables in one side of the bipartite graph
                num_var2: number of variables in the other side of the bipartite graph
                num_gibbs_iter: number of Gibbs sweeps used for sampling (hard Gibbs sweeps, s in the paper)
                num_eval_k: number of samples generated for the evaluation
                num_train_k: number of samples from each RBM used for training
                name: used as the namescope of variables/operations.
        """
        self.prior = None
        self.num_var1 = num_var1
        self.num_var2 = num_var2
        self.num_var = num_var1 + num_var2
        self.training_size = training_size
        self.num_gibbs_iter = num_gibbs_iter
        self.num_train_k = num_train_k
        self.name = name
        self.num_ais_betas = 500
        self.num_ais_samples = num_eval_k
        self.check_posterior_modes = True
        self.mf_variable = None
        if self.check_posterior_modes:
            self.mf_variable = tf.Variable(np.zeros((num_eval_k, self.num_var)), trainable=False, dtype=tf.float32,
                                           name='posterior_mean_fields', collections=[tf.GraphKeys.LOCAL_VARIABLES])

        self.posterior_samples = None
        self.initialize_samples()

    def initialize_samples(self):
        """
        This function initializes the persistent chains. 
        """
        init_samples = np.random.randint(0, 2, size=(self.training_size, self.num_train_k, self.num_var1))
        self.posterior_samples = tf.Variable(init_samples, dtype=tf.float32, trainable=False)

    def general_gibbs(self, rbm_param, index):
        """ This function retrieves the chains, applies forward Gibbs sampling and stores the updated chains back.
        
        Args:
            rbm_param: An RBMParam object containing RBM parameters
            index: indices of current batch in the whole training dataset.
        """
        init_samples = tf.gather(self.posterior_samples, index)
        init_samples = tf.reshape(init_samples, [-1, self.num_var1])
        new_samples = self.forward_gibbs_hard(rbm_param, init_samples)
        updated_samples = tf.reshape(new_samples[:, :self.num_var1], [-1, self.num_train_k, self.num_var1])
        update_chains = tf.scatter_update(self.posterior_samples, index, updated_samples)
        with tf.control_dependencies([update_chains]):
            new_samples = tf.identity(new_samples)

        return new_samples

    def forward_gibbs_hard(self, rbm_param, init_samples):
        """ Runs the Gibbs sampling steps starting from initial samples.
         
         Args:
             rbm_param: An RBMParam object containing RBM parameters
             init_samples: persistent chains
        """
        raise NotImplementedError

    def kl_from_this(self, rbm_param, rbm, samples, log_z, is_training):
        """ computes kl distance from RBM posterior to the prior.

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             rbm: RBM prior
             samples: samples drawn from posterior
             log_z: a tensor containing log_z values for the RBM posteriors
             is_training: a flag indicating whether we are building training graph or not.
        """
        raise NotImplementedError

    def energy(self, rbm_param, samples):
        """ computes energy of samples using the rbm parameters
         
         Args:
             rbm_param: An RBMParam object containing RBM parameters
             samples: samples drawn from posterior
        """
        raise NotImplementedError

    def log_prob(self, rbm_param, samples, log_z, stop_grad):
        """ Computes log probability of samples under the RBM posterior distribution

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             samples: samples drawn from posterior
             log_z: a tensor containing log_z values for the RBM posteriors
             stop_grad: a flag indicating whether we should apply stop gradient to the parameters of the posterior or not.
        """
        raise NotImplementedError

    def ais_samples_log_z(self, rbm_param):
        """ Uses AIS algorithm to samples from posterior. Used at the evaluation phase.
        
         Args:
             rbm_param: An RBMParam object containing RBM parameters
        """
        raise NotImplementedError


class ConcreteRBM(PosteriorRBM):
    def __init__(self, training_size, num_var1, num_var2, num_gibbs_iter, num_eval_k, num_train_k, beta):
        """
            This class is an abstract class implementing the super class for undirected posteriors.

            Args:
                training_size: an integer indicating the size of training dataset.
                num_var1: number of variables in one side of the bipartite graph
                num_var2: number of variables in the other side of the bipartite graph
                num_gibbs_iter: number of Gibbs sweeps used for sampling (hard Gibbs sweeps, s in the paper)
                num_eval_k: number of samples generated for the evaluation
                num_train_k: number of samples from each RBM used for training
                beta: inverse temperature parameter used for continuous relaxation
        """
        PosteriorRBM.__init__(self, training_size, num_var1, num_var2, num_gibbs_iter, num_eval_k, num_train_k)
        self.beta = beta
        self.use_concrete = False
        self.num_sweep = 1

    def forward_gibbs_relax(self, rbm_param, init_samples):
        """ Runs the Gibbs sampling steps starting from initial samples. This function implements relaxed
        Gibbs sampling steps.

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             init_samples: persistent chains
        """
        samples1 = tf.expand_dims(init_samples[:, :self.num_var1], 1)
        samples2 = tf.expand_dims(init_samples[:, self.num_var1:], 1)
        b1 = tf.expand_dims(rbm_param.b1, 1)
        b2 = tf.expand_dims(rbm_param.b2, 1)
        w = rbm_param.w
        w_tran = tf.transpose(w, [0, 2, 1])
        if self.num_sweep >= 1:
            with tf.name_scope('%s_relax_gibbs_iterations' % self.name):
                for iter in range(self.num_sweep):
                    scope_name = 'gibbs_iter_%d' % iter
                    with tf.name_scope(scope_name):
                        alpha2 = tf.matmul(samples1, w) + b2
                        if self.use_concrete:
                            samples2 = sample_concrete(alpha2, self.beta)
                        else:
                            samples2, _ = sample_through_pwlinear(alpha2, self.beta)
                        alpha1 = tf.matmul(samples2, w_tran) + b1
                        q1 = tf.nn.sigmoid(alpha1)
                        if self.use_concrete:
                            samples1 = sample_concrete(alpha1, self.beta)
                        else:
                            samples1, _ = sample_through_pwlinear(alpha1, self.beta)

                samples1, samples2 = tf.squeeze(samples1, 1), tf.squeeze(samples2, 1)
                samples = tf.concat(axis=1, values=[samples1, samples2])
                samples_mf = tf.concat(axis=1, values=[tf.squeeze(q1, 1), samples2])  # rao-blackwellized in one side

            return samples, samples_mf

    def forward_gibbs_hard(self, rbm_param, init_samples):
        """ Runs the Gibbs sampling steps starting from initial samples. This function implements the hard (binary)
        Gibbs sampling steps.

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             init_samples: persistent chains
        """
        with tf.name_scope('%s_hard_gibbs_iterations' % self.name):
            samples1 = tf.expand_dims(init_samples, 1)
            b1 = tf.expand_dims(rbm_param.b1, 1)
            b2 = tf.expand_dims(rbm_param.b2, 1)
            w = rbm_param.w
            w_tran = tf.transpose(w, [0, 2, 1])
            for iter in range(self.num_gibbs_iter):
                scope_name = 'gibbs_iter_%d' % iter
                with tf.name_scope(scope_name):
                    alpha2 = tf.matmul(samples1, w) + b2
                    samples2 = tf.to_float(tf.random_uniform(shape=tf.shape(alpha2)) < tf.nn.sigmoid(alpha2))
                    samples2 = tf.stop_gradient(samples2)
                    alpha1 = tf.matmul(samples2, w_tran) + b1
                    samples1 = tf.to_float(tf.random_uniform(shape=tf.shape(alpha1)) < tf.nn.sigmoid(alpha1))
                    samples1 = tf.stop_gradient(samples1)

            samples1, samples2 = tf.squeeze(samples1, 1), tf.squeeze(samples2, 1)
            samples = tf.concat(axis=1, values=[samples1, samples2])
            samples = tf.stop_gradient(samples)
        return samples

    def kl_from_this(self, rbm_param, rbm, samples_mf, log_z_posterior, is_training):
        """ computes kl distance from RBM posterior to the prior.

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             rbm: RBM prior
             samples: samples drawn from posterior
             log_z_posterior: a tensor containing log_z values for the RBM posteriors
             is_training: a flag indicating whether we are building training graph or not.
        """
        entropy_term = - self.log_prob(rbm_param, samples_mf, log_z_posterior, stop_grad=True)
        log_z_prior = rbm.log_z_train if is_training else rbm.log_z_value
        cross_entropy_term = rbm.energy_tf(samples_mf) + log_z_prior

        return cross_entropy_term - entropy_term

    def energy(self, rbm_param, samples):
        """ computes energy of samples using the rbm parameters

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             samples: samples drawn from posterior
        """
        samples1 = samples[:, :self.num_var1]
        samples2 = samples[:, self.num_var1:]

        neg_energy = tf.reduce_sum(rbm_param.b1 * samples1, axis=1) + tf.reduce_sum(rbm_param.b2 * samples2, axis=1) + \
                     tf.reduce_sum(tf.squeeze(tf.matmul(tf.expand_dims(samples1, 1), rbm_param.w), 1) * samples2, axis=1)

        return - neg_energy

    def log_prob(self, rbm_param, samples, log_z, stop_grad):
        """ Computes log probability of samples under the RBM posterior distribution

         Args:
             rbm_param: An RBMParam object containing RBM parameters
             samples: samples drawn from posterior
             log_z: a tensor containing log_z values for the RBM posteriors
             stop_grad: a flag indicating whether we should apply stop gradient to the parameters of the posterior or not.
        """
        if stop_grad:
            b1_sg = tf.stop_gradient(rbm_param.b1)
            b2_sg = tf.stop_gradient(rbm_param.b2)
            w_sg = tf.stop_gradient(rbm_param.w)
            rbm_param = RBMParam.get_rbm_param_from_param(b1_sg, b2_sg, w_sg)

        return - self.energy(rbm_param, samples) - log_z

    def ais_samples_log_z(self, rbm_param):
        """ Uses AIS algorithm to samples from posterior. Used at the evaluation phase.

         Args:
             rbm_param: An RBMParam object containing RBM parameters
        """
        a1 = tf.assert_equal(tf.shape(rbm_param.b1)[0], 1)
        a2 = tf.assert_equal(tf.shape(rbm_param.b2)[0], 1)
        a3 = tf.assert_equal(tf.shape(rbm_param.w)[0], 1)

        with tf.control_dependencies([a1, a2, a3]):
            b1 = tf.squeeze(rbm_param.b1, 0)
            b2 = tf.squeeze(rbm_param.b2, 0)
            w = tf.squeeze(rbm_param.w, 0)

        b = tf.concat(values=[b1, b2], axis=0)
        betas = tf.linspace(tf.constant(0.), tf.constant(1.), num=self.num_ais_betas)
        with tf.variable_scope(self.name):
            ais_log_z, info = qupa.ais.evaluation_log_z(
                b, w, init_biases=b, betas=betas, num_samples=self.num_ais_samples,
                extra_info=['samples', 'logz_stddev'])
            ais_samples = info['samples']
            logz_stddev = info['logz_stddev']

        # mean_field updates for checking the number of modes.
        if self.check_posterior_modes:
            with tf.name_scope('mf_gibbs_iterations'):
                mf1 = ais_samples[:, :self.num_var1]
                for iter in range(40):
                    scope_name = 'gibbs_iter_%d' % iter
                    with tf.name_scope(scope_name):
                        mf2 = tf.nn.sigmoid(tf.matmul(mf1, w) + b2)
                        mf1 = tf.nn.sigmoid(tf.matmul(mf2, w, transpose_b=True) + b1)
                mf = tf.concat(values=[mf1, mf2], axis=1)

                mf = tf.clip_by_value(mf, 1e-5, 1 - 1e-5)
                l = tf.log(mf / (1. - mf))
                kl = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=mf, logits=l), axis=1)
                kl -= tf.reduce_sum(mf1 * b1 + mf2 * b2, axis=1) + tf.reduce_sum(tf.matmul(mf1, w) * mf2, 1)
                kl += ais_log_z
                op = tf.print([tf.reduce_mean(kl), tf.reduce_max(kl), tf.reduce_min(kl), logz_stddev, ais_log_z])
                with tf.control_dependencies([op]):
                    self.mf_variable = self.mf_variable.assign(mf)

        return ais_samples, ais_log_z