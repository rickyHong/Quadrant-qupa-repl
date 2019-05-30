# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import numpy as np

import qupa
from qupa.pcd import PCD

from util import Print
from dist_util import DistUtil, FactorialBernoulliUtil
from smoothing_util import SmoothingDist


class RBM(DistUtil):
    def __init__(self, num_var1, num_var2, weight_decay, name='RBM', num_samples=100, num_gibbs_iter=40,
                 kld_term=None, use_qupa=False):
        """
        Initialize bias and weight parameters, create sampling operations (gibbs or QuPA). This class implements
        the KL divergence computation for DVAE and DVAE++.
        
        Args:
            num_var1:               number of vars of left side of RBM
            num_var2:               number of vars of right side of RBM
            weight_decay:           regularization for the weight matrix
            name:                   name
            num_samples:            number of RBM samples drawn in each iterations (used for computing log Z gradient)
            num_gibbs_iter:         number of gibbs step for pcd or mcmc sweeps for QuPA
            kld_term:               Use 'dvae_spike_exp' for DVAE, 'dvaepp_exp', 'dvaepp_power' for DVAE++, 
                                    'guassian_integral', 'marginal_type1' for DVAE#.
            use_qupa:               A boolean flag indicating whether QuPA will be used for sampling. Setting this
                                    variable to False will use PCD for sampling
        """
        assert kld_term in {'dvae_spike_exp', 'dvaepp_power', 'dvaepp_exp', 'guassian_integral', 'marginal_type1',
                            'marginal_type2', 'dvaess_con', 'kld_as_function'}, \
            'kld_term defined by %s in argument is not defined.' % kld_term
        self.kld_term = kld_term
        self.num_var1 = num_var1
        self.num_var2 = num_var2
        self.num_var = num_var1 + num_var2
        self.weight_decay = weight_decay
        self.name = name

        # bias on the left side
        self.b1 = tf.Variable(tf.zeros(shape=[self.num_var1, 1], dtype=tf.float32), name='bias1')
        tf.add_to_collection('p_collections', self.b1)
        # bias on the right side
        self.b2 = tf.Variable(tf.zeros(shape=[self.num_var2, 1], dtype=tf.float32), name='bias2')
        tf.add_to_collection('p_collections', self.b2)
        # pairwise weight
        self.w = tf.Variable(tf.zeros(shape=[self.num_var1, self.num_var2], dtype=tf.float32), name='pairwise')
        tf.add_to_collection('p_collections', self.w)

        # sampling options
        self.num_samples = num_samples
        self.use_qupa = use_qupa

        # concat b
        b = tf.concat(values=[tf.squeeze(self.b1), tf.squeeze(self.b2)], axis=0)

        if not self.use_qupa:
            Print('Using PCD')
            # init pcd class implemented in QuPA
            self.sampler = PCD(left_size=self.num_var1, right_size=self.num_var2,
                               num_samples=self.num_samples, dtype=tf.float32)
        else:
            Print('Using QuPA')
            # init population annealing class in QuPA
            self.sampler = qupa.PopulationAnnealer(left_size=self.num_var1, right_size=self.num_var2,
                                                   num_samples=self.num_samples, dtype=tf.float32)

        # This returns a scalar tensor with the gradient of log z. Don't trust its value.
        self.log_z_train = self.sampler.training_log_z(b, self.w, num_mcmc_sweeps=num_gibbs_iter)

        # This returns the internal log z variable in QuPA sampler. We will use this variable in evaluation.
        self.log_z_value = self.sampler.log_z_var

        # get always the samples after updating train log z
        with tf.control_dependencies([self.log_z_train]):
            self.samples = self.sampler.samples()

        # Define inverse temperatures used for AIS. Increasing the # of betas improves the precision of log z estimates.
        betas = tf.linspace(tf.constant(0.), tf.constant(1.), num=10000)
        # Define log_z estimation for evaluation.
        eval_logz = qupa.ais.evaluation_log_z(b, self.w, init_biases=None, betas=betas, num_samples=1024)

        # Update QuPA internal log z variable with the eval_logz
        self.log_z_update = self.log_z_value.assign(eval_logz)

    def energy_tf(self, samples):
        """
        Computes the energy function -b^T * z - z^T * w * z for the Boltzmann machine.
        Args:
            samples: matrix with size (num_samples * num_vars)

        Returns: 
            energy for each sample.
        """
        samples1 = tf.slice(samples, [0, 0], [-1, self.num_var1])
        samples2 = tf.slice(samples, [0, self.num_var1], [-1, -1])

        energy = tf.matmul(samples1, self.b1) + tf.matmul(samples2, self.b2) + tf.reduce_sum(
            tf.matmul(samples1, self.w) * samples2, 1, keepdims=True)
        energy = - tf.squeeze(energy, axis=1)
        return energy

    def log_prob(self, samples, stop_grad, is_training=True):
        """
        Computes the log of normalized probability of samples: log(exp(-E(z))/Z) = -E(z) - log(Z)
        Args:
            samples:  matrix with size (num_samples * num_vars)
            stop_grad: not used
            is_training: a flag indicating whether the graph is the training graph

        Returns: 
            log prob. for each sample.
        """
        log_z = self.log_z_train if is_training else self.log_z_value
        return - self.energy_tf(samples) - log_z

    def log_marginal_unnormalized(self, samples):
        """
        Computes the marginal log probability without considering log partition function.
        
        Args:
             samples: a tensor containing samples.
        """
        samples1 = tf.slice(samples, [0, 0], [-1, self.num_var1])

        bias_2 = tf.transpose(self.b2) + tf.matmul(samples1, self.w)
        log_prob = tf.matmul(samples1, self.b1) + tf.reduce_sum(tf.nn.softplus(bias_2), axis=1)
        return log_prob

    def cross_entropy_from_hierarchical(self, logit_q, log_ratio):
        """ 
        This computes the cross entropy term for the overlapping distributions presented in DVAE++.
        
        Args:
            logit_q:  the logit of the bernoulli distribution defined for each var. 
            log_ratio: \log \frac{r(\zeta|z=1)}{r(\zeta|z=0) for each \zeta

        Returns:
            cross_entropy: cross_entropy tensor for each \zeta 
        """
        logit_q1 = tf.slice(logit_q, [0, 0], [-1, self.num_var1])
        logit_q2 = tf.slice(logit_q, [0, self.num_var1], [-1, -1])
        log_ratio1 = tf.slice(log_ratio, [0, 0], [-1, self.num_var1])

        q1 = tf.nn.sigmoid(logit_q1)                                 # \mu_1 in DVAE++ paper
        q2 = tf.nn.sigmoid(logit_q2)                                 # \mu_2 in DVAE++ paper
        q1_pert = tf.nn.sigmoid(logit_q1 + log_ratio1)               # \nu_1 in DVAE++ paper
        cross_entropy = - tf.matmul(q1, self.b1) - tf.matmul(q2, self.b2) + \
                        - tf.reduce_sum(tf.matmul(q1_pert, self.w) * q2, 1, keep_dims=True)
        cross_entropy = tf.squeeze(cross_entropy, axis=1)
        cross_entropy = cross_entropy + self.log_z_train
        return cross_entropy

    def kl_dist_from_func(self, hierarchical, post_samples, is_training=True):
        """ Implements the KL divergence as a function. It can be used for concrete/pwl relaxation.
        
        Args:
            hierarchical: list of approx. post. distributions.
            post_samples: approx. post. samples (reparameterized)
            is_training: a flag indicating whether we are building the training computation graph

        """
        assert isinstance(hierarchical, list), 'the hierarchical distribution is a list of dist util objects.'
        for f in hierarchical:
            assert isinstance(f, FactorialBernoulliUtil), 'dist util should be a subclass of FactorialBernoulliUtil.'
        assert self.kld_term == 'kld_as_function'
        # assert len(hierarchical) == 1
        num_groups = len(hierarchical)
        assert num_groups % 2 == 0 or num_groups == 1

        entropy = 0
        all_q = []
        for factorial in hierarchical:
            entropy += tf.reduce_sum(factorial.entropy(), axis=1)
            all_q.append(tf.nn.sigmoid(factorial.logit_mu))

        # for single layer cross entropy is closed form
        if num_groups == 1:
            q = tf.nn.sigmoid(hierarchical[0].logit_mu)
            cross_entropy = self.energy_tf(q)
        else:
            # Rao Blackwellization of the second side.
            half_groups = num_groups // 2
            post = tf.concat(values=post_samples[:half_groups] + all_q[half_groups:], axis=1)
            cross_entropy = self.energy_tf(post)

        if is_training:
            kl = cross_entropy - entropy + self.log_z_train
        else:
            kl = cross_entropy - entropy + self.log_z_value

        return kl

    def kl_dist_from(self, hierarchical, post_samples=None, is_training=True):
        """
        Compute the KLD from a hierarchical approx. post. to the prior (RBM).
        The function implements the KLD presented in DVAE++.
        Args:
            hierarchical: list of approx. post. distributions.
            post_samples: approx. post. samples (reparameterized)
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            kl: a tensor containing the KL distance for each element in the batch
        """
        assert isinstance(hierarchical, list), 'the hierarchical distribution is a list of dist util objects.'
        for f in hierarchical:
            assert isinstance(f, FactorialBernoulliUtil), 'dist util should be a subclass of FactorialBernoulliUtil.'

        num_latent_layers = len(hierarchical)
        # when num_latent_layers == 1, the KL term has a closed form solution which is implemented in this function
        if self.kld_term is not None and self.kld_term == "kld_as_function":
            return self.kl_dist_from_func(hierarchical, post_samples, is_training)

        entropy = 0
        logit_q = []
        log_ratio = []
        for factorial, samples in zip(hierarchical, post_samples):
            entropy += tf.reduce_sum(factorial.entropy(), 1)
            logit_q.append(factorial.logit_mu)
            log_ratio.append(factorial.log_ratio(samples))

        logit_q = tf.concat(logit_q, axis=1)
        log_ratio = tf.concat(log_ratio, axis=1)
        samples = tf.concat(post_samples, axis=1)

        # the mean-field solution (num_latent_layers == 1) reduces to log_ratio = 0.
        if num_latent_layers == 1:
            log_ratio *= 0.

        if is_training:
            cross_entropy = self.cross_entropy_from_hierarchical(logit_q, log_ratio)
        else:
            cross_entropy = - self.log_prob(samples, stop_grad=False, is_training=is_training)

        kl = cross_entropy - entropy

        return kl

    def entropy(self):
        raise NotImplementedError

    def get_weight_decay(self):
        """
        Compute weight_decay * ||self.w||^2. Putting L2 norm on the pairwise weights makes the RBM problem a bit easier
        to sample from sometimes.
        Returns:
            l2 norm loss.
        """
        return self.weight_decay * tf.nn.l2_loss(self.w)

    def estimate_log_z(self, sess):
        """
        Estimate log_z using AIS implemented in QuPA
        Args:
            sess: tensorflow session
            
        Returns: 
            log_z: the estimate of log partition function 
        """
        import time
        s = time.time()
        log_z = sess.run(self.log_z_update)
        total_time = time.time() - s
        Print('Estimated log partition function with QuPA: %0.4f in %0.2f sec' % (log_z, total_time))

        return log_z


class MarginalRBMType1Generic(RBM):
    def __init__(self, num_var1, num_var2, weight_decay, name='RBM', num_samples=100, num_gibbs_iter=40,
                 use_qupa=False, smoothing_dist=SmoothingDist()):
        """
        This class implements the KL divergence and log prob for the Gaussian Integral trick presented in DVAE#.

        Args:
            num_var1:               number of vars of left side of RBM
            num_var2:               number of vars of right side of RBM
            weight_decay:           regularization for the weight matrix
            name:                   name
            num_samples:            number of RBM samples drawn in each iterations (used for computing log Z gradient)
            num_gibbs_iter:         number of gibbs step for pcd or mcmc sweeps for QuPA
            use_qupa:               A boolean flag indicating whether QuPA will be used for sampling. Setting this
                                    variable to False will use PCD for sampling
            smoothing_dist          the instance of the smoothing distribution class. This will be used for computing
                                    additional terms in the augmented Boltzmann distribution
        """
        RBM.__init__(self, num_var1=num_var1, num_var2=num_var2, weight_decay=weight_decay, name=name,
                     num_samples=num_samples, num_gibbs_iter=num_gibbs_iter, use_qupa=use_qupa,
                     kld_term='marginal_type1')

        self.smoothing_dist = smoothing_dist
        self.num_mf_iter = 5
        self.use_mf = True

    def energy_tf_augmented(self, samples_z, samples_zeta):
        """
        Computes the augmented energy function introduced in DVAE#.
        Args:
            samples_z: matrix of samples (z) with size (num_samples * num_vars)
            samples_zeta: matrix of samples (\zeta) with size (num_samples * num_vars)

        Returns: 
            energy for each sample.

        """
        log_r_zeta_0 = self.smoothing_dist.log_pdf(samples_zeta)
        log_r_zeta_1 = self.smoothing_dist.log_pdf(1. - samples_zeta)
        # b_zeta is log r(\zeta|z=1) - log r(\zeta|z=0)
        b_zeta = log_r_zeta_1 - log_r_zeta_0
        c_zeta = log_r_zeta_0
        return self.energy_tf(samples_z) - tf.reduce_sum(b_zeta * samples_z + c_zeta, axis=1)

    def log_prob(self, samples, stop_grad=False, is_training=False):
        """
        Computes log probability of samples under the marginal distribution if is_training is true, otherwise
        it computes the log probability of samples under the original RBM.
        
        The marginal distribution is introduced in DVAE#, sec 3.1.
        Args:
            samples: approx. post. samples (re-parameterized)
            stop_grad: this argument does not do anything in this function.
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            log_prob: a tensor containing log prob value for each element in samples
        """
        log_Z = self.log_z_train
        if is_training:
            if self.use_mf:
                mf_z, log_w = self.mf_iters(samples)
                energy_samples = self.energy_tf_augmented(mf_z, samples)
                log_prob = log_w - log_Z  # evaluate log p for continuous
                log_prob = tf.stop_gradient(log_prob)  # stop gradient
                grad_log_prob = - energy_samples - log_Z  # create tensor for gradients
                log_prob_cont_grad = grad_log_prob - tf.stop_gradient(grad_log_prob)
            else:
                raise NotImplementedError("We only have implemented MF method.")
        else:  # evaluate log p for assuming samples are discrete
            log_prob = - self.energy_tf(samples) - self.log_z_value
            log_prob = tf.stop_gradient(log_prob)         # apply stop gradient
            log_prob_cont_grad = 0.                       # no gradient

        return log_prob + log_prob_cont_grad

    def mf_iters(self, samples_zeta):
        """
        This function defines the mean-field optimization iterations. It also computes the log partition
        function for the augmented Boltzmann machine using Eq. 6 in DVAE#.
        Args:
            samples_zeta: approx. post. samples (re-parameterized)

        Returns:
            mf:  mean-field vector for each samples in the input tensor
            log_Z: log partition function estimation for the augmented Boltzmann machine.
        """

        log_r_zeta_0 = self.smoothing_dist.log_pdf(samples_zeta)
        log_r_zeta_1 = self.smoothing_dist.log_pdf(1. - samples_zeta)
        # b_zeta is log r(\zeta|z=1) - log r(\zeta|z=0)
        b_zeta = log_r_zeta_1 - log_r_zeta_0

        tf.summary.histogram('b_zeta', b_zeta)
        tf.summary.histogram('b_zeta_clipped', tf.clip_by_value(b_zeta, -2, 2))
        tf.summary.scalar('b_zeta_num', tf.reduce_mean(tf.reduce_sum(tf.to_float(tf.abs(b_zeta) < 1.), axis=1), axis=0))

        # define mean-field iterations
        mf = 0.5 * tf.ones_like(samples_zeta)
        with tf.name_scope('%s_gibbs_iterations' % self.name):
            mf1 = mf[:, :self.num_var1]
            mf2 = mf[:, self.num_var1:]
            for iter in range(self.num_mf_iter):
                # adjust unary terms using h(\zeta)
                b1 = tf.transpose(self.b1) + b_zeta[:, :self.num_var1]
                b2 = tf.transpose(self.b2) + b_zeta[:, self.num_var1:]
                w = self.w
                scope_name = 'gibbs_iter_%d' % iter
                with tf.name_scope(scope_name):
                    mf2 = tf.nn.sigmoid(tf.matmul(mf1, w) + b2)
                    mf2 = tf.stop_gradient(mf2)
                    mf1 = tf.nn.sigmoid(tf.matmul(mf2, w, transpose_b=True) + b1)
                    mf1 = tf.stop_gradient(mf1)

            mf = tf.concat(values=[mf1, mf2], axis=1)

        # estimate log partition function for the augmented Boltzmann machine.
        with tf.name_scope('%s_log_p_est' % self.name):
            mf_clipped = tf.clip_by_value(mf, 1e-7, 1. - 1e-7)
            mf_entropy = tf.reduce_sum(- mf * tf.log(mf_clipped) - (1.-mf) * tf.log(1 - mf_clipped), axis=1)
            neg_energy_mf = - self.energy_tf_augmented(mf, samples_zeta)
            log_Z = neg_energy_mf + mf_entropy
            log_Z = tf.stop_gradient(log_Z)

            return mf, log_Z

    def kl_dist_from(self, hierarchical, post_samples=None, is_training=True):
        """
        Compute the KLD from approx. post. to the marginal distribution presented in Sec 3.1 if
        is_training is True. Otherwise, it computes KL to the discrete RBM.

        Args:
            hierarchical: list of approx. post. distributions.
            post_samples: approx. post. samples (re-parameterized)
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            kld: KL divergence for each element in the batch
        """
        assert isinstance(hierarchical, list), 'the hierarchical distribution is a list of dist util objects.'
        for f in hierarchical:
            assert isinstance(f, FactorialBernoulliUtil), \
                'each dist util object should be a FactorialBernoulliUtil or its subclasses.'

        entropy = 0
        for i, factorial in enumerate(hierarchical):
            entropy -= tf.reduce_sum(factorial.log_prob_per_var(post_samples[i], stop_grad=True), axis=-1)

        post_samples = tf.concat(post_samples, axis=1)
        cross_entropy = self.cross_entropy_from_hierarchical(post_samples, is_training)
        return cross_entropy - entropy

    def cross_entropy_from_hierarchical(self, post_samples, is_training=False):
        """
        Computes a sampling-based estimate of the cross-entropy from a hierarchical posterior to marginal.
        Args:
            post_samples: approx. post. samples (re-parameterized)
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            cross_entropy: cross_entropy each element in the batch 
        """
        neg_log_prob = - self.log_prob(post_samples, stop_grad=False, is_training=is_training)
        return neg_log_prob
