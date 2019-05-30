# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import numpy as np
import math

from smoothing_util import SmoothingDist


def sigmoid_cross_entropy_with_logits(logits, labels):
    """"
    See tensorflow.nn.sigmoid_cross_entropy_with_logits documentation.
    """
    return logits - logits * labels + tf.nn.softplus(-logits)


class DistUtil:
    def reparameterize(self, is_training):
        raise NotImplementedError

    def kl_dist_from(self, dist_util_obj, aux):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_prob(self, samples, stop_gradient):
        raise NotImplementedError


class FactorialBernoulliUtil(DistUtil):
    num_param = 1

    def __init__(self, param, kw_param={}):
        """
        Set the logits of the factorial Bernoulli distribution.
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
            kw_param: not required.
        """
        assert isinstance(param, list), 'param should be a list.'
        assert len(param) == 1, 'param should have a length of one corresponding to logit_mu.'
        self.logit_mu = param[0]

    def reparameterize(self, is_training):
        """ 
        Samples from the bernoulli distribution. Can be used only during test.
        Args:
            is_training: a flag indicating whether we are building the training computation graph
            
        Returns:
            z: samples from bernoulli distribution
        """
        if is_training:
            raise NotImplementedError('Reparameterization of a bernoulli distribution is not differentiable.')
        else:
            q = tf.nn.sigmoid(self.logit_mu)
            rho = tf.random_uniform(shape=tf.shape(q), dtype=tf.float32)
            ind_one = tf.less(rho, q)
            z = tf.where(ind_one, tf.ones_like(q), tf.zeros_like(q))
            return z, None

    def log_ratio(self, zeta):
        """
        A dummy function for this class.
        Args:
            zeta: approximate post samples

        Returns:
            log_ratio: 0.
        """
        log_ratio = 0. * (2 * zeta - 1)
        return log_ratio

    def entropy(self):
        """
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        mu = tf.nn.sigmoid(self.logit_mu)
        ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=mu)
        return ent

    def log_prob_per_var(self, samples, stop_grad):
        """
        Compute the log probability of samples under distribution of this object.
            - (x - x * z + log(1 + exp(-x))),  where x is logits, and z is samples.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        logit_mu = tf.stop_gradient(self.logit_mu) if stop_grad else self.logit_mu
        log_prob = - sigmoid_cross_entropy_with_logits(logits=logit_mu, labels=samples)
        return log_prob

    def log_prob(self, samples, stop_grad, is_training=False):
        """
        Call log_prob_per_var() and then compute the sum of log_prob of all the variables.
        Args:
            samples: matrix of size (num_samples * num_vars)
            stop_grad: a flag to stop gradient with respect to the parameters of the distribution
            is_training: does nothing.

        Returns: 
            log_p: A vector of log_prob for each sample.
        """
        log_p = self.log_prob_per_var(samples, stop_grad)
        log_p = tf.reduce_sum(log_p, 1)
        return log_p

    def kl_dist_from(self, dist_util_obj, aux):
        assert isinstance(dist_util_obj, FactorialBernoulliUtil)
        q = tf.nn.sigmoid(self.logit_mu)
        ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=q)
        cross_ent = sigmoid_cross_entropy_with_logits(logits=dist_util_obj.logit_mu, labels=q)
        return tf.reduce_sum(cross_ent - ent, axis=1)


class MixtureGeneric(FactorialBernoulliUtil):
    num_param = 1

    def __init__(self, param, kw_param={}):
        """
        Creates a mixture of two overlapping distributions by setting the logits of the factorial Bernoulli distribution
        defined on the z components. This is a generic class that can work with any type of smoothing distribution
        that extends the SmoothingDist class.
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
            kw_param: a dictionary containing the key 'smoothing_dist' which is an object representing the overlapping
            distribution.
        """

        assert isinstance(param, list), 'param should be a list.'
        assert len(param) == MixtureGeneric.num_param, 'param should have a length of %d.' % MixtureGeneric.num_param
        assert isinstance(kw_param['smoothing_dist'], SmoothingDist)
        FactorialBernoulliUtil.__init__(self, [param[0]])
        self.smoothing_dist = kw_param['smoothing_dist']

    def reparameterize(self, is_training):
        """"
        This function uses ancestral sampling to sample from mixture of two overlapping distributions. 
        It then uses the implicit gradient idea to compute the gradient of samples with respect to logit_q. 
        This idea is presented in DVAE# sec 3.4. This function does not implement the gradient of samples with respect
        to beta or other parameters of the smoothing transformation.

        Args:
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            zeta: samples from mixture of overlapping distributions.
        """
        q = tf.nn.sigmoid(self.logit_mu)

        z, _ = FactorialBernoulliUtil.reparameterize(self, is_training=False)
        shape = tf.shape(z)
        zeta = self.smoothing_dist.sample(shape)
        zeta = tf.where(tf.equal(z, 0.), zeta, 1. - zeta)

        pdf_0 = self.smoothing_dist.pdf(zeta, stop_grad=False)
        pdf_1 = self.smoothing_dist.pdf(1. - zeta, stop_grad=False)
        cdf_0 = self.smoothing_dist.cdf(zeta)
        cdf_1 = 1. - self.smoothing_dist.cdf(1. - zeta)

        grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
        grad_q = tf.stop_gradient(grad_q)
        grad_term = grad_q * q
        grad_term -= tf.stop_gradient(grad_term)

        zeta = tf.stop_gradient(zeta) + grad_term

        return zeta, z

    def log_prob_per_var(self, samples, stop_grad):
        """
        Compute the log probability of samples under mixture of overlapping distributions.
        Args:
            samples: matrix of size (num_samples * num_vars)
            stop_grad: a flag to stop gradient with respect to the parameters of the distribution

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        q = tf.nn.sigmoid(self.logit_mu)
        q = tf.stop_gradient(q) if stop_grad else q
        pdf_0 = self.smoothing_dist.pdf(samples, stop_grad)
        pdf_1 = self.smoothing_dist.pdf(1. - samples, stop_grad)
        log_prob = tf.log(q * pdf_1 + (1 - q) * pdf_0)
        return log_prob

    def log_prob(self, samples, stop_grad):
        """
        Call log_prob_per_var() and then compute the sum of log_prob of all the variables.
        Args:
            samples: matrix of size (num_samples * num_vars)
            stop_grad: a flag to stop gradient with respect to the parameters of the distribution

        Returns: 
            log_p: A vector of log_prob for each sample.
        """
        log_p = self.log_prob_per_var(samples, stop_grad)
        log_p = tf.reduce_sum(log_p, 1)
        return log_p

    def log_ratio(self, zeta):
        """
        Compute log_ratio needed for gradients of KL (presented in DVAE++).
        Args:
            zeta: approximate post samples

        Returns:
            log_ratio: log r(\zeta|z=1) - log r(\zeta|z=0) 
        """
        log_pdf_0 = self.smoothing_dist.log_pdf(zeta)
        log_pdf_1 = self.smoothing_dist.log_pdf(1. - zeta)
        log_ratio = log_pdf_1 - log_pdf_0
        return log_ratio


# This class implements the continuous relaxation proposed in Andriyash et al. ICLR 2019
class RelaxDist(FactorialBernoulliUtil):
    num_param = 1

    def __init__(self, param, kw_param={}):
        """
        Creates a mixture of two overlapping distributions by setting the logits of the factorial Bernoulli distribution
        defined on the z components. This is a generic class that can work with any type of smoothing distribution
        that extends the SmoothingDist class.
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
            kw_param: a dictionary containing the key 'smoothing_dist' which is an object representing the overlapping
            distribution.
        """

        assert isinstance(param, list), 'param should be a list.'
        assert len(param) == RelaxDist.num_param, 'param should have a length of %d.' % MixtureGeneric.num_param
        FactorialBernoulliUtil.__init__(self, [param[0]])
        self.smoothing_fun = kw_param['smoothing_fun']
        self.beta = kw_param['beta']

    def reparameterize(self, is_training):
        zeta, z = self.smoothing_fun(self.logit_mu, beta=self.beta)
        return zeta, z


def sample_concrete(logit_q, beta):
    """ Implements concrete relaxation
    Args:
        logit_q: the logit of Bernoulli distribution
        beta: inverse temperature used for smoothing
    """
    rho = tf.clip_by_value(tf.random_uniform(tf.shape(logit_q)), 1e-5, 1. - 1e-5)
    samples = tf.nn.sigmoid(beta * (logit_q + tf.log(rho / (1 - rho))))
    z = tf.stop_gradient(tf.round(samples))
    return samples, z


def sample_through_pwlinear(logit_q, beta):
    """ Implements the piece-wise linear relaxation proposed in https://arxiv.org/abs/1810.00116
    Args:
        logit_q: the logit of Bernoulli distribution
        beta: inverse temperature used for smoothing
    """

    q = tf.nn.sigmoid(logit_q)
    u = tf.random_uniform(tf.shape(q))
    # the original PWL applies this stop_gradient to reduce bias.
    # slope = tf.stop_gradient(0.25 * beta / (q * (1. - q) + 1e-2))
    slope = 0.25 * beta / (q * (1. - q) + 1e-2)
    zeta = tf.minimum(1., tf.maximum(0., 0.5 + slope * (u - (1. - q))))
    z = tf.stop_gradient(tf.round(zeta))
    return zeta, z
