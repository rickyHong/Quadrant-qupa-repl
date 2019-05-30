# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf


class SmoothingDist:
    def pdf(self, zeta, stop_grad):
        """ Implements r(\zeta|z=0)"""
        raise NotImplementedError

    def cdf(self, zeta):
        """ Implements R(\zeta|z=0)"""
        raise NotImplementedError

    def sample(self, shape):
        """ Samples from r(\zeta|z=0)"""
        raise NotImplementedError

    def log_pdf(self, zeta):
        """ Computes log r(\zeta|z=0)"""
        raise NotImplementedError


class PowerLaw(SmoothingDist):
    """ This class implements the smoothing distribution class for power function."""
    def __init__(self, params):
        self._lambda = 1. / params['beta']

    def pdf(self, zeta, stop_grad):
        _lambda = tf.stop_gradient(self._lambda) if stop_grad else self._lambda
        pdf = tf.pow(zeta + 1e-7, _lambda - 1.) * _lambda
        return pdf

    def cdf(self, zeta):
        cdf = tf.pow(zeta + 1e-7, self._lambda)
        return cdf

    def sample(self, shape):
        rho = tf.random_uniform(shape)
        zeta = tf.pow(rho, 1. / self._lambda)
        return zeta

    def log_pdf(self, zeta):
        log_pdf = (self._lambda - 1.) * tf.log(zeta + 1e-7) + tf.log(self._lambda)
        return log_pdf