# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf

from dist_util import FactorialBernoulliUtil, MixtureGeneric, RelaxDist
from rbm_posterior import RBMParam, ConcreteRBM
from nets import FeedForwardNetwork


class SimpleDecoder:
    def __init__(self, num_output, num_latent_units, config_recon, output_dist_util=FactorialBernoulliUtil):
        """ This function creates hierarchical decoder using a series of fully connected neural networks.   
        
        Args:
            num_output: number of output in the final output tensor. This can be equal to the length of x (the observed
              random variable).
            num_latent_units: number of latent units used in the prior.
            config_recon: a dictionary containing the hyper-parameters of the reconstruct network. See below for the keys required in the dictionary.
            output_dist_util: optional class indicating the distribution type of the output of the network.
              Only used to determine how outputs of the network should be "split". Default is FactorialBernoulliUtil,
              which has one parameter and so requires no splitting.
        """
        self.num_latent_units = num_latent_units
        self.num_output = num_output
        self.output_dist_util = output_dist_util

        # The final likelihood function p(x|z). The following makes the network that generate the output
        # used for the likelihood function.
        num_input = self.num_latent_units
        num_output = self.num_output * self.output_dist_util.num_param
        num_det_hiddens = [config_recon['num_det_units']] * config_recon['num_det_layers']
        weight_decay_recon = config_recon['weight_decay_dec']
        name = config_recon['name']
        use_batch_norm = config_recon['batch_norm']
        self.net = FeedForwardNetwork(
            num_input=num_input, num_hiddens=num_det_hiddens, num_output=num_output, name='%s_output' % name,
            weight_decay_coeff=weight_decay_recon, output_split=self.output_dist_util.num_param,
            use_batch_norm=use_batch_norm, collections='p_collections')

    def generator(self, prior_samples):
        """ This function generates samples using ancestral sampling from decoder. It accepts
        the samples from prior. This function can be used when samples from the model are being generated.
        
        Args:
            prior_samples:  A tensor containing samples from p(z).

        Returns:
            The output of likelihood function measured using the generated samples. 
        """
        return self.reconstruct(prior_samples, is_training=False, batch_norm_update=False)

    def reconstruct(self, post_samples, is_training, batch_norm_update):
        """ Given all the samples from the approximate posterior this function creates a network for
         p(x|z). It's output can be fed into a dist util object to create a distribution.
        
        Args:
            post_samples: A tensor containing samples for q(z | x) or p(z).
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.
            batchnorm_update: a boolean flag indicating whether the batch_norm moving averages should be updated.

        Returns:
            output_dist: a FactorialBernoulliUtil object containing the logit probability of output.
        """
        if isinstance(post_samples, list):
            post_samples = tf.concat(post_samples, axis=1)

        hiddens = self.net.build_network(post_samples, is_training, batch_norm_update)
        return hiddens

    def get_weight_decay(self):
        """ Returns the weight decay loss for the decoder networks.
        
        Returns:
            wd_loss: a scalar tensor containing weight decay loss.
        """
        return self.net.get_weight_decay_loss()


class SimpleEncoder:
    def __init__(self, num_input, config, dist_util, dist_util_param={}):
        """ This function creates hierarchical encoder using a series of fully connected neural networks.   

        Args:
            num_input: number of input that will be fed to the networks. This can be equal to the length of x (the 
             observed random variable).
            config: a dictionary containing the hyper-parameters of the encoder. See below for the keys required in the dictionary.
            dist_util: is a class used for creating parameters of the posterior.
            dist_util_param: parameters that will be passed to the dist util when creating the prior objects.
        """
        self.num_input = num_input
        # number of latent layers (levels in the hierarchy)
        self.num_latent_layers = 0 if config is None else config['num_latent_layers']
        # the following keys are extracted to form the encoder.
        if self.num_latent_layers > 0:
            self.num_latent_units = config['num_latent_units']    # number of latent units per layer.
            self.num_det_units = config['num_det_units_enc']      # number of dererministic units in each layer.
            self.num_det_layers = config['num_det_layers_enc']    # number of deterministic layers in each conditional p(z_i | z_{k<i})
            self.weight_decay = config['weight_decay_enc']        # weight decay coefficient.
            self.name = config['name']                            # name used for variable scopes.
            self.use_batch_norm = config['batch_norm']
        self.nets = []
        self.dist_util = dist_util
        self.dist_util_param = dist_util_param
        self.collections = 'q_collections'

        # Define all the networks required for the autoregressive posterior.
        for i in range(self.num_latent_layers):
            num_det_hiddens = [self.num_det_units] * self.num_det_layers
            num_input = self.num_input + i * self.num_latent_units
            num_output = self.num_latent_units * self.dist_util.num_param
            network = FeedForwardNetwork(
                num_input=num_input, num_hiddens=num_det_hiddens, num_output=num_output, name='%s_enc_%d' % (self.name, i),
                weight_decay_coeff=self.weight_decay, output_split=self.dist_util.num_param,
                use_batch_norm=self.use_batch_norm, collections=self.collections)
            self.nets.append(network)

    def hierarchical_posterior(self, input, is_training):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to num_latent_layers and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent units in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        posterior = []
        post_samples_zeta = []
        post_samples_z = []

        for i in range(self.num_latent_layers):
            network_input = tf.concat(axis=-1, values=[input] + post_samples_zeta)  # concat x, z0, z1, ...
            network = self.nets[i]
            param = network.build_network(network_input, is_training, batchnorm_update=is_training) # create network
            # In the evaluation, we will use Bernoulli instead of continuous relaxations.
            if not is_training and self.dist_util in {MixtureGeneric, RelaxDist}:
                posterior_dist = FactorialBernoulliUtil([param[0]])
            else:
                dist_util_param = self.dist_util_param
                posterior_dist = self.dist_util(param, dist_util_param)                    # init posterior dist.

            posterior.append(posterior_dist)
            samples, _ = posterior_dist.reparameterize(is_training)                      # reparameterize
            post_samples_zeta.append(samples)

        return posterior, post_samples_zeta

    def get_weight_decay(self):
        """ Returns the weight decay loss for all the encoder networks.

        Returns:
            wd_loss: a scalar tensor containing weight decay loss.
        """
        wd_loss = 0
        for net in self.nets:
            wd_loss += net.get_weight_decay_loss()

        return wd_loss


class RBMEncoder:
    def __init__(self, num_input, config, posterior_rbm):
        """ This function creates RBM encoder using a fully connected neural network.

        Args:
            num_input: number of input that will be fed to the networks. This can be equal to the length of x (the 
             observed random variable).
            config: a dictionary containing the hyper-parameters of the encoder. See below for the keys required in the dictionary.
            posterior_rbm: Posterior Class
        """
        self.num_input = num_input
        # number of latent layers (levels in the hierarchy)
        self.num_latent_layers = 0 if config is None else config['num_latent_layers']
        # the following keys are extracted to form the encoder.
        if self.num_latent_layers > 0:
            assert config['num_latent_layers'] == 1
            self.num_latent_units = config['num_latent_units']    # number of latent units per layer.
            self.num_det_units = config['num_det_units_enc']      # number of dererministic units in each layer.
            self.num_det_layers = config['num_det_layers_enc']    # number of deterministic layers in each conditional p(z_i | z_{k<i})
            self.weight_decay = config['weight_decay_enc']        # weight decay coefficient.
            self.name = config['name']                            # name used for variable scopes.
            self.use_batch_norm = config['batch_norm']
        self.posterior_rbm = posterior_rbm
        self.rank = config['rank']
        self.collections = 'q_collections'

        num_det_hiddens = [self.num_det_units] * self.num_det_layers
        num_input = self.num_input
        if isinstance(posterior_rbm, ConcreteRBM):
            self.param_class = RBMParam
            if self.rank == 0:
                num_output = posterior_rbm.num_var1 + posterior_rbm.num_var2 + posterior_rbm.num_var1 * posterior_rbm.num_var2
        else:
            raise NotImplementedError

        network = FeedForwardNetwork(
            num_input=num_input, num_hiddens=num_det_hiddens, num_output=num_output, name='%s_enc' % self.name,
            weight_decay_coeff=self.weight_decay, output_split=1, use_batch_norm=self.use_batch_norm,
            collections=self.collections)
        self.net = network

    def posterior(self, input, index, is_training):
        """ This function defines RBM approximate posterior distribution.

        Args:
            input: a tensor containing input tensor.
            index: a tensor containing the index of training data.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior_param: RBM posterior parameters.
            post_samples: A list of relaxed samples and rao-blackwelllized samples.
            post_samples_log_z: A list of log_z values.
        """
        param = self.net.build_network(input, is_training, batchnorm_update=is_training)           # create network
        posterior_param = self.param_class.get_rbm_param_nn(param, self.posterior_rbm, self.rank, is_training, self.collections)
        if is_training:
            if self.param_class == RBMParam:
                tf.summary.histogram('b1', posterior_param.b1)
                tf.summary.histogram('b2', posterior_param.b2)
                tf.summary.histogram('w', posterior_param.w)

            post_hard_samples = self.posterior_rbm.general_gibbs(posterior_param, index)
            post_samples, post_samples_mf = self.posterior_rbm.forward_gibbs_relax(posterior_param, post_hard_samples)
            post_samples = (post_samples, post_samples_mf)
            post_samples_log_z = 0.
        else:
            post_samples, post_samples_log_z = self.posterior_rbm.ais_samples_log_z(posterior_param)
            post_samples = (post_samples, post_samples)
            posterior_param.w = tf.tile(posterior_param.w, [self.posterior_rbm.num_ais_samples, 1, 1])

        return posterior_param, post_samples, post_samples_log_z

    def get_weight_decay(self):
        """ Returns the weight decay loss for all the encoder networks.

        Returns:
            wd_loss: a scalar tensor containing weight decay loss.
        """
        wd_loss = self.net.get_weight_decay_loss()
        return wd_loss

