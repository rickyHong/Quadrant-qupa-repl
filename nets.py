# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import re
import tensorflow.contrib.slim as slim


def add_vars_to_collection(scope_name, collection):
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for p in params:
        if re.search(scope_name, p.name):
            tf.add_to_collection(collection, p)


class FeedForwardNetwork:
    def __init__(self, num_input, num_hiddens, num_output, name, weight_decay_coeff, output_split=1,
                 use_batch_norm=False, collections=''):
        """ Initializes a simple fully connected network.

        Args:
            num_input: Number of columns (features) in the input tensor.
            num_hiddens: a list of integers containing the number of hidden units in each layer.
            num_output: an integer representing the number of output features.
            name: name used for the variable scope.
            weight_decay_coeff: the coefficient of weight decay loss.
            use_batch_norm: a boolean flag indicating whether to use batch norm or not.
            output_split: number of splits that should be applied to the output tensor in the feature axis. This can be 
              used to for example create tensors for both mu and log_sigma for Normal dist together.
            collections: A string containing the collections that variables will be added to.
        """
        assert isinstance(num_hiddens, list), 'number of hiddens should be a list of integer values.'
        assert isinstance(num_output, int) or num_output is None, 'num_output should be either an integer or None.'

        self.num_hidden_layers = len(num_hiddens)
        self.num_hidden_units = [num_input]
        self.num_hidden_units.extend(num_hiddens)
        self.num_hidden_units.append(num_output)
        self.output_split = output_split
        self.use_batch_norm = use_batch_norm
        self.name = name
        self.weight_decay_loss = 0
        self.weight_decay_coeff = weight_decay_coeff
        self.collections = collections

    def build_network(self, data, is_training, batchnorm_update):
        """ This function creates a fully connected network characterized by the current object.

        Args:
            data: is 2D tensor in the NC (batch, feature) format.
            is_training: a boolean flag indicating whether the network is part of train or test graph. 
            batchnorm_update: a boolean flag indicating whether the batch_norm moving averages should be updated.

        Returns:
            The output of the network.
        """
        assert data.get_shape().ndims == 2, 'the rank of input tensor should be 2.'
        with tf.variable_scope(self.name, reuse=not is_training) as scope:
            hidden = data
            for i in range(self.num_hidden_layers+1):
                output_size = self.num_hidden_units[i + 1]
                if output_size is None:
                    break
                with tf.variable_scope('layer_%d' % i):
                    hidden = tf.layers.dense(hidden, output_size,
                                             kernel_initializer=slim.variance_scaling_initializer())
                    # don't apply batch normalization and non-linearity to the last layer.
                    if i < self.num_hidden_layers:
                        if self.use_batch_norm:
                            hidden = tf.layers.batch_normalization(hidden, momentum=0.99, epsilon=1e-2,
                                                                   training=batchnorm_update, name='bn')
                        hidden = tf.nn.tanh(hidden)

                var_scope_name = self.name + '/' + 'layer_%d' % i
                if is_training:
                    add_vars_to_collection(var_scope_name, self.collections)

            output_splits = tf.split(hidden, self.output_split, axis=1)
            return output_splits

    def get_weight_decay_loss(self):
        """ Compute weight decay loss.

        Returns:
            Weight decay loss computed for the parameters of this network.
        """
        # find all variables with name 'kernel' and 'gamma' in the scope for this object.
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regex1 = self.name + '\/.*\/kernel'
        regex2 = self.name + '\/.*\/gamma'
        l2_norm_loss = 0
        for p in params:
            if re.search(regex1, p.name) or re.search(regex2, p.name):
                l2_norm_loss += tf.nn.l2_loss(p)

        return self.weight_decay_coeff * l2_norm_loss

