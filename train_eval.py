# Copyright 2018 D-Wave Systems Inc.
# DVAE## licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from thirdparty import input_data

from util import fill_feed_dict, create_reconstruction_image, tile_image_tf, tile_image_np, Print, get_global_step_var


def check_num_modes(mean_fields):
    # using unique to get the number of modes
    d = np.asarray(mean_fields * 100, dtype=np.uint8)
    unique, counts = np.unique(d, return_counts=True, axis=0)
    # sort
    ind = np.argsort(counts)[::-1]
    counts = counts[ind]

    ind = counts > 5
    print 'num_modes:', len(counts[ind])


def evaluate(sess, neg_elbo, log_iw, images_placeholder, k_iw_pl, k, data_set, batch_size, mean_fields=None):
    """ This function evaluates negative elbo and negative log likelihood using tensors created in the vae class.
    
    Args:
        sess: tf sess.
        neg_elbo: a tensor containing neg_elbo for each element in the image placeholder.
        log_iw: a tensor containing log p(x, z) / q(z|x) for each element in the image placeholder.
        images_placeholder: a place holder that is used to feed test batches for computing neg_elbo and log_iw.
        k_iw_pl: a place holder for the number of importance samples.
        k: the number of importance samples.
        data_set: a dataset object that can be used to generate test batches.
        batch_size: batch size used for evaluation.
        mean_fields: the mean_field tensor extracted for the undirected posteriors.

    Returns:
        neg_elbo_value: the value of neg elbo computed on the whole test dataset.
        neg_log_likelihood: the value of neg log likelihood computed on the whole test dataset.
    """
    # And run one epoch of eval.
    steps_per_epoch = data_set.num_examples // batch_size    # number of mini-batches in the dataset.
    assert data_set.num_examples % batch_size == 0, 'batch size should divide the dataset size.'
    assert batch_size == 1, 'batch size should be one'

    log_p_all = []
    neg_elbo_all = []
    for step in xrange(steps_per_epoch):
        index_placeholder = None
        feed_dict = fill_feed_dict(data_set, images_placeholder, index_placeholder, batch_size, k_iw_pl, k)

        if mean_fields is not None and step < 5:
            neg_elbo_np, log_p_np, mean_fields_np = sess.run([neg_elbo, log_iw, mean_fields], feed_dict=feed_dict)
            check_num_modes(mean_fields_np)
        else:
            neg_elbo_np, log_p_np = sess.run([neg_elbo, log_iw], feed_dict=feed_dict)

        log_p_all.append(log_p_np)
        neg_elbo_all.append(neg_elbo_np)

    neg_log_likelihood = - np.mean(log_p_all)
    neg_elbo_value = np.mean(neg_elbo_all)

    return neg_elbo_value, neg_log_likelihood


def run_training(vae, cont_train, config_train, log_dir):
    """ The main function that will derive training of a vae.
    Args:
        vae: is an object from the class VAE. 
        cont_train: a boolean flag indicating whether train should continue from the checkpoint stored in the log_dir.
        config_train: a dictionary containing config. training (hyperparameters).
        log_dir: path to a directory that will used for storing both tensorboard files and checkpoints.

    Returns:
        test_neg_ll_value: the value of test log-likelihood.
    """
    Print('Starting training.')
    batch_size = config_train['batch_size']
    # Get the train, val, test sets of on MNIST.
    data_dir = config_train['data_dir']
    eval_batch_size = config_train['eval_batch_size']
    data_sets = input_data.read_data_set(data_dir, dataset=config_train['dataset'], get_subset=config_train['subset_eval'])

    # place holder for input.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, vae.num_input))
    index_placeholder = tf.placeholder(tf.int32, shape=(None))
    k_iw_placeholder = tf.placeholder(tf.int32, shape=())

    # define training graph.
    neg_elbo, sigmoid_output, wd_loss, _ = \
        vae.neg_elbo(input_placeholder, input_index=index_placeholder, is_training=True, k=config_train['k'])

    train_op = vae.training(neg_elbo, wd_loss)
    if config_train['use_iw'] and config_train['k'] > 1:
        neg_elbo = neg_elbo[0]

    # create images for reconstruction.
    image = create_reconstruction_image(input_placeholder, sigmoid_output[:batch_size], batch_size)
    tf.summary.image('recon', image, max_outputs=1)

    # define graph to generate random samples from model.
    num_samples = 100
    random_samples = vae.generate_samples(num_samples)
    tiled_samples = tile_image_tf(random_samples, n=int(np.sqrt(num_samples)), m=int(np.sqrt(num_samples)), width=28, height=28)
    tf.summary.image('generated_sample', tiled_samples, max_outputs=1)

    # merge all summary for training graph
    train_summary_op = tf.summary.merge_all()

    # define a parallel graph for evaluation. Enable parameter sharing by setting is_training to False.
    neg_elbo_eval, output_eval, _, log_p_eval = vae.neg_elbo(input_placeholder, input_index=index_placeholder,
                                                             is_training=False, k=k_iw_placeholder)

    # the following will create summaries that will be used in the evaluation graph.
    val_neg_elbo, test_neg_elbo = tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())
    val_neg_ll, test_neg_ll = tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())
    val_summary = tf.summary.scalar('val/neg_elbo', val_neg_elbo)
    test_summary = tf.summary.scalar('test/neg_elbo', test_neg_elbo)
    val_ll_summary = tf.summary.scalar('val/neg_ll', val_neg_ll)
    test_ll_summary = tf.summary.scalar('test/neg_ll', test_neg_ll)
    eval_summary_op = tf.summary.merge([val_summary, test_summary, val_ll_summary, test_ll_summary])

    # start checkpoint saver.
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()

    # Run the Op to initialize the variables.
    if cont_train:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            init_step = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
            Print('Initializing model from %s from step %d' % (log_dir, init_step))
        else:
            raise('No Checkpoint was fount at %s' % log_dir)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        init_step = 0

    # Instantiate a SummaryWriter to output summaries and the Graph.
    # Create train/validation/test summary directories
    summary_writer = tf.summary.FileWriter(log_dir)

    # And then after everything is built, start the training loop.
    duration = 0.0
    best_val_neg_ll = np.finfo(float).max
    num_iter = config_train['num_iter']
    for step in xrange(init_step, num_iter):
        start_time = time.time()
        # perform one iteration of training.
        feed_dict = fill_feed_dict(data_sets.train, input_placeholder, index_placeholder, batch_size, k_iw_placeholder, 1)
        _, neg_elbo_value = sess.run([train_op, neg_elbo], feed_dict=feed_dict)
        duration += time.time() - start_time

        # Save a checkpoint and evaluate the model periodically.
        eval_iter = 20000 if num_iter > 1e5 else 10000
        if (step + 1) % eval_iter == 0 or (step + 1) == num_iter:
            # if vae has rbm in its prior we should update its log Z.
            if vae.should_compute_log_z():
                vae.prior.estimate_log_z(sess)

            mf_var = vae.posterior.mf_variable if vae.encoder_type == 'rbm' else None
            # validate on the validation and test set
            val_neg_elbo_value, val_neg_ll_value = evaluate(sess, neg_elbo_eval, log_p_eval, input_placeholder,
                                                            k_iw_placeholder, config_train['k_iw'], data_sets.validation,
                                                            batch_size=eval_batch_size, mean_fields=mf_var)

            # We don't evaluate on the test set during training, as it can be slow for undirected posteriors.
            test_neg_elbo_value, test_neg_ll_value = val_neg_elbo_value, val_neg_ll_value
            summary_str = sess.run(
                eval_summary_op, feed_dict={val_neg_elbo: val_neg_elbo_value, test_neg_elbo: test_neg_elbo_value,
                                            val_neg_ll: val_neg_ll_value, test_neg_ll: test_neg_ll_value})
            summary_writer.add_summary(summary_str, step)

            Print('Step %d: val ELBO = %.2f test ELBO = %.2f, val NLL = %.2f, test NLL = %.2f' %
                  (step, val_neg_elbo_value, test_neg_elbo_value, val_neg_ll_value, test_neg_ll_value))
            # save model if it is better on validation set:
            if val_neg_ll_value < best_val_neg_ll:
                best_val_neg_ll = val_neg_ll_value
                saver.save(sess, log_dir + '/', global_step=step)

        # Write the summaries and print an overview fairly often.
        report_iter = 1000
        if step % report_iter == 0 and step > 500:
            # print status to stdout.
            Print('Step %d, %.3f sec per step, train ELBO =%0.2f' % (step, duration/report_iter, neg_elbo_value))
            duration = 0.0
            # Update the events file.
            summary_str = sess.run(train_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # in the last iteration, we load the best model based on the validation performance, and evaluate it on test
        if (step + 1) == num_iter:
            Print('Final evaluation using the best saved model')
            # reload the best model this is good when a model overfits.
            ckpt = tf.train.get_checkpoint_state(log_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            Print('Done restoring the model at step: %d' % sess.run(get_global_step_var()))
            if vae.should_compute_log_z():
                vae.prior.estimate_log_z(sess)

            mf_var = vae.posterior.mf_variable if vae.encoder_type == 'rbm' else None
            # final main evaluation
            test_neg_elbo_value, test_neg_ll_value = evaluate(sess, neg_elbo_eval, log_p_eval, input_placeholder,
                                                              k_iw_placeholder, config_train['k_iw'], data_sets.test,
                                                              eval_batch_size, mf_var)
            summary_str = sess.run(
                eval_summary_op, feed_dict={val_neg_elbo: val_neg_elbo_value, test_neg_elbo: test_neg_elbo_value,
                                            val_neg_ll: val_neg_ll_value, test_neg_ll: test_neg_ll_value})
            Print('Step %d: val ELBO = %.2f test ELBO = %.2f, val NLL = %.2f, test NLL = %.2f' %
                  (step, val_neg_elbo_value, test_neg_elbo_value, val_neg_ll_value, test_neg_ll_value))
            summary_writer.add_summary(summary_str, step+1)
            summary_writer.flush()

            sess.close()
            tf.reset_default_graph()
            return test_neg_ll_value



