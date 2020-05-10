"""Compute image metrics: IS, FID."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow_gan as tfgan


dim_latent = 1024
num_latents = 100

def get_Sparseprior(batch_size):
    z = np.zeros([batch_size, dim_latent], dtype=np.float32)
    for i in range(batch_size):
        z_idx = np.random.choice(dim_latent, [num_latents])
        z[i, z_idx] = np.random.normal(0, 1, size=[num_latents])
    return tf.identity(z)

# generate points in latent space as input for the generator
def get_prior(batch_size):
    # generate points in the latent space
    x_input = np.random.randn(num_latents * batch_size)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(batch_size, num_latents)
    return tf.convert_to_tensor(x_input,dtype='float32')

def get_image_metrics_for_samples_Sparse(
    real_images, generator, num_eval_samples):
  """Compute inception score and FID."""
  max_classifier_batch = 10
  num_batches = num_eval_samples //  max_classifier_batch

  def sample_fn(arg):
    del arg
    samples = generator(get_Sparseprior(max_classifier_batch))
    # Ensure data is in [0, 1], as expected by TFGAN.
    # Resizing to appropriate size is done by TFGAN.
    return (samples + 1) / 2

  fake_outputs = tfgan.eval.sample_and_run_inception(
      sample_fn,
      sample_inputs=[1.0] * num_batches)  # Dummy inputs.

  fake_logits = fake_outputs['logits']
  inception_score = tfgan.eval.classifier_score_from_logits(fake_logits)

  real_outputs = tfgan.eval.run_inception(real_images, num_batches=num_batches)
  fid = tfgan.eval.frechet_classifier_distance_from_activations(
      real_outputs['pool_3'], fake_outputs['pool_3'])

  return {
      'inception_score': inception_score,
      'fid': fid}

def get_image_metrics_for_samples(
    real_images, generator, num_eval_samples):
  """Compute inception score and FID."""
  max_classifier_batch = 10
  num_batches = num_eval_samples //  max_classifier_batch

  def sample_fn(arg):
    del arg
    samples = generator(get_prior(max_classifier_batch))
    # Ensure data is in [0, 1], as expected by TFGAN.
    # Resizing to appropriate size is done by TFGAN.
    return (samples + 1) / 2

  fake_outputs = tfgan.eval.sample_and_run_inception(
      sample_fn,
      sample_inputs=[1.0] * num_batches)  # Dummy inputs.

  fake_logits = fake_outputs['logits']
  inception_score = tfgan.eval.classifier_score_from_logits(fake_logits)

  real_outputs = tfgan.eval.run_inception(real_images, num_batches=num_batches)
  fid = tfgan.eval.frechet_classifier_distance_from_activations(
      real_outputs['pool_3'], fake_outputs['pool_3'])

  return {
      'inception_score': inception_score,
      'fid': fid}