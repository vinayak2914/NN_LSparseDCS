import tensorflow as tf
import numpy as np
from numpy.random import seed
import nets_keras as nets
import tensorflow_probability as tfp
import random
import collections
import os
#import sonnet as snt
import file_utils
import math
import scipy.io


tfd = tfp.distributions

output_dir = 'DeepcsTF2'
dataset = 'mnist'
batch_size = 64
num_measurements = 25
num_z_iters = 3
z_project_method = 'norm'
epochs = 200
num_latents = 100
dim_latent = 1000
export_every = 100
z_step_size = tf.Variable(tf.exp(math.log(0.01)), dtype=tf.float32)
rand_seed = 14
optimizer = tf.keras.optimizers.Adam(1e-4)
tf.random.set_seed(rand_seed)
seed(rand_seed)


def get_rep_loss(img1, img2, measure_net):
	batch_size = tf.shape(img1)[0].numpy()
	m1 = measure_net(img1)
	m2 = measure_net(img2)
	
	img_diff_norm = tf.norm(img1 - img2, axis=-1)
	m_diff_norm = tf.norm(m1 - m2, axis=-1)
	return tf.square(img_diff_norm - m_diff_norm)


def get_measurement_error(target_meas, sample_img, measure_net):
	# m_targets,_ = measure_net(tf.reshape(target_img, [tf.shape(sample_img)[0].numpy(), 784]))
	# m_samples,_ = measure_net(tf.reshape(sample_img, [tf.shape(sample_img)[0].numpy(), 784]))
	# sum_sqr = tf.reduce_sum(tf.square(m_samples - m_targets), -1)
	return tf.reduce_sum(tf.square(measure_net(sample_img) - target_meas), -1)


def gen_loss_fn(data, samples, measure_net):
	return get_measurement_error(data, samples, measure_net)


def get_optimisation_cost(initial_z, optimised_z):
	optimisation_cost = tf.reduce_mean(tf.reduce_sum((optimised_z - initial_z) ** 2, -1))
	return optimisation_cost


def project_z(z, project_method='clip'):
	if project_method == 'norm':
		z_p = tf.nn.l2_normalize(z, axis=-1)
	elif project_method == 'clip':
		z_p = tf.clip_by_value(z, -1, 1)
	else:
		raise ValueError('Unknown project_method: {}'.format(project_method))
	return z_p


def prox_hardthresh(z):
	z_np = z.numpy()
	[z_abs_50, z_idx_50] = tf.math.top_k(tf.abs(z), k=num_latents, sorted=True, name=None)
	z_idx_np = z_idx_50.numpy()
	z_thres = np.zeros([batch_size, dim_latent])
	# b = -np.sort(-z_np)
	# b = tf.sort(z, axis= 1, direction='DESCENDING', name=None)
	for i in range(batch_size):
		# min = b[i, num_latents]
		# z_np[i, z_np[i,:] < min] = 0
		z_thres[i, z_idx_np[i, :]] = z_np[i, z_idx_np[i, :]]
	z = tf.identity(z_thres)
	return z


def optimise_and_sample_Sparse(init_z, data, generator_net, measure_net):
	if num_z_iters == 0:
		z_final = init_z
	else:
		init_loop_vars = (0, project_z(init_z, z_project_method))
		loop_cond = lambda i, _: i < num_z_iters
		
		def loop_body(i, z):
			with tf.GradientTape() as tape:
				loop_samples = generator_net(z)
				tape.watch(z)
				gen_loss = gen_loss_fn(data, loop_samples, measure_net)
			# print(tf.reduce_mean(gen_loss))
			z_grad = tape.gradient(gen_loss, z)
			z = z - z_step_size * z_grad
			indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
			indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
			z = tf.math.multiply(indices_float, z)
			z = project_z(z, z_project_method)
			return i + 1, z
		
		_, z_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
		# return module.generator_net(z_final, is_training), z_final
		return z_final


def optimise_and_sample(init_z, data, generator_net, measure_net):
	if num_z_iters == 0:
		z_final = init_z
	else:
		init_loop_vars = (0, project_z(init_z, z_project_method))
		loop_cond = lambda i, _: i < num_z_iters
		
		def loop_body(i, z):
			with tf.GradientTape() as tape:
				loop_samples = generator_net(z)
				tape.watch(z)
				gen_loss = gen_loss_fn(data, loop_samples, measure_net)
			# print(tf.reduce_mean(gen_loss))
			z_grad = tape.gradient(gen_loss, z)
			z = z - z_step_size * z_grad
			z = project_z(z, z_project_method)
			return i + 1, z
		
		_, z_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
		# return module.generator_net(z_final, is_training), z_final
		return z_final


def make_output_dir(output_dir):
	# Returns whether the path is directory or not
	if not tf.io.gfile.isdir(output_dir):
		tf.io.gfile.makedirs(output_dir)
	return


def preprocess(x):
	return x * 2 - 1


def postprocess(x):
	return (x + 1) / 2


def get_np_data(dataset, split):
	"""Get dataset as numpy array"""
	index = 0 if split == 'train' else 1
	# index = 0 indicates only training images to be loaded without labels
	if dataset == 'mnist':
		x, _ = tf.keras.datasets.mnist.load_data()[index]
		x = x / 255
		x = x.astype(np.float32)
		# x = x.reshape(60000, 784)
		x = x.reshape((-1, 28, 28, 1))
		x = preprocess(x)
	return x


def get_train_dataset(dataset, batch_size):
	x_train = get_np_data(dataset, split='train')
	# x_train = x_train[0:6400, :, :, :]
	x_test = get_np_data(dataset, split='test')
	x_valid = x_test[0:int(x_test.shape[0]/2), :, :, :]
	dataset = tf.data.Dataset.from_tensor_slices(x_train)
	dataset_valid = tf.data.Dataset.from_tensor_slices(x_valid)
	dataset_test = tf.data.Dataset.from_tensor_slices(x_test)
	# dataset = dataset.shuffle(100000).repeat().batch(batch_size)
	dataset = dataset.shuffle(100000).batch(batch_size)
	dataset_valid = dataset_valid.shuffle(100000).batch(batch_size)
	dataset_test = dataset_test.shuffle(100000).batch(batch_size)
	return dataset,dataset_valid,dataset_test


def get_Sparseprior(batch_size):
	# prior_mean = tf.zeros(shape=(num_latents), dtype=tf.float32)
	# prior_scale = tf.ones(shape=(num_latents), dtype=tf.float32)
	# dist = tfd.Binomial(total_count=5., probs=.5)
	z = np.zeros([batch_size, dim_latent], dtype=np.float32)
	for i in range(batch_size):
		z_idx = np.random.choice(dim_latent, [num_latents])
		z[i, z_idx] = np.random.normal(0, 1, size=[num_latents])
	return tf.identity(z)


def get_prior(num_latents):
	prior_mean = tf.zeros(shape=(num_latents), dtype=tf.float32)
	prior_scale = tf.ones(shape=(num_latents), dtype=tf.float32)
	return tfd.Normal(loc=prior_mean, scale=prior_scale)

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# return tfd.Normal(loc=prior_mean, scale=prior_scale)


def get_flatten_list(optimization_var_list):
	flat_list = []
	for sublist in optimization_var_list:
		if isinstance(sublist, list):
			for item in sublist:
				flat_list.append(item)
		else:
			flat_list.append(sublist)
	
	return flat_list



prior = get_prior(num_latents)
# Initialize Generator  and Measurement Parameters

# generator_net = snt.nets.MLP([500, 500, 784], activation=tf.nn.leaky_relu)
# measure_net = snt.nets.MLP([500, 500, num_measurements], activation=tf.nn.leaky_relu)
# generatorSparse_net = snt.nets.MLP([500, 500, 784], activation=tf.nn.leaky_relu)
# measureSparse_net = snt.nets.MLP([500, 500, num_measurements], activation=tf.nn.leaky_relu)
make_output_dir(output_dir)
generator_net = nets.MLPGenNet()
measure_net = nets.MLPMesNet(num_measurements)
generatorSparse_net = nets.MLPGenNetSparse()
measureSparse_net = nets.MLPMesNetSparse(num_measurements)

sparse_reconstloss = np.zeros(epochs)
sonnet_reconstloss = np.zeros(epochs)
sparse_valid_reconstloss = np.zeros(epochs)
sonnet_valid_reconstloss = np.zeros(epochs)


for epoch in range(epochs):
	sparse_reconstloss_itr = []
	sonnet_reconstloss_itr = []
	trn_images,valid_images,_ = get_train_dataset(dataset, batch_size)
	print('epoch %s: started' % (epoch))
	step = 0
	for x_batch_train in trn_images:
		step = step +1
		if step == 900:
			break
		print('Iteration %s: started' % (step))
		# x_batch_train = next(iter(trn_images))
		generator_inputs = prior.sample(batch_size)
		generatorSparse_inputs = get_Sparseprior(batch_size)
		# generator_inputs = get_prior(batch_size)
		x_batch_copy = tf.identity(x_batch_train)
		# Sparse Deep Compressive Sensing
		with tf.GradientTape() as tapeSparse:
			z_i_Sparse = tf.identity(generatorSparse_inputs)
			x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])
			meas_img_Sparse = measureSparse_net(x_img_reshape)
			optimised_z_Sparse = optimise_and_sample_Sparse(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
			                                                measureSparse_net)
			optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
			initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
			generator_loss_Sparse = tf.reduce_mean(gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
			recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
			r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
			r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
			r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
			meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
			total_loss_Sparse = (generator_loss_Sparse + meas_loss_Sparse)
			gen_var_Sparse = generatorSparse_net.trainable_variables
			meas_var_Sparse = measureSparse_net.trainable_variables
			train_var_Sparse = gen_var_Sparse + meas_var_Sparse
			# optimization_vars = get_flatten_list([train_var, z_step_size])
			optimization_vars_Sparse = train_var_Sparse
			tapeSparse.watch(optimization_vars_Sparse)
		gradients_Sparse = tapeSparse.gradient(total_loss_Sparse, optimization_vars_Sparse)
		optimizer.apply_gradients(zip(gradients_Sparse, optimization_vars_Sparse))
		# generator_inputs = tf.identity(optimised_z)
		print('total_loss_Sparse %s' % (total_loss_Sparse))
		print('generator_loss_Sparse %s' % (generator_loss_Sparse))
		print('recont_loss_Sparse %s\n' % (recont_loss_Sparse))
		reconstructions_Sparse = generatorSparse_net(optimised_z_Sparse)
		sparse_reconstloss_itr.append(recont_loss_Sparse)
		
		
		if step % export_every == 0:
			# Create an object which gets data and does the processing.
			data_np = postprocess(x_batch_train)
			reconstructions_np_Sparse = postprocess(reconstructions_Sparse)
			sample_exporter = file_utils.FileExporter(
				os.path.join(output_dir, 'reconstructions_sparse'))
			reconstructions_np_Sparse = tf.reshape(reconstructions_np_Sparse, data_np.shape)
			sample_exporter.save(reconstructions_np_Sparse, 'reconstructions_sparse')
			sample_exporter.save(data_np, 'data')
		
		# Deep Compressive Sensing
		with tf.GradientTape() as tape:
			z_i = tf.identity(generator_inputs)
			x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])
			meas_img = measure_net(x_img_reshape)
			optimised_z = optimise_and_sample(z_i, meas_img, generator_net, measure_net)
			optimized_sample = generator_net(optimised_z)
			initial_sample = generator_net(z_i)
			generator_loss = tf.reduce_mean(gen_loss_fn(meas_img, optimized_sample, measure_net))
			recont_loss = tf.reduce_mean(tf.norm(optimized_sample - x_img_reshape, axis=-1))
			r1 = get_rep_loss(optimized_sample, initial_sample, measure_net)
			r2 = get_rep_loss(optimized_sample, x_img_reshape, measure_net)
			r3 = get_rep_loss(initial_sample, x_img_reshape, measure_net)
			meas_loss = tf.reduce_mean((r1 + r2 + r3) / 3.0)
			total_loss = (generator_loss + meas_loss)
			gen_var = generator_net.trainable_variables
			meas_var = measure_net.trainable_variables
			train_var = gen_var + meas_var
			# optimization_vars = get_flatten_list([train_var, z_step_size])
			optimization_vars = train_var
			tape.watch(optimization_vars)
		gradients = tape.gradient(total_loss, optimization_vars)
		optimizer.apply_gradients(zip(gradients, optimization_vars))
		# generator_inputs = tf.identity(optimised_z)
		print('total_loss %s' % (total_loss))
		print('generator_loss %s' % (generator_loss))
		print('recont_loss %s\n' % (recont_loss))
		reconstructions = generator_net(optimised_z)
		sonnet_reconstloss_itr.append(recont_loss)
		
		
		if step % export_every == 0:
			# Create an object which gets data and does the processing.
			data_np = postprocess(x_batch_train)
			reconstructions_np = postprocess(reconstructions)
			sample_exporter = file_utils.FileExporter(
				os.path.join(output_dir, 'reconstructions'))
			reconstructions_np = tf.reshape(reconstructions_np, data_np.shape)
			sample_exporter.save(reconstructions_np, 'reconstructions')
			sample_exporter.save(data_np, 'data')
	
	sparse_reconstloss[epoch] = np.mean(np.array(sparse_reconstloss_itr))
	sonnet_reconstloss[epoch] = np.mean(np.array(sonnet_reconstloss_itr))
	# Validation
	step = 0
	sparse_reconstloss_valid_itr = []
	sonnet_reconstloss_valid_itr = []
	for x_batch_valid in valid_images:
		step = step +1
		if step == 10:
			break
		generatorSparse_inputs_v = get_Sparseprior(batch_size)
		z_i_Sparse_v = tf.identity(generatorSparse_inputs_v)
		x_img_reshape_v = tf.reshape(x_batch_valid, [-1, tf.shape(x_batch_valid)[1] * tf.shape(x_batch_valid)[2]])
		meas_img_Sparse_v = measureSparse_net(x_img_reshape_v)
		optimised_z_Sparse_v = optimise_and_sample_Sparse(z_i_Sparse_v, meas_img_Sparse_v, generatorSparse_net,
		                                                measureSparse_net)
		reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v)
		sparse_reconstloss_valid = tf.reduce_mean(tf.norm(reconstructions_Sparse_v - x_img_reshape_v, axis=-1))
		sparse_reconstloss_valid_itr.append(sparse_reconstloss_valid)
		print('Valid_recont_loss_Sparse %s\n' % (sparse_reconstloss_valid))
		
		generator_inputs_v = prior.sample(batch_size)
		z_i_v = tf.identity(generator_inputs_v)
		meas_img_v = measure_net(x_img_reshape_v)
		optimised_z_v = optimise_and_sample(z_i_v, meas_img_v, generator_net, measure_net)
		reconstructions_v = generator_net(optimised_z_v)
		sonnet_reconstloss_valid = tf.reduce_mean(tf.norm(reconstructions_v - x_img_reshape_v, axis=-1))
		sonnet_reconstloss_valid_itr.append(sonnet_reconstloss_valid)
		print('Valid_recont_loss %s\n' % (sonnet_reconstloss_valid))
		
	sonnet_valid_reconstloss[epoch] = np.mean(np.array(sonnet_reconstloss_valid_itr))
	sparse_valid_reconstloss[epoch] = np.mean(np.array(sparse_reconstloss_valid_itr))

# Testing
sparse_test_reconstloss = []
sonnet_test_reconstloss = []
_,_,test_images = get_train_dataset(dataset, batch_size)
step = 0
for x_batch_test in test_images:
	step = step + 1
	if step == 10:
		break
	generatorSparse_inputs_tst = get_Sparseprior(batch_size)
	z_i_Sparse_tst = tf.identity(generatorSparse_inputs_tst)
	x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
	meas_img_Sparse_tst = measureSparse_net(x_img_reshape_tst)
	optimised_z_Sparse_tst = optimise_and_sample_Sparse(z_i_Sparse_tst, meas_img_Sparse_tst, generatorSparse_net,
	                                                  measureSparse_net)
	reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst)
	sparse_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_Sparse_tst - x_img_reshape_tst, axis=-1))
	sparse_test_reconstloss.append(sparse_test_reconstloss_itr)
	print('Test_recont_loss_Sparse %s\n' % (sparse_test_reconstloss_itr))
	
	data_np = postprocess(x_batch_test)
	reconstructions_Sparse_tst_np = postprocess(reconstructions_Sparse_tst)
	sample_exporter = file_utils.FileExporter(
		os.path.join(output_dir, 'reconstructions_Sparse_test'))
	reconstructions_Sparse_tst_np = tf.reshape(reconstructions_Sparse_tst_np, data_np.shape)
	sample_exporter.save(reconstructions_Sparse_tst_np, 'reconstructions_Sparse_test')
	sample_exporter.save(data_np, 'data')
	
	
	
	generator_inputs_tst = prior.sample(batch_size)
	z_i_test = tf.identity(generator_inputs_tst)
	meas_img_tst = measure_net(x_img_reshape_tst)
	optimised_z_tst = optimise_and_sample(z_i, meas_img_tst, generator_net, measure_net)
	reconstructions_tst = generator_net(optimised_z_tst)
	sonnet_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_tst - x_img_reshape_tst, axis=-1))
	sonnet_test_reconstloss.append(sonnet_test_reconstloss_itr)
	print('Test_recont_loss %s\n' % (sonnet_test_reconstloss_itr))
	
	data_np = postprocess(x_batch_test)
	reconstructions_tst_np = postprocess(reconstructions_tst)
	sample_exporter = file_utils.FileExporter(
		os.path.join(output_dir, 'reconstructions_test'))
	reconstructions_tst_np = tf.reshape(reconstructions_tst_np, data_np.shape)
	sample_exporter.save(reconstructions_tst_np, 'reconstructions_test')
	sample_exporter.save(data_np, 'data')
	
reconstructions_Sparse_tst_np_reshape = tf.reshape(reconstructions_Sparse_tst_np,[64,784])
data_np_reshape = tf.reshape(data_np,[64,784])
reconstructions_tst_np_reshape = tf.reshape(reconstructions_tst_np,[64,784])
scipy.io.savemat('Sparse_TF2_DCS.mat',mdict={'sonnet_test_reconstloss': sonnet_test_reconstloss,'sparse_test_reconstloss': sparse_test_reconstloss,'sparse_reconstloss':sparse_reconstloss,'sonnet_reconstloss':sonnet_reconstloss,'sonnet_valid_reconstloss':sonnet_valid_reconstloss,'sparse_valid_reconstloss':sparse_valid_reconstloss, 'data_np_reshape':data_np_reshape.numpy(),'reconstructions_tst_np_reshape':reconstructions_tst_np_reshape.numpy(),'reconstructions_Sparse_tst_np_reshape':reconstructions_Sparse_tst_np_reshape.numpy()})
aa  = 1
