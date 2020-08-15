# Learned Non-linear measurement
import tensorflow as tf
import numpy as np
from numpy.random import seed
import nets_keras as nets
# import tensorflow_probability as tfp
import random
import collections
import os
# import sonnet as snt
import file_utils
import math
import scipy.io
from scipy.io import loadmat

# from skimage.metrics import structural_similarity as ssim
# import numpy as np
# import matplotlib.pyplot as plt


# tfd = tfp.distributions


dataset = 'mnist'
lambda_l1 = tf.Variable(tf.exp(math.log(0.1)), dtype=tf.float32)
lbd = 1
gamma_mcp = 7
lambda_mcp = 0.1
l_p = 0.5
batch_size = 64
num_measurements = 25
num_z_iters = 5
z_project_method = 'norm'
epochs = 200
num_latents = 100
dim_latent = 784

l0_dim = 100
sparse_dim = 50

export_every = 100
z_step_size = tf.Variable(tf.exp(math.log(0.01)), dtype=tf.float32)
rand_seed = 14
optimizer = tf.keras.optimizers.Adam(1e-4)
tf.random.set_seed(rand_seed)
seed(rand_seed)
output_dir = ('DeepcsTF2_mcpLpz%d_m%d_spr%d_%d' % (num_z_iters, num_measurements, sparse_dim, lbd))


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

def get_huber_loss(z,l,r):
    loss = np.zeros(z.shape,dtype = float)
    for i in range(z.shape[0]):
      # loss.append(h(z[i,:],0))
      x1 = l*tf.abs(z[i,:]) - .5*(tf.square(z[i,:])/r)
      # print (x1.shape)
      x2 = (l*l*r)/2
      # print (x2)
      loss[i,:] = tf.where(tf.less_equal(tf.abs(z[i,:]),l*r), loss[i,:], x1)
      # print (tf.less_equal(tf.abs(z[i,:]),l*r))
      loss[i,:] = tf.where(tf.greater(tf.abs(z[i,:]),l*r),loss[i,:],x2)
      # print (tf.greater(tf.abs(z[i,:]),l*r))
      # print (loss[i,:])
    return tf.cast(tf.reduce_sum(loss,axis = 1),tf.float32)


def optimization_loss_sparse(z_i_Sparse, x_img_reshape, generatorSparse_net, measureSparse_net):
    meas_img_Sparse = measureSparse_net(x_img_reshape)
    optimised_z_Sparse = optimise_and_sample_Sparse(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                    measureSparse_net)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


def optimization_loss_l0(z_i_Sparse, x_img_reshape, generatorSparse_net, measureSparse_net):
    meas_img_Sparse = measureSparse_net(x_img_reshape)
    optimised_z_Sparse = optimise_and_sample_l0(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                measureSparse_net)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


def optimization_loss_l1(z_i_Sparse, x_img_reshape, generatorSparse_net, measureSparse_net):
    meas_img_Sparse = measureSparse_net(x_img_reshape)
    optimised_z_Sparse = optimise_and_sample_l1(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                measureSparse_net)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


def optimization_loss_lp(z_i_Sparse, x_img_reshape, generatorSparse_net, measureSparse_net):
    meas_img_Sparse = measureSparse_net(x_img_reshape)
    optimised_z_Sparse = optimise_and_sample_lp(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                measureSparse_net)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


def optimization_loss_mcp(z_i_Sparse, x_img_reshape, generatorSparse_net, measureSparse_net):
    meas_img_Sparse = measureSparse_net(x_img_reshape)
    optimised_z_Sparse = optimise_and_sample_mcp(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                 measureSparse_net)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


def optimization_loss(z_i_Sparse, x_img_reshape, generatorSparse_net, measureSparse_net):
    meas_img_Sparse = measureSparse_net(x_img_reshape)
    optimised_z_Sparse = optimise_and_sample(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                             measureSparse_net)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(optimized_sample_Sparse - x_img_reshape, axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measureSparse_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, x_img_reshape, measureSparse_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, x_img_reshape, measureSparse_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


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
                l1_z = tf.reduce_mean(tf.abs(z))
                loop_samples = generator_net(z)
                tape.watch(z)
                gen_loss = gen_loss_fn(data, loop_samples, measure_net)
            # print(tf.reduce_mean(gen_loss))
            z_grad = tape.gradient(gen_loss, z)
            z = z - z_step_size * z_grad
            # indices_to_remove = tf.math.less(tf.abs(z), lambda_l1)
            indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
            indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
            z = tf.math.multiply(indices_float, z)
            z = project_z(z, z_project_method)
            return i + 1, z

        _, z_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
        # return module.generator_net(z_final, is_training), z_final
        return z_final


def optimise_and_sample_l0(init_z, data, generator_net, measure_net):
    if num_z_iters == 0:
        z_final = init_z
    else:
        init_loop_vars = (0, project_z(init_z, z_project_method))
        loop_cond = lambda i, _: i < num_z_iters

        def loop_body(i, z):
            with tf.GradientTape() as tape:
                l1_z = tf.reduce_mean(tf.abs(z))
                loop_samples = generator_net(z)
                tape.watch(z)
                gen_loss = gen_loss_fn(data, loop_samples, measure_net)
            # print(tf.reduce_mean(gen_loss))
            z_grad = tape.gradient(gen_loss, z)
            z = z - z_step_size * z_grad
            indices_to_remove = tf.math.less(tf.abs(z), lambda_l1)
            # indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
            indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
            z = tf.math.multiply(indices_float, z)
            z = project_z(z, z_project_method)
            return i + 1, z

        _, z_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
        # return module.generator_net(z_final, is_training), z_final
        return z_final


def optimise_and_sample_l1(init_z, data, generator_net, measure_net):
    if num_z_iters == 0:
        z_final = init_z
    else:
        init_loop_vars = (0, project_z(init_z, z_project_method))
        loop_cond = lambda i, _: i < num_z_iters

        def loop_body(i, z):
            with tf.GradientTape() as tape:
                l1_z = tf.reduce_mean(tf.abs(z),axis = 1)
                loop_samples = generator_net(z)
                tape.watch(z)
                gen_loss = gen_loss_fn(data, loop_samples, measure_net) + lambda_l1 * l1_z
            # print(tf.reduce_mean(gen_loss))
            z_grad = tape.gradient(gen_loss, z)
            z = z - z_step_size * z_grad
            # indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
            # indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
            # z = tf.math.multiply(indices_float, z)
            z = project_z(z, z_project_method)
            return i + 1, z

        _, z_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
        # return module.generator_net(z_final, is_training), z_final
        return z_final


def optimise_and_sample_lp(init_z, data, generator_net, measure_net):
    if num_z_iters == 0:
        z_final = init_z
    else:
        init_loop_vars = (0, project_z(init_z, z_project_method))
        loop_cond = lambda i, _: i < num_z_iters

        def loop_body(i, z):
            with tf.GradientTape() as tape:
                # l1_z = tf.reduce_mean(tf.abs(z))
                lp_z = tf.norm(z, ord=l_p, axis=1)
                loop_samples = generator_net(z)
                tape.watch(z)
                gen_loss = gen_loss_fn(data, loop_samples, measure_net) + lambda_l1 * lp_z
            # print(tf.reduce_mean(gen_loss))
            z_grad = tape.gradient(gen_loss, z)
            z = z - z_step_size * z_grad
            # indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
            # indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
            # z = tf.math.multiply(indices_float, z)
            z = project_z(z, z_project_method)
            return i + 1, z

        _, z_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
        # return module.generator_net(z_final, is_training), z_final
        return z_final


def optimise_and_sample_mcp(init_z, data, generator_net, measure_net):
    if num_z_iters == 0:
        z_final = init_z
    else:
        init_loop_vars = (0, project_z(init_z, z_project_method))
        loop_cond = lambda i, _: i < num_z_iters

        def loop_body(i, z):
            with tf.GradientTape() as tape:
                # l1_z = tf.reduce_mean(tf.abs(z))
                mcp_loss = get_huber_loss(z,lambda_mcp,gamma_mcp)
                # mcp_z = tf.abs(z) - g
                loop_samples = generator_net(z)
                tape.watch(z)
                gen_loss = gen_loss_fn(data, loop_samples, measure_net) + mcp_loss
                # gen_loss = gen_loss_fn(data, loop_samples, measure_net)
            # print(tf.reduce_mean(gen_loss))
            z_grad = tape.gradient(gen_loss, z)
            z = z - z_step_size * z_grad
            # indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
            # indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
            # z = tf.math.multiply(indices_float, z)
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


def get_train_dataset(x_train, batch_size):
    # x_train = get_np_data(dataset, split='train')
    # choose random instances
    ix = np.random.randint(0, x_train.shape[0], batch_size)
    # retrieve selected images
    X = tf.convert_to_tensor(x_train[ix], dtype='float32')

    return X


def get_test_dataset(step, x_test, batch_size):
    # retrieve selected images
    X = tf.convert_to_tensor(x_test[step * batch_size:(step * batch_size) + batch_size], dtype='float32')
    return X


def get_Sparseprior(batch_size):
    z = np.zeros([batch_size, l0_dim], dtype=np.float32)
    for i in range(batch_size):
        z_idx = np.random.choice(l0_dim, [sparse_dim])
        z[i, z_idx] = np.random.normal(0, 1, size=[sparse_dim])
    return tf.identity(z)


# generate points in latent space as input for the generator
def get_prior(latent_dim, batch_size):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * batch_size)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(batch_size, latent_dim)
    return tf.convert_to_tensor(x_input, dtype='float32')


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


train_dataset = get_np_data(dataset, split='train')
test_dataset = get_np_data(dataset, split='test')
valid_dataset = test_dataset[0:int(test_dataset.shape[0] / 2), :, :, :]
bat_per_epo = int(train_dataset.shape[0] / batch_size)
bat_valid_per_epo = int(valid_dataset.shape[0] / batch_size)
bat_test_per_epo = int(test_dataset.shape[0] / batch_size)
data_size = [train_dataset.shape[1], train_dataset.shape[2]]
data_dim = data_size[0] * data_size[1]
# dim_latent = data_dim

load_weights = 0
load_epoch = 39

if load_weights == 1:

    start_epoch = load_epoch + 1
    mat_file = os.getcwd() + '/' + output_dir + '/saved_var_%d_%d_%d.mat' % (num_measurements, num_latents, dim_latent)
    dict_mat = loadmat(mat_file)

    total_loss_Sparse_itr_mean = np.reshape(dict_mat.get('total_loss_Sparse_itr_mean'), -1)
    total_loss_l0_itr_mean = np.reshape(dict_mat.get('total_loss_l0_itr_mean'), -1)
    total_loss_l1_itr_mean = np.reshape(dict_mat.get('total_loss_l1_itr_mean'), -1)
    total_loss_lp_itr_mean = np.reshape(dict_mat.get('total_loss_lp_itr_mean'), -1)
    total_loss_mcp_itr_mean = np.reshape(dict_mat.get('total_loss_mcp_itr_mean'), -1)
    total_loss_itr_mean = np.reshape(dict_mat.get('total_loss_itr_mean'), -1)
    sparseDCS_reconstloss = np.reshape(dict_mat.get('sparseDCS_reconstloss'), -1)
    l0DCS_reconstloss = np.reshape(dict_mat.get('l0DCS_reconstloss'), -1)
    l1DCS_reconstloss = np.reshape(dict_mat.get('l1DCS_reconstloss'), -1)
    lpDCS_reconstloss = np.reshape(dict_mat.get('lpDCS_reconstloss'), -1)
    mcpDCS_reconstloss = np.reshape(dict_mat.get('mcpDCS_reconstloss'), -1)
    DCS_reconstloss = np.reshape(dict_mat.get('DCS_reconstloss'), -1)
    # DCS_valid_reconstloss = np.reshape(dict_mat.get('DCS_valid_reconstloss'),-1)
    # sparseDCS_valid_reconstloss = np.reshape(dict_mat.get('sparseDCS_valid_reconstloss'),-1)

    generator_net_file = os.getcwd() + '/' + output_dir + '/gen_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    measure_net_file = os.getcwd() + '/' + output_dir + '/meas_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    generatorSparse_net_file = os.getcwd() + '/' + output_dir + '/genSpar_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    measureSparse_net_file = os.getcwd() + '/' + output_dir + '/measSpar_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    generatorl0_net_file = os.getcwd() + '/' + output_dir + '/genl0_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    measurel0_net_file = os.getcwd() + '/' + output_dir + '/measl0_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    generatorl1_net_file = os.getcwd() + '/' + output_dir + '/genl1_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    measurel1_net_file = os.getcwd() + '/' + output_dir + '/measl1_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    generatorlp_net_file = os.getcwd() + '/' + output_dir + '/genlp_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    measurelp_net_file = os.getcwd() + '/' + output_dir + '/measlp_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    generatormcp_net_file = os.getcwd() + '/' + output_dir + '/genmcp_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)
    measuremcp_net_file = os.getcwd() + '/' + output_dir + '/measmcp_n_%d_%d_%d_%d' % (
    num_measurements, num_latents, sparse_dim, load_epoch)

    # load_sparse_input = get_Sparseprior(batch_size)
    load_sparse_input = get_prior(num_latents, batch_size)
    load_l0_input = get_prior(num_latents, batch_size)
    load_l1_input = get_prior(num_latents, batch_size)
    load_lp_input = get_prior(num_latents, batch_size)
    load_mcp_input = get_prior(num_latents, batch_size)
    load_input = get_prior(num_latents, batch_size)

    load_Sparse_z = tf.identity(load_sparse_input)
    load_l0_z = tf.identity(load_l0_input)
    load_l1_z = tf.identity(load_l1_input)
    load_lp_z = tf.identity(load_lp_input)
    load_mcp_z = tf.identity(load_mcp_input)
    load_z = tf.identity(load_input)

    generator_net = nets.MLPGenNet()
    measure_net = nets.MLPMesNet(num_measurements)
    generatorSparse_net = nets.MLPGenNetSparse()
    measureSparse_net = nets.MLPMesNetSparse(num_measurements)
    generatorl0_net = nets.MLPGenNetll()
    measurel0_net = nets.MLPMesNetll(num_measurements)
    generatorl1_net = nets.MLPGenNetl()
    measurel1_net = nets.MLPMesNetl(num_measurements)
    generatorlp_net = nets.MLPGenNetlp()
    measurelp_net = nets.MLPMesNetlp(num_measurements)
    generatormcp_net = nets.MLPGenNetmcp()
    measuremcp_net = nets.MLPMesNetmcp(num_measurements)

    generator_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=optimizer)
    generatorSparse_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                                optimizer=optimizer)
    measure_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=optimizer)
    measureSparse_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=optimizer)
    generatorl0_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=optimizer)
    measurel0_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=optimizer)
    generatorl1_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=optimizer)
    measurel1_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=optimizer)
    generatorlp_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=optimizer)
    measurelp_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=optimizer)
    generatormcp_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=optimizer)
    measuremcp_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=optimizer)

    load_sample_Sparse = generatorSparse_net(load_Sparse_z, is_training=True)
    load_meas_img_Sparse = measureSparse_net(load_sample_Sparse)

    load_sample_l0 = generatorl0_net(load_l0_z, is_training=True)
    load_meas_img_l0 = measurel0_net(load_sample_l0)

    load_sample_l0 = generatorl1_net(load_l0_z, is_training=True)
    load_meas_img_l0 = measurel1_net(load_sample_l0)

    load_sample_lp = generatorlp_net(load_lp_z, is_training=True)
    load_meas_img_lp = measurelp_net(load_sample_lp)

    load_sample_mcp = generatormcp_net(load_mcp_z, is_training=True)
    load_meas_img_mcp = measuremcp_net(load_sample_mcp)

    load_sample = generator_net(load_z, is_training=True)
    load_meas_img = measure_net(load_sample)

    generator_net.load_weights(generator_net_file)
    measure_net.load_weights(measure_net_file)
    generatorSparse_net.load_weights(generatorSparse_net_file)
    measureSparse_net.load_weights(measureSparse_net_file)
    generatorl0_net.load_weights(generatorl0_net_file)
    measurel0_net.load_weights(measurel0_net_file)
    generatorl1_net.load_weights(generatorl1_net_file)
    measurel1_net.load_weights(measurel1_net_file)
    generatorlp_net.load_weights(generatorlp_net_file)
    measurelp_net.load_weights(measurelp_net_file)
    generatormcp_net.load_weights(generatormcp_net_file)
    measuremcp_net.load_weights(measuremcp_net_file)

else:

    ######################################################################################
    start_epoch = 0
    make_output_dir(output_dir)
    generator_net = nets.MLPGenNet()
    measure_net = nets.MLPMesNet(num_measurements)
    generatorSparse_net = nets.MLPGenNetSparse()
    measureSparse_net = nets.MLPMesNetSparse(num_measurements)
    generatorl0_net = nets.MLPGenNetll()
    measurel0_net = nets.MLPMesNetll(num_measurements)
    generatorl1_net = nets.MLPGenNetl()
    measurel1_net = nets.MLPMesNetl(num_measurements)
    generatorlp_net = nets.MLPGenNetlp()
    measurelp_net = nets.MLPMesNetlp(num_measurements)
    generatormcp_net = nets.MLPGenNetmcp()
    measuremcp_net = nets.MLPMesNetmcp(num_measurements)

    sparseDCS_reconstloss = np.zeros(epochs)
    l1DCS_reconstloss = np.zeros(epochs)
    l0DCS_reconstloss = np.zeros(epochs)
    lpDCS_reconstloss = np.zeros(epochs)
    mcpDCS_reconstloss = np.zeros(epochs)
    DCS_reconstloss = np.zeros(epochs)
    # sparseDCS_valid_reconstloss = np.zeros(epochs)
    # DCS_valid_reconstloss = np.zeros(epochs)
    total_loss_Sparse_itr_mean = np.zeros(epochs)
    total_loss_l0_itr_mean = np.zeros(epochs)
    total_loss_l1_itr_mean = np.zeros(epochs)
    total_loss_lp_itr_mean = np.zeros(epochs)
    total_loss_mcp_itr_mean = np.zeros(epochs)
    total_loss_itr_mean = np.zeros(epochs)

######################################################################################


for epoch in range(start_epoch, epochs):
    sparseDCS_reconstloss_itr = []
    l0DCS_reconstloss_itr = []
    l1DCS_reconstloss_itr = []
    lpDCS_reconstloss_itr = []
    mcpDCS_reconstloss_itr = []
    DCS_reconstloss_itr = []
    total_loss_Sparse_itr = []
    total_loss_l0_itr = []
    total_loss_l1_itr = []
    total_loss_lp_itr = []
    total_loss_mcp_itr = []
    total_loss_itr = []

    print('epoch %s: started' % (epoch))
    # step = 0
    for step in range(bat_per_epo):
        x_batch_train = get_train_dataset(train_dataset, batch_size)
        print('> Epoch:%d  Iteration:%d:' % (epoch, step))
        # generatorSparse_inputs = get_Sparseprior(batch_size)
        generator_inputs_sparse = get_Sparseprior(batch_size)
        generator_inputs_l0 = get_prior(num_latents, batch_size)
        generator_inputs_l1 = get_prior(num_latents, batch_size)
        generator_inputs_lp = get_prior(num_latents, batch_size)
        generator_inputs_mcp = get_prior(num_latents, batch_size)
        generator_inputs = get_prior(num_latents, batch_size)
        x_batch_copy = tf.identity(x_batch_train)

        # mcp Deep Compressive Sensing
        with tf.GradientTape() as tapemcp:
            # z_i_Sparse = tf.identity(generatorSparse_inputs)
            z_i_mcp = tf.identity(generator_inputs_mcp)
            x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])

            total_loss_mcp, generator_loss_mcp, recont_loss_mcp, optimised_z_mcp = optimization_loss_mcp(z_i_mcp,
                                                                                                         x_img_reshape,
                                                                                                         generatormcp_net,
                                                                                                         measuremcp_net)
            gen_var_mcp = generatormcp_net.trainable_variables
            meas_var_mcp = measuremcp_net.trainable_variables
            train_var_mcp = gen_var_mcp + meas_var_mcp

            optimization_vars_mcp = train_var_mcp
            tapemcp.watch(optimization_vars_mcp)
        gradients_mcp = tapemcp.gradient(total_loss_mcp, optimization_vars_mcp)
        optimizer.apply_gradients(zip(gradients_mcp, optimization_vars_mcp))

        print('total_loss_mcp %s' % (total_loss_mcp))
        print('generator_loss_mcp %s' % (generator_loss_mcp))
        print('recont_loss_mcp %s\n' % (recont_loss_mcp))
        reconstructions_mcp = generatormcp_net(optimised_z_mcp)
        mcpDCS_reconstloss_itr.append(recont_loss_mcp)
        total_loss_mcp_itr.append(total_loss_mcp)

        if step % export_every == 0:
            rescont_mcp_file = 'reconstructions_mcp_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np_mcp = postprocess(reconstructions_mcp)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_mcp'))
            reconstructions_np_mcp = tf.reshape(reconstructions_np_mcp, data_np.shape)
            sample_exporter.save(reconstructions_np_mcp, rescont_mcp_file)
            sample_exporter.save(data_np, data_file)

        # lp Deep Compressive Sensing
        with tf.GradientTape() as tapelp:
            # z_i_Sparse = tf.identity(generatorSparse_inputs)
            z_i_lp = tf.identity(generator_inputs_lp)
            x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])

            total_loss_lp, generator_loss_lp, recont_loss_lp, optimised_z_lp = optimization_loss_lp(z_i_lp,
                                                                                                    x_img_reshape,
                                                                                                    generatorlp_net,
                                                                                                    measurelp_net)
            gen_var_lp = generatorlp_net.trainable_variables
            meas_var_lp = measurelp_net.trainable_variables
            train_var_lp = gen_var_lp + meas_var_lp

            optimization_vars_lp = train_var_lp
            tapelp.watch(optimization_vars_lp)
        gradients_lp = tapelp.gradient(total_loss_lp, optimization_vars_lp)
        optimizer.apply_gradients(zip(gradients_lp, optimization_vars_lp))

        print('total_loss_lp %s' % (total_loss_lp))
        print('generator_loss_lp %s' % (generator_loss_lp))
        print('recont_loss_lp %s\n' % (recont_loss_lp))
        reconstructions_lp = generatorlp_net(optimised_z_lp)
        lpDCS_reconstloss_itr.append(recont_loss_lp)
        total_loss_lp_itr.append(total_loss_lp)

        if step % export_every == 0:
            rescont_lp_file = 'reconstructions_lp_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np_lp = postprocess(reconstructions_lp)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_lp'))
            reconstructions_np_lp = tf.reshape(reconstructions_np_lp, data_np.shape)
            sample_exporter.save(reconstructions_np_lp, rescont_lp_file)
            sample_exporter.save(data_np, data_file)

        # l0 Deep Compressive Sensing
        with tf.GradientTape() as tapel0:
            # z_i_Sparse = tf.identity(generatorSparse_inputs)
            z_i_l0 = tf.identity(generator_inputs_l0)
            x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])

            total_loss_l0, generator_loss_l0, recont_loss_l0, optimised_z_l0 = optimization_loss_l0(z_i_l0,
                                                                                                    x_img_reshape,
                                                                                                    generatorl0_net,
                                                                                                    measurel0_net)
            gen_var_l0 = generatorl0_net.trainable_variables
            meas_var_l0 = measurel0_net.trainable_variables
            train_var_l0 = gen_var_l0 + meas_var_l0

            optimization_vars_l0 = train_var_l0
            tapel0.watch(optimization_vars_l0)
        gradients_l0 = tapel0.gradient(total_loss_l0, optimization_vars_l0)
        optimizer.apply_gradients(zip(gradients_l0, optimization_vars_l0))

        print('total_loss_l0 %s' % (total_loss_l0))
        print('generator_loss_l0 %s' % (generator_loss_l0))
        print('recont_loss_l0 %s\n' % (recont_loss_l0))
        reconstructions_l0 = generatorl0_net(optimised_z_l0)
        l0DCS_reconstloss_itr.append(recont_loss_l0)
        total_loss_l0_itr.append(total_loss_l0)

        if step % export_every == 0:
            rescont_l0_file = 'reconstructions_l0_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np_l0 = postprocess(reconstructions_l0)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_l0'))
            reconstructions_np_l0 = tf.reshape(reconstructions_np_l0, data_np.shape)
            sample_exporter.save(reconstructions_np_l0, rescont_l0_file)
            sample_exporter.save(data_np, data_file)

        # l1 Deep Compressive Sensing
        with tf.GradientTape() as tapel1:
            # z_i_Sparse = tf.identity(generatorSparse_inputs)
            z_i_l1 = tf.identity(generator_inputs_l1)
            x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])

            total_loss_l1, generator_loss_l1, recont_loss_l1, optimised_z_l1 = optimization_loss_l1(z_i_l1,
                                                                                                    x_img_reshape,
                                                                                                    generatorl1_net,
                                                                                                    measurel1_net)
            gen_var_l1 = generatorl1_net.trainable_variables
            meas_var_l1 = measurel1_net.trainable_variables
            train_var_l1 = gen_var_l1 + meas_var_l1

            optimization_vars_l1 = train_var_l1
            tapel1.watch(optimization_vars_l1)
        gradients_l1 = tapel1.gradient(total_loss_l1, optimization_vars_l1)
        optimizer.apply_gradients(zip(gradients_l1, optimization_vars_l1))

        print('total_loss_l1 %s' % (total_loss_l1))
        print('generator_loss_l1 %s' % (generator_loss_l1))
        print('recont_loss_l1 %s\n' % (recont_loss_l1))
        reconstructions_l1 = generatorl1_net(optimised_z_l1)
        l1DCS_reconstloss_itr.append(recont_loss_l1)
        total_loss_l1_itr.append(total_loss_l1)

        if step % export_every == 0:
            rescont_l1_file = 'reconstructions_l1_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np_l1 = postprocess(reconstructions_l1)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_l1'))
            reconstructions_np_l1 = tf.reshape(reconstructions_np_l1, data_np.shape)
            sample_exporter.save(reconstructions_np_l1, rescont_l1_file)
            sample_exporter.save(data_np, data_file)

        # sparse Deep Compressive Sensing
        with tf.GradientTape() as tapeSparse:
            # z_i_Sparse = tf.identity(generatorSparse_inputs)
            z_i_l0 = tf.identity(generator_inputs_sparse)
            x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])

            total_loss_Sparse, generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse = optimization_loss_sparse(
                z_i_l0, x_img_reshape, generatorSparse_net, measureSparse_net)
            gen_var_Sparse = generatorSparse_net.trainable_variables
            meas_var_Sparse = measureSparse_net.trainable_variables
            train_var_Sparse = gen_var_Sparse + meas_var_Sparse

            optimization_vars_Sparse = train_var_Sparse
            tapeSparse.watch(optimization_vars_Sparse)
        gradients_Sparse = tapeSparse.gradient(total_loss_Sparse, optimization_vars_Sparse)
        optimizer.apply_gradients(zip(gradients_Sparse, optimization_vars_Sparse))
        # generator_inputs = tf.identity(optimised_z)
        print('total_loss_Sparse %s' % (total_loss_Sparse))
        print('generator_loss_Sparse %s' % (generator_loss_Sparse))
        print('recont_loss_Sparse %s\n' % (recont_loss_Sparse))
        reconstructions_Sparse = generatorSparse_net(optimised_z_Sparse)
        # reconstructions_Sparse = generatorSparse_net(optimised_z_Sparse,is_training = False)
        sparseDCS_reconstloss_itr.append(recont_loss_Sparse)
        total_loss_Sparse_itr.append(total_loss_Sparse)

        if step % export_every == 0:
            rescont_sparse_file = 'reconstructions_Sparse_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np_Sparse = postprocess(reconstructions_Sparse)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_Sparse'))
            reconstructions_np_Sparse = tf.reshape(reconstructions_np_Sparse, data_np.shape)
            sample_exporter.save(reconstructions_np_Sparse, rescont_sparse_file)
            sample_exporter.save(data_np, data_file)

        # Deep Compressive Sensing
        with tf.GradientTape() as tape:
            z_i = tf.identity(generator_inputs)
            x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])

            total_loss, generator_loss, recont_loss, optimised_z = optimization_loss(z_i, x_img_reshape, generator_net,
                                                                                     measure_net)
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
        # reconstructions = generator_net(optimised_z,is_training = False)
        DCS_reconstloss_itr.append(recont_loss)
        total_loss_itr.append(total_loss)

        if step % export_every == 0:
            rescont_file = 'reconstructions_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np = postprocess(reconstructions)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions'))
            reconstructions_np = tf.reshape(reconstructions_np, data_np.shape)
            sample_exporter.save(reconstructions_np, rescont_file)
            sample_exporter.save(data_np, data_file)

    sparseDCS_reconstloss[epoch] = np.mean(np.array(sparseDCS_reconstloss_itr))
    l0DCS_reconstloss[epoch] = np.mean(np.array(l0DCS_reconstloss_itr))
    l1DCS_reconstloss[epoch] = np.mean(np.array(l1DCS_reconstloss_itr))
    lpDCS_reconstloss[epoch] = np.mean(np.array(lpDCS_reconstloss_itr))
    mcpDCS_reconstloss[epoch] = np.mean(np.array(mcpDCS_reconstloss_itr))
    DCS_reconstloss[epoch] = np.mean(np.array(DCS_reconstloss_itr))


    total_loss_Sparse_itr_mean[epoch] = np.mean(np.array(total_loss_Sparse_itr))
    total_loss_l0_itr_mean[epoch] = np.mean(np.array(total_loss_l0_itr))
    total_loss_l1_itr_mean[epoch] = np.mean(np.array(total_loss_l1_itr))
    total_loss_lp_itr_mean[epoch] = np.mean(np.array(total_loss_lp_itr))
    total_loss_mcp_itr_mean[epoch] = np.mean(np.array(total_loss_mcp_itr))
    total_loss_itr_mean[epoch] = np.mean(np.array(total_loss_itr))
    # Validation
    # step = 0
    # sparseDCS_reconstloss_valid_itr = []
    # DCS_reconstloss_valid_itr = []
    # for step in range(bat_valid_per_epo):
    #     x_batch_valid = get_train_dataset(valid_dataset, batch_size)
    #     # step = step + 1
    #     # if step == 10:
    #     #     break
    #     generatorSparse_inputs_v = get_Sparseprior(batch_size)
    #     z_i_Sparse_v = tf.identity(generatorSparse_inputs_v)
    #     x_img_reshape_v = tf.reshape(x_batch_valid, [-1, tf.shape(x_batch_valid)[1] * tf.shape(x_batch_valid)[2]])
    #     meas_img_Sparse_v = measureSparse_net(x_img_reshape_v)
    #     optimised_z_Sparse_v = optimise_and_sample_Sparse(z_i_Sparse_v, meas_img_Sparse_v, generatorSparse_net,
    #                                                       measureSparse_net)
    #     reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v)
    #     # reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v,is_training = False)
    #     sparseDCS_reconstloss_valid = tf.reduce_mean(tf.norm(reconstructions_Sparse_v - x_img_reshape_v, axis=-1))
    #     sparseDCS_reconstloss_valid_itr.append(sparseDCS_reconstloss_valid)
    #     print('Valid_recont_loss_Sparse %s\n' % (sparseDCS_reconstloss_valid))

    #     # generator_inputs_v = prior.sample(batch_size)
    #     generator_inputs_v = get_prior(num_latents,batch_size)
    #     z_i_v = tf.identity(generator_inputs_v)
    #     meas_img_v = measure_net(x_img_reshape_v)
    #     optimised_z_v = optimise_and_sample(z_i_v, meas_img_v, generator_net, measure_net)
    #     reconstructions_v = generator_net(optimised_z_v)
    #     # reconstructions_v = generator_net(optimised_z_v,is_training = False)
    #     DCS_reconstloss_valid = tf.reduce_mean(tf.norm(reconstructions_v - x_img_reshape_v, axis=-1))
    #     DCS_reconstloss_valid_itr.append(DCS_reconstloss_valid)
    #     print('Valid_recont_loss %s\n' % (DCS_reconstloss_valid))

    # DCS_valid_reconstloss[epoch] = np.mean(np.array(DCS_reconstloss_valid_itr))
    # sparseDCS_valid_reconstloss[epoch] = np.mean(np.array(sparseDCS_reconstloss_valid_itr))

    # Model Saving
    if (epoch + 1) % 2 == 0:
        filename1 = os.getcwd() + '/' + output_dir + '/gen_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename2 = os.getcwd() + '/' + output_dir + '/meas_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename3 = os.getcwd() + '/' + output_dir + '/genSpar_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename4 = os.getcwd() + '/' + output_dir + '/measSpar_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename5 = os.getcwd() + '/' + output_dir + '/genl1_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename6 = os.getcwd() + '/' + output_dir + '/measl1_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename7 = os.getcwd() + '/' + output_dir + '/genlp_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename8 = os.getcwd() + '/' + output_dir + '/measlp_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename9 = os.getcwd() + '/' + output_dir + '/genmcp_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename10 = os.getcwd() + '/' + output_dir + '/measmcp_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename11 = os.getcwd() + '/' + output_dir + '/genl0_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)
        filename12 = os.getcwd() + '/' + output_dir + '/measl0_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, sparse_dim, epoch)

        generator_net.save_weights(filename1, save_format='tf')
        measure_net.save_weights(filename2, save_format='tf')
        generatorSparse_net.save_weights(filename3, save_format='tf')
        measureSparse_net.save_weights(filename4, save_format='tf')
        generatorl1_net.save_weights(filename5, save_format='tf')
        measurel1_net.save_weights(filename6, save_format='tf')
        generatorlp_net.save_weights(filename7, save_format='tf')
        measurelp_net.save_weights(filename8, save_format='tf')
        generatormcp_net.save_weights(filename9, save_format='tf')
        measuremcp_net.save_weights(filename10, save_format='tf')
        generatorl0_net.save_weights(filename11, save_format='tf')
        measurel0_net.save_weights(filename12, save_format='tf')

        mat_file = os.getcwd() + '/' + output_dir + '/saved_var_%d_%d_%d.mat' % (
        num_measurements, num_latents, sparse_dim)
        scipy.io.savemat(mat_file, mdict={'total_loss_Sparse_itr_mean': total_loss_Sparse_itr_mean,
                                          'total_loss_l0_itr_mean': total_loss_l1_itr_mean,
                                          'total_loss_l1_itr_mean': total_loss_l1_itr_mean,
                                          'total_loss_lp_itr_mean': total_loss_lp_itr_mean,
                                          'total_loss_mcp_itr_mean': total_loss_mcp_itr_mean,
                                          'total_loss_itr_mean': total_loss_itr_mean,
                                          'sparseDCS_reconstloss': sparseDCS_reconstloss,
                                          'l0DCS_reconstloss': l1DCS_reconstloss,
                                          'l1DCS_reconstloss': l1DCS_reconstloss,
                                          'lpDCS_reconstloss': lpDCS_reconstloss,
                                          'mcpDCS_reconstloss': mcpDCS_reconstloss,
                                          'DCS_reconstloss': DCS_reconstloss,
                                          # 'DCS_valid_reconstloss': DCS_valid_reconstloss,
                                          # 'l1DCS_valid_reconstloss': l1DCS_valid_reconstloss,
                                          # 'sparseDCS_valid_reconstloss': sparseDCS_valid_reconstloss
                                          })

# Testing
sparseDCS_test_reconstloss = []
l0DCS_test_reconstloss = []
l1DCS_test_reconstloss = []
lpDCS_test_reconstloss = []
mcpDCS_test_reconstloss = []
DCS_test_reconstloss = []

for step in range(bat_test_per_epo):
    x_batch_test = get_train_dataset(test_dataset, batch_size)
    # generatorSparse_inputs_tst = get_Sparseprior(batch_size)

    generatormcp_inputs_tst = get_prior(num_latents, batch_size)
    z_i_mcp_tst = tf.identity(generatormcp_inputs_tst)
    x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
    meas_img_mcp_tst = measuremcp_net(x_img_reshape_tst)
    optimised_z_mcp_tst = optimise_and_sample_mcp(z_i_mcp_tst, meas_img_mcp_tst, generatormcp_net,
                                                  measuremcp_net)
    reconstructions_mcp_tst = generatormcp_net(optimised_z_mcp_tst)
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    mcpDCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_mcp_tst - x_img_reshape_tst, axis=-1))
    mcpDCS_test_reconstloss.append(mcpDCS_test_reconstloss_itr)
    print('Test_recont_loss_mcp %s\n' % (mcpDCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_mcp_tst_np = postprocess(reconstructions_mcp_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_mcp_tst'))
    reconstructions_mcp_tst_np = tf.reshape(reconstructions_mcp_tst_np, data_np.shape)
    rescont_test_mcp_file = 'reconstructions_mcp_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_mcp_tst_np, rescont_test_mcp_file)
    sample_exporter.save(data_np, data_file)

    generatorlp_inputs_tst = get_prior(num_latents, batch_size)
    z_i_lp_tst = tf.identity(generatorlp_inputs_tst)
    x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
    meas_img_lp_tst = measurelp_net(x_img_reshape_tst)
    optimised_z_lp_tst = optimise_and_sample_lp(z_i_lp_tst, meas_img_lp_tst, generatorlp_net,
                                                measurelp_net)
    reconstructions_lp_tst = generatorlp_net(optimised_z_lp_tst)
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    lpDCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_lp_tst - x_img_reshape_tst, axis=-1))
    lpDCS_test_reconstloss.append(lpDCS_test_reconstloss_itr)
    print('Test_recont_loss_lp %s\n' % (lpDCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_lp_tst_np = postprocess(reconstructions_lp_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_lp_tst'))
    reconstructions_lp_tst_np = tf.reshape(reconstructions_lp_tst_np, data_np.shape)
    rescont_test_lp_file = 'reconstructions_lp_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_lp_tst_np, rescont_test_lp_file)
    sample_exporter.save(data_np, data_file)

    generatorl0_inputs_tst = get_prior(num_latents, batch_size)
    z_i_l0_tst = tf.identity(generatorl0_inputs_tst)
    x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
    meas_img_l0_tst = measurel0_net(x_img_reshape_tst)
    optimised_z_l0_tst = optimise_and_sample_l0(z_i_l0_tst, meas_img_l0_tst, generatorl0_net,
                                                measurel0_net)
    reconstructions_l0_tst = generatorl0_net(optimised_z_l0_tst)
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    l0DCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_l0_tst - x_img_reshape_tst, axis=-1))
    l0DCS_test_reconstloss.append(l0DCS_test_reconstloss_itr)
    print('Test_recont_loss_l0 %s\n' % (l0DCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_l0_tst_np = postprocess(reconstructions_l0_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_l0_tst'))
    reconstructions_l0_tst_np = tf.reshape(reconstructions_l0_tst_np, data_np.shape)
    rescont_test_l0_file = 'reconstructions_l0_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_l0_tst_np, rescont_test_l0_file)
    sample_exporter.save(data_np, data_file)

    generatorl1_inputs_tst = get_prior(num_latents, batch_size)
    z_i_l1_tst = tf.identity(generatorl1_inputs_tst)
    x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
    meas_img_l1_tst = measurel1_net(x_img_reshape_tst)
    optimised_z_l1_tst = optimise_and_sample_l1(z_i_l1_tst, meas_img_l1_tst, generatorl1_net,
                                                measurel1_net)
    reconstructions_l1_tst = generatorl1_net(optimised_z_l1_tst)
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    l1DCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_l1_tst - x_img_reshape_tst, axis=-1))
    l1DCS_test_reconstloss.append(l1DCS_test_reconstloss_itr)
    print('Test_recont_loss_l1 %s\n' % (l1DCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_l1_tst_np = postprocess(reconstructions_l1_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_l1_tst'))
    reconstructions_l1_tst_np = tf.reshape(reconstructions_l1_tst_np, data_np.shape)
    rescont_test_l1_file = 'reconstructions_l1_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_l1_tst_np, rescont_test_l1_file)
    sample_exporter.save(data_np, data_file)

    # x_batch_test = get_train_dataset(test_dataset, batch_size)
    generatorSparse_inputs_tst = get_Sparseprior(batch_size)
    z_i_Sparse_tst = tf.identity(generatorSparse_inputs_tst)
    x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
    meas_img_Sparse_tst = measureSparse_net(x_img_reshape_tst)
    optimised_z_Sparse_tst = optimise_and_sample_Sparse(z_i_Sparse_tst, meas_img_Sparse_tst, generatorSparse_net,
                                                        measureSparse_net)
    reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst)
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    sparseDCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_Sparse_tst - x_img_reshape_tst, axis=-1))
    sparseDCS_test_reconstloss.append(sparseDCS_test_reconstloss_itr)
    print('Test_recont_loss_ll0 %s\n' % (sparseDCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_Sparse_tst_np = postprocess(reconstructions_Sparse_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_l0_test'))
    reconstructions_Sparse_tst_np = tf.reshape(reconstructions_Sparse_tst_np, data_np.shape)
    rescont_test_sparse_file = 'reconstructions_l0_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_Sparse_tst_np, rescont_test_sparse_file)
    sample_exporter.save(data_np, data_file)

    generator_inputs_tst = get_prior(num_latents, batch_size)
    z_i_test = tf.identity(generator_inputs_tst)
    meas_img_tst = measure_net(x_img_reshape_tst)
    optimised_z_tst = optimise_and_sample(z_i_test, meas_img_tst, generator_net, measure_net)
    reconstructions_tst = generator_net(optimised_z_tst)
    # reconstructions_tst = generator_net(optimised_z_tst,is_training = False)
    DCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_tst - x_img_reshape_tst, axis=-1))
    DCS_test_reconstloss.append(DCS_test_reconstloss_itr)
    print('Test_recont_loss %s\n' % (DCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_tst_np = postprocess(reconstructions_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_test'))
    reconstructions_tst_np = tf.reshape(reconstructions_tst_np, data_np.shape)
    rescont_test_file = 'reconstructions_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_tst_np, rescont_test_file)
    sample_exporter.save(data_np, data_file)

reconstructions_Sparse_tst_np_reshape = tf.reshape(reconstructions_Sparse_tst_np, [batch_size, data_dim])
reconstructions_l0_tst_np_reshape = tf.reshape(reconstructions_l0_tst_np, [batch_size, data_dim])
reconstructions_l1_tst_np_reshape = tf.reshape(reconstructions_l1_tst_np, [batch_size, data_dim])
reconstructions_lp_tst_np_reshape = tf.reshape(reconstructions_lp_tst_np, [batch_size, data_dim])
reconstructions_mcp_tst_np_reshape = tf.reshape(reconstructions_mcp_tst_np, [batch_size, data_dim])
data_np_reshape = tf.reshape(data_np, [batch_size, data_dim])
reconstructions_tst_np_reshape = tf.reshape(reconstructions_tst_np, [batch_size, data_dim])
mat_file = os.getcwd() + '/' + output_dir + '/Final_saved_%d_%d_%d.mat' % (num_measurements, num_latents, sparse_dim)
scipy.io.savemat(mat_file, mdict={'total_loss_Sparse_itr_mean': total_loss_Sparse_itr_mean,
                                  'total_loss_l0_itr_mean': total_loss_l0_itr_mean,
                                  'total_loss_l1_itr_mean': total_loss_l1_itr_mean,
                                  'total_loss_lp_itr_mean': total_loss_lp_itr_mean,
                                  'total_loss_mcp_itr_mean': total_loss_mcp_itr_mean,
                                  'total_loss_itr_mean': total_loss_itr_mean,
                                  'DCS_test_reconstloss': DCS_test_reconstloss,
                                  'sparseDCS_test_reconstloss': sparseDCS_test_reconstloss,
                                  'lpDCS_test_reconstloss': lpDCS_test_reconstloss,
                                  'mcpDCS_test_reconstloss': mcpDCS_test_reconstloss,
                                  'l0DCS_test_reconstloss': l0DCS_test_reconstloss,
                                  'l1DCS_test_reconstloss': l1DCS_test_reconstloss,
                                  'sparseDCS_reconstloss': sparseDCS_reconstloss,
                                  'l0DCS_reconstloss': l0DCS_reconstloss,
                                  'l1DCS_reconstloss': l1DCS_reconstloss,
                                  'lpDCS_reconstloss': lpDCS_reconstloss,
                                  'mcpDCS_reconstloss': mcpDCS_reconstloss,
                                  'DCS_reconstloss': DCS_reconstloss,
                                  # 'DCS_valid_reconstloss': DCS_valid_reconstloss,
                                  # 'sparseDCS_valid_reconstloss': sparseDCS_valid_reconstloss,
                                  # 'l1DCS_valid_reconstloss': sparseDCS_valid_reconstloss,
                                  'data_np_reshape': data_np_reshape.numpy(),
                                  'reconstructions_tst_np_reshape': reconstructions_tst_np_reshape.numpy(),
                                  'reconstructions_l0_tst_np_reshape': reconstructions_l0_tst_np_reshape.numpy(),
                                  'reconstructions_l1_tst_np_reshape': reconstructions_l1_tst_np_reshape.numpy(),
                                  'reconstructions_lp_tst_np_reshape': reconstructions_lp_tst_np_reshape.numpy(),
                                  'reconstructions_mcp_tst_np_reshape': reconstructions_mcp_tst_np_reshape.numpy(),
                                  'reconstructions_Sparse_tst_np_reshape': reconstructions_Sparse_tst_np_reshape.numpy()})
print('saving .mat completed\n')
aa = 1
