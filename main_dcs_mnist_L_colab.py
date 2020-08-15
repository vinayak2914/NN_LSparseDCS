# Learned Non-linear measurement
import tensorflow as tf
import numpy as np
from numpy.random import seed
import nets_keras as nets
import tensorflow_probability as tfp
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
batch_size = 64
num_measurements = 25
num_z_iters = 5
z_project_method = 'norm'
epochs = 200
num_latents = 10
dim_latent = 784
export_every = 100
z_step_size = tf.Variable(tf.exp(math.log(0.01)), dtype=tf.float32)
rand_seed = 14
optimizer = tf.keras.optimizers.Adam(1e-4)
tf.random.set_seed(rand_seed)
seed(rand_seed)
output_dir = ('DeepcsTF2Lin_z%d_m%d_spr%d' % (num_z_iters, num_measurements, num_latents))


def measure_net(input_layer, meas_mtx):
    # noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    # return input_layer + noise
    m = num_measurements
    # std = 1
    # meas_mtx = tf.random.normal(shape=[1,m,input_layer.shape[1]*input_layer.shape[2]], mean=0.0, stddev=1/m, dtype=tf.float32)

    meas_mtx = tf.repeat(meas_mtx, repeats=input_layer.shape[0], axis=0)

    # x_signal = tf.reshape(tf.transpose(input_layer,perm=[0,2,1,3]),[input_layer.shape[0],input_layer.shape[1]*input_layer.shape[2],input_layer.shape[3]])
    x_signal = tf.reshape(input_layer, [-1, input_layer.shape[1], 1])
    # print (x_signal.shape)
    # print (meas_mtx.shape)
    xy_batch_dot = tf.keras.backend.batch_dot(x_signal, meas_mtx, axes=(1, 2))
    # print (xy_batch_dot.shape)
    # meas_out = tf.transpose(xy_batch_dot,perm=[0,2,1])
    meas_out = tf.reshape(xy_batch_dot, [-1, xy_batch_dot.shape[2]])
    # return tf.abs(xy_batch_dot)
    # return  tf.square(tf.abs(meas_out))
    return meas_out


def get_rep_loss(img1, img2, measure_net):
    batch_size = tf.shape(img1)[0].numpy()
    m1 = measure_net(img1, meas_mtx)
    m2 = measure_net(img2, meas_mtx)

    img_diff_norm = tf.norm(img1 - img2, axis=-1)
    m_diff_norm = tf.norm(m1 - m2, axis=-1)
    return tf.square(img_diff_norm - m_diff_norm)


def get_measurement_error(target_meas, sample_img, measure_net):
    # m_targets,_ = measure_net(tf.reshape(target_img, [tf.shape(sample_img)[0].numpy(), 784]))
    # m_samples,_ = measure_net(tf.reshape(sample_img, [tf.shape(sample_img)[0].numpy(), 784]))
    # sum_sqr = tf.reduce_sum(tf.square(m_samples - m_targets), -1)
    return tf.reduce_sum(tf.square(tf.keras.layers.Flatten()(measure_net(sample_img, meas_mtx)) -
                                   tf.keras.layers.Flatten()(target_meas)), -1)


def gen_loss_fn(data, samples, measure_net):
    return get_measurement_error(data, samples, measure_net)


# def optimization_loss(meas_img_Sparse, optimized_sample_Sparse, measure_net):
#     return tf.reduce_mean(gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measure_net))


def get_optimisation_cost(initial_z, optimised_z):
    optimisation_cost = tf.reduce_mean(tf.reduce_sum((optimised_z - initial_z) ** 2, -1))
    return optimisation_cost


def optimization_loss_sparse(z_i_Sparse, x_img_reshape, generatorSparse_net, measure_net, meas_mtx):
    meas_img_Sparse = measure_net(tf.keras.layers.Flatten()(x_batch_copy), meas_mtx)
    optimised_z_Sparse,v = optimise_and_sample_Sparse(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                    measure_net, is_training=True)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse, is_training=True) +v
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse, is_training=True) +v
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measure_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(optimized_sample_Sparse)
                                                - tf.keras.layers.Flatten()(x_batch_copy), axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measure_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
    meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
    return (generator_loss_Sparse + meas_loss_Sparse), generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse


def optimization_loss(z_i_Sparse, x_img_reshape, generatorSparse_net, measure_net, meas_mtx):
    meas_img_Sparse = measure_net(tf.keras.layers.Flatten()(x_batch_copy), meas_mtx)
    optimised_z_Sparse= optimise_and_sample(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
                                                    measure_net, is_training=True)
    optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse, is_training=True)
    initial_sample_Sparse = generatorSparse_net(z_i_Sparse, is_training=True)
    generator_loss_Sparse = tf.reduce_mean(
        gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measure_net))
    recont_loss_Sparse = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(optimized_sample_Sparse)
                                                - tf.keras.layers.Flatten()(x_batch_copy), axis=-1))
    r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measure_net)
    r2_Sparse = get_rep_loss(optimized_sample_Sparse, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
    r3_Sparse = get_rep_loss(initial_sample_Sparse, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
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


def optimise_and_sample_Sparse(init_z, data, generator_net, measure_net, is_training):
    if num_z_iters == 0:
        z_final = init_z
    else:
        v_k = get_prior(dim_latent, batch_size)
        init_loop_vars = (0, project_z(init_z, z_project_method),v_k)

        loop_cond = lambda i, _,v_k: i < num_z_iters

        def loop_body(i, z,v_k):
            # z_grad_update = 0
            with tf.GradientTape() as tape:
                loop_samples = generator_net(z, is_training)
                tape.watch(z)
                gen_loss = gen_loss_fn(data, loop_samples, measure_net)

            # print(tf.reduce_mean(gen_loss))
            z_grad = tape.gradient(gen_loss, z)
            # z_grad_update = z_step_size * 0.9 * z_grad_update + z_step_size * z_grad
            z = z - z_step_size * z_grad
            indices_to_remove = tf.abs(z) >= tf.math.top_k(tf.abs(z), num_latents)[0][..., -1, None]
            indices_float = tf.dtypes.cast(indices_to_remove, tf.float32)
            z = tf.math.multiply(indices_float, z)
            z = project_z(z, z_project_method)

            loop_samples_update = generator_net(z, is_training)
            y_update = data - measure_net(loop_samples_update, meas_mtx)
            temp = measure_net(v_k, meas_mtx) - y_update
            v_k = v_k - 2 * 0.1 * measure_net(temp, tf.transpose(meas_mtx,perm=[0, 2, 1]))
            v_k = tfp.math.soft_threshold(v_k, .1)

            return i + 1, z,v_k

        _, z_final,v_final = tf.while_loop(loop_cond, loop_body, init_loop_vars)
        # return module.generator_net(z_final, is_training), z_final
        return z_final,v_final


def optimise_and_sample(init_z, data, generator_net, measure_net, is_training):
    if num_z_iters == 0:
        z_final = init_z
    else:
        init_loop_vars = (0, project_z(init_z, z_project_method))
        loop_cond = lambda i,_: i < num_z_iters

        def loop_body(i, z):
            with tf.GradientTape() as tape:
                loop_samples = generator_net(z, is_training)
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
        x = x / np.iinfo(x.dtype).max
        x = x.astype(np.float32)
        # x = x.reshape(60000, 784)
        x = x.reshape((-1, x.shape[1], x.shape[2], 1))
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
    # x_train = get_np_data(dataset, split='train')
    # choose random instances
    # ix = np.random.randint(0, x_train.shape[0], batch_size)
    # retrieve selected images
    X = tf.convert_to_tensor(x_test[step * batch_size:(step * batch_size) + batch_size], dtype='float32')
    return X


def get_Sparseprior(batch_size):
    z = np.zeros([batch_size, dim_latent], dtype=np.float32)
    for i in range(batch_size):
        z_idx = np.random.choice(dim_latent, [num_latents])
        z[i, z_idx] = np.random.normal(0, 1, size=[num_latents])
    return tf.identity(z)


# generate points in latent space as input for the generator
def get_prior(latent_dim, batch_size):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * batch_size)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(batch_size, latent_dim)
    return tf.convert_to_tensor(x_input, dtype='float32')


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
load_epoch = 33

if load_weights == 1:

    print('loading .mat files\n')
    start_epoch = load_epoch + 1
    mat_file = os.getcwd() + '/' + output_dir + '/saved_var_%d_%d_%d.mat' % (
        num_measurements, num_latents, dim_latent)
    dict_mat = loadmat(mat_file)

    total_loss_Sparse_itr_mean = np.reshape(dict_mat.get('total_loss_Sparse_itr_mean'), -1)
    total_loss_itr_mean = np.reshape(dict_mat.get('total_loss_itr_mean'), -1)
    sparseDCS_reconstloss = np.reshape(dict_mat.get('sparseDCS_reconstloss'), -1)
    DCS_reconstloss = np.reshape(dict_mat.get('DCS_reconstloss'), -1)
    DCS_valid_reconstloss = np.reshape(dict_mat.get('DCS_valid_reconstloss'), -1)
    sparseDCS_valid_reconstloss = np.reshape(dict_mat.get('sparseDCS_valid_reconstloss'), -1)
    meas_mtx = dict_mat.get('meas_mtx')
    meas_mtx = tf.convert_to_tensor(meas_mtx)
    # meas_mtx = tf.random.normal(shape=[1,50,28*28], mean=0.0, stddev=1/50, dtype=tf.float32)

    generator_net_file = os.getcwd() + '/' + output_dir + '/gen_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, dim_latent, load_epoch)
    # measure_net_file = os.getcwd()+'/DeepcsTF2_m%d_spr%d/meas_n_%d_%d_%d_%d' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent,load_epoch)
    generatorSparse_net_file = os.getcwd() + '/' + output_dir + '/genSpar_n_%d_%d_%d_%d' % (
        num_measurements, num_latents, dim_latent, load_epoch)
    # measureSparse_net_file = os.getcwd()+'/DeepcsTF2_m%d_spr%d/measSpar_n_%d_%d_%d_%d' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent,load_epoch)

    load_sparse_input = get_Sparseprior(batch_size)
    load_input = get_prior(num_latents, batch_size)
    load_Sparse_z = tf.identity(load_sparse_input)
    load_z = tf.identity(load_input)

    generator_net = nets.MLPGenNet()
    generatorSparse_net = nets.MLPGenNetSparse()
    # load_sample_Sparse = generatorSparse_net(load_Sparse_z, is_training=True)
    # load_meas_img_Sparse = gaussian_noise_layer(load_sample_Sparse, std)

    # load_sample_Sparse = generator_net(load_z, is_training=True)
    # load_meas_img_Sparse = gaussian_noise_layer(load_sample_Sparse, std)
    # generator_net.compile(loss=optimization_loss(load_meas_img_Sparse, load_sample_Sparse, gaussian_noise_layer),
    #                       optimizer=optimizer)
    # generatorSparse_net.compile(loss=optimization_loss(load_meas_img_Sparse, load_sample_Sparse, gaussian_noise_layer),
    #                             optimizer=optimizer)
    generator_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=optimizer)
    generatorSparse_net.compile(loss=tf.keras.losses.MeanSquaredError(),
                                optimizer=optimizer)
    # measure_net.compile(loss= tf.keras.losses.MeanSquaredError(),
    #                   optimizer=optimizer)
    # measureSparse_net.compile(loss= tf.keras.losses.MeanSquaredError(),
    #                   optimizer=optimizer)
    load_sample_Sparse = generatorSparse_net(load_Sparse_z, is_training=True)
    load_sample_Sparse = generator_net(load_z, is_training=True)

    generator_net.load_weights(generator_net_file)
    generatorSparse_net.load_weights(generatorSparse_net_file)
    print('loading models completed\n')
else:

    ######################################################################################
    start_epoch = 0
    make_output_dir(output_dir)
    generator_net = nets.MLPGenNet()
    # measure_net = nets.SNConvMesNet(num_measurements)
    generatorSparse_net = nets.MLPGenNetSparse()
    # measureSparse_net = nets.SNConvMesNetSparse(num_measurements)
    meas_mtx = tf.random.normal(shape=[1, num_measurements, train_dataset.shape[1] * train_dataset.shape[2]], mean=0.0,
                                stddev=1 / num_measurements, dtype=tf.float32)

    sparseDCS_reconstloss = np.zeros(epochs)
    DCS_reconstloss = np.zeros(epochs)
    total_loss_Sparse_itr_mean = np.zeros(epochs)
    total_loss_itr_mean = np.zeros(epochs)
    sparseDCS_valid_reconstloss = np.zeros(epochs)
    DCS_valid_reconstloss = np.zeros(epochs)

######################################################################################

for epoch in range(start_epoch, epochs):
    sparseDCS_reconstloss_itr = []
    total_loss_Sparse_itr = []
    DCS_reconstloss_itr = []
    total_loss_itr = []
    print('epoch %s: started' % (epoch))
    for step in range(bat_per_epo):
        x_batch_train = get_train_dataset(train_dataset, batch_size)
        x_batch_test = get_test_dataset(step, test_dataset, batch_size)
        print('> Epoch:%d  Iteration:%d:' % (epoch, step))
        generatorSparse_inputs = get_Sparseprior(batch_size)
        generator_inputs = get_prior(num_latents, batch_size)
        x_batch_copy = tf.identity(x_batch_train)
        # Sparse Deep Compressive Sensing
        with tf.GradientTape() as tapeSparse:
            z_i_Sparse = tf.identity(generatorSparse_inputs)
            total_loss_Sparse, generator_loss_Sparse, recont_loss_Sparse, optimised_z_Sparse = optimization_loss_sparse(
                z_i_Sparse,
                x_batch_copy,
                generatorSparse_net, measure_net, meas_mtx)
            # x_img_reshape = tf.reshape(x_batch_copy, [-1, tf.shape(x_batch_copy)[1] * tf.shape(x_batch_copy)[2]])
            # meas_img_Sparse = measure_net(tf.keras.layers.Flatten()(x_batch_copy), meas_mtx)
            # optimised_z_Sparse = optimise_and_sample_Sparse(z_i_Sparse, meas_img_Sparse, generatorSparse_net,
            #                                                 measure_net, is_training=True)
            # optimized_sample_Sparse = generatorSparse_net(optimised_z_Sparse, is_training=True)
            # initial_sample_Sparse = generatorSparse_net(z_i_Sparse, is_training=True)
            # generator_loss_Sparse = optimization_loss(meas_img_Sparse, optimized_sample_Sparse, measure_net)
            # recont_loss_Sparse = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(optimized_sample_Sparse)
            #                                             - tf.keras.layers.Flatten()(x_batch_copy), axis=-1))
            # r1_Sparse = get_rep_loss(optimized_sample_Sparse, initial_sample_Sparse, measure_net)
            # r2_Sparse = get_rep_loss(optimized_sample_Sparse, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
            # r3_Sparse = get_rep_loss(initial_sample_Sparse, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
            # meas_loss_Sparse = tf.reduce_mean((r1_Sparse + r2_Sparse + r3_Sparse) / 3.0)
            # total_loss_Sparse = (generator_loss_Sparse + meas_loss_Sparse)
            gen_var_Sparse = generatorSparse_net.trainable_variables
            # meas_var_Sparse = measureSparse_net.trainable_variables
            train_var_Sparse = gen_var_Sparse
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
        # reconstructions_Sparse = generatorSparse_net(optimised_z_Sparse,is_training = False)
        sparseDCS_reconstloss_itr.append(recont_loss_Sparse)
        total_loss_Sparse_itr.append(total_loss_Sparse)

        if step % export_every == 0:
            rescont_sparse_file = 'reconstructions_sparse_%d_%d' % (epoch, step)
            data_file = 'data_%d_%d' % (epoch, step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_copy)
            reconstructions_np_Sparse = postprocess(reconstructions_Sparse)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_sparse'))
            reconstructions_np_Sparse = tf.reshape(reconstructions_np_Sparse, data_np.shape)
            sample_exporter.save(reconstructions_np_Sparse, rescont_sparse_file)
            sample_exporter.save(data_np, data_file)

        # Deep Compressive Sensing
        with tf.GradientTape() as tape:
            z_i = tf.identity(generator_inputs)
            total_loss, generator_loss, recont_loss, optimised_z = optimization_loss(z_i, x_batch_copy, generator_net,
                                                                                     measure_net, meas_mtx)
            # meas_img = measure_net(tf.keras.layers.Flatten()(x_batch_copy), meas_mtx)
            # optimised_z = optimise_and_sample(z_i, meas_img, generator_net, measure_net, is_training=True)
            # optimized_sample = generator_net(optimised_z, is_training=True)
            # initial_sample = generator_net(z_i, is_training=True)
            # generator_loss = tf.reduce_mean(gen_loss_fn(meas_img, optimized_sample, measure_net))
            # recont_loss = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(optimized_sample)
            #                                      - tf.keras.layers.Flatten()(x_batch_copy), axis=-1))
            # r1 = get_rep_loss(optimized_sample, initial_sample, measure_net)
            # r2 = get_rep_loss(optimized_sample, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
            # r3 = get_rep_loss(initial_sample, tf.keras.layers.Flatten()(x_batch_copy), measure_net)
            # meas_loss = tf.reduce_mean((r1 + r2 + r3) / 3.0)
            # total_loss = (generator_loss + meas_loss)
            gen_var = generator_net.trainable_variables
            # meas_var = measure_net.trainable_variables
            train_var = gen_var
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
            data_np = postprocess(x_batch_copy)
            reconstructions_np = postprocess(reconstructions)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions'))
            reconstructions_np = tf.reshape(reconstructions_np, data_np.shape)
            sample_exporter.save(reconstructions_np, rescont_file)
            sample_exporter.save(data_np, data_file)

    sparseDCS_reconstloss[epoch] = np.mean(np.array(sparseDCS_reconstloss_itr))
    DCS_reconstloss[epoch] = np.mean(np.array(DCS_reconstloss_itr))
    total_loss_Sparse_itr_mean[epoch] = np.mean(np.array(total_loss_Sparse_itr))
    total_loss_itr_mean[epoch] = np.mean(np.array(total_loss_itr))
    # Validation
    step = 0
    sparseDCS_reconstloss_valid_itr = []
    DCS_reconstloss_valid_itr = []
    for step in range(bat_valid_per_epo):
        x_batch_valid = get_train_dataset(valid_dataset, batch_size)
        generatorSparse_inputs_v = get_Sparseprior(batch_size)
        z_i_Sparse_v = tf.identity(generatorSparse_inputs_v)
        meas_img_Sparse_v = measure_net(tf.keras.layers.Flatten()(x_batch_valid), meas_mtx)
        optimised_z_Sparse_v,v = optimise_and_sample_Sparse(z_i_Sparse_v, meas_img_Sparse_v, generatorSparse_net,
                                                          measure_net, is_training=True)
        reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v)+v
        # reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v,is_training = False)
        sparseDCS_reconstloss_valid = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(reconstructions_Sparse_v)
                                                             - tf.keras.layers.Flatten()(x_batch_valid), axis=-1))
        sparseDCS_reconstloss_valid_itr.append(sparseDCS_reconstloss_valid)
        print('Valid_recont_loss_Sparse %s\n' % (sparseDCS_reconstloss_valid))

        # generator_inputs_v = prior.sample(batch_size)
        generator_inputs_v = get_prior(num_latents, batch_size)
        z_i_v = tf.identity(generator_inputs_v)
        meas_img_v = measure_net(tf.keras.layers.Flatten()(x_batch_valid), meas_mtx)
        optimised_z_v = optimise_and_sample(z_i_v, meas_img_v, generator_net, measure_net, is_training=True)
        reconstructions_v = generator_net(optimised_z_v)
        # reconstructions_v = generator_net(optimised_z_v,is_training = False)
        DCS_reconstloss_valid = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(reconstructions_v)
                                                       - tf.keras.layers.Flatten()(x_batch_valid), axis=-1))
        DCS_reconstloss_valid_itr.append(DCS_reconstloss_valid)
        print('Valid_recont_loss %s\n' % (DCS_reconstloss_valid))

    DCS_valid_reconstloss[epoch] = np.mean(np.array(DCS_reconstloss_valid_itr))
    sparseDCS_valid_reconstloss[epoch] = np.mean(np.array(sparseDCS_reconstloss_valid_itr))

    # Model Saving
    if (epoch + 1) % 2 == 0:
        filename1 = os.getcwd() + '/' + output_dir + '/gen_n_%d_%d_%d_%d' % (
            num_measurements, num_latents, dim_latent, epoch)
        # filename2 = os.getcwd() + '/'+output_dir + '/meas_n_%d_%d_%d_%d' % (
        #  num_measurements, num_latents, dim_latent, epoch)
        filename3 = os.getcwd() + '/' + output_dir + '/genSpar_n_%d_%d_%d_%d' % (
            num_measurements, num_latents, dim_latent, epoch)
        # filename4 = os.getcwd() + '/'+output_dir + '/measSpar_n_%d_%d_%d_%d' % (
        #  num_measurements, num_latents, dim_latent, epoch)
        generator_net.save_weights(filename1, save_format='tf')
        # generator_net.save(filename1, save_format='tf')
        # measure_net.save(filename2, save_format='tf')
        generatorSparse_net.save_weights(filename3, save_format='tf')
        # measureSparse_net.save(filename4, save_format='tf')
        mat_file = os.getcwd() + '/' + output_dir + '/saved_var_%d_%d_%d.mat' % (
            num_measurements, num_latents, dim_latent)
        scipy.io.savemat(mat_file, mdict={'meas_mtx': meas_mtx.numpy(),
                                          'total_loss_Sparse_itr_mean': total_loss_Sparse_itr_mean,
                                          'total_loss_itr_mean': total_loss_itr_mean,
                                          'sparseDCS_reconstloss': sparseDCS_reconstloss,
                                          'DCS_reconstloss': DCS_reconstloss,
                                          'DCS_valid_reconstloss': DCS_valid_reconstloss,
                                          'sparseDCS_valid_reconstloss': sparseDCS_valid_reconstloss})

# Testing
sparseDCS_test_reconstloss = []
DCS_test_reconstloss = []

for step in range(bat_test_per_epo):
    # x_batch_test = get_train_dataset(test_dataset, batch_size)
    x_batch_test = get_test_dataset(step, test_dataset, batch_size)
    generatorSparse_inputs_tst = get_Sparseprior(batch_size)
    z_i_Sparse_tst = tf.identity(generatorSparse_inputs_tst)
    meas_img_Sparse_tst = measure_net(tf.keras.layers.Flatten()(x_batch_test), meas_mtx)
    optimised_z_Sparse_tst,v = optimise_and_sample_Sparse(z_i_Sparse_tst, meas_img_Sparse_tst, generatorSparse_net,
                                                        measure_net, is_training=True)
    reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst)+v
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    sparseDCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(reconstructions_Sparse_tst)
                                                            - tf.keras.layers.Flatten()(x_batch_test), axis=-1))
    sparseDCS_test_reconstloss.append(sparseDCS_test_reconstloss_itr)
    print('Test_recont_loss_Sparse %s\n' % (sparseDCS_test_reconstloss_itr))

    # datagt_np = postprocess(x_batch_test)
    data_np = postprocess(x_batch_test)
    reconstructions_Sparse_tst_np = postprocess(reconstructions_Sparse_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_Sparse_test'))
    reconstructions_Sparse_tst_np = tf.reshape(reconstructions_Sparse_tst_np, data_np.shape)
    rescont_test_sparse_file = 'reconstructions_Sparse_test_%d_%d_%d' % (num_measurements, num_latents, step)
    data_file = 'data_%d_%d_%d' % (num_measurements, num_latents, step)
    datagt_file = 'datagt_%d_%d_%d' % (num_measurements, num_latents, step)
    sample_exporter.save(reconstructions_Sparse_tst_np, rescont_test_sparse_file)
    sample_exporter.save(data_np, data_file)
    # sample_exporter.save(datagt_np, datagt_file)

    generator_inputs_tst = get_prior(num_latents, batch_size)
    z_i_test = tf.identity(generator_inputs_tst)
    meas_img_tst = measure_net(tf.keras.layers.Flatten()(x_batch_test), meas_mtx)
    optimised_z_tst = optimise_and_sample(z_i_test, meas_img_tst, generator_net, measure_net, is_training=True)
    reconstructions_tst = generator_net(optimised_z_tst)
    # reconstructions_tst = generator_net(optimised_z_tst,is_training = False)
    DCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(tf.keras.layers.Flatten()(reconstructions_tst)
                                                      - tf.keras.layers.Flatten()(x_batch_test), axis=-1))
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

reconstructions_Sparse_tst_np_reshape = tf.reshape(reconstructions_Sparse_tst_np,
                                                   [batch_size, data_dim, reconstructions_Sparse_tst_np.shape[3]])
data_np_reshape = tf.reshape(data_np, [batch_size, data_dim, data_np.shape[3]])
# datagt_np_reshape = tf.reshape(datagt_np, [batch_size, data_dim,datagt_np.shape[3]])
reconstructions_tst_np_reshape = tf.reshape(reconstructions_tst_np,
                                            [batch_size, data_dim, reconstructions_tst_np.shape[3]])
mat_file = os.getcwd() + '/' + output_dir + '/Final_saved_%d_%d_%d.mat' % (
    num_measurements, num_latents, dim_latent)
scipy.io.savemat(mat_file, mdict={'meas_mtx': meas_mtx.numpy(),
                                  'total_loss_Sparse_itr_mean': total_loss_Sparse_itr_mean,
                                  'total_loss_itr_mean': total_loss_itr_mean,
                                  'DCS_test_reconstloss': DCS_test_reconstloss,
                                  'sparseDCS_test_reconstloss': sparseDCS_test_reconstloss,
                                  'sparseDCS_reconstloss': sparseDCS_reconstloss,
                                  'DCS_reconstloss': DCS_reconstloss,
                                  'DCS_valid_reconstloss': DCS_valid_reconstloss,
                                  'sparseDCS_valid_reconstloss': sparseDCS_valid_reconstloss,
                                  'data_np_reshape': data_np_reshape.numpy(),
                                  # 'datagt_np_reshape': datagt_np_reshape.numpy(),
                                  'reconstructions_tst_np_reshape': reconstructions_tst_np_reshape.numpy(),
                                  'reconstructions_Sparse_tst_np_reshape': reconstructions_Sparse_tst_np_reshape.numpy()})
print('saving .mat completed\n')
aa = 1