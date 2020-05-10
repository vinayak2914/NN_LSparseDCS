# Learned Non-linear measurement
import tensorflow as tf
import numpy as np
from numpy.random import seed
import net_nonlin_learn as nets
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



tfd = tfp.distributions


dataset = 'mnist'
batch_size = 64
num_measurements = 20
num_z_iters = 3
z_project_method = 'norm'
epochs = 200
num_latents = 100
dim_latent = 784
export_every = 100
z_step_size = tf.Variable(tf.exp(math.log(0.01)), dtype=tf.float32)
rand_seed = 14
optimizer = tf.keras.optimizers.Adam(1e-4)
tf.random.set_seed(rand_seed)
seed(rand_seed)
output_dir = ('DeepcsTF2_%d_%d'% (num_measurements,num_latents))

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
        x = x / np.iinfo(x.dtype).max
        x = x.astype(np.float32)
        x = x.reshape((-1, x.shape[1], x.shape[2], 1))
        x = preprocess(x)
    return x


def get_train_dataset(x_train, batch_size):
    # x_train = get_np_data(dataset, split='train')
    # choose random instances
    ix = np.random.randint(0, x_train.shape[0], batch_size)
    # retrieve selected images
    X = tf.convert_to_tensor(x_train[ix],dtype='float32')
    # x_train = x_train[0:6400, :, :, :]
    # x_test = get_np_data(dataset, split='test')
    # x_valid = x_test[0:int(x_test.shape[0] / 2), :, :, :]
    # dataset = tf.data.Dataset.from_tensor_slices(x_train)
    # dataset_valid = tf.data.Dataset.from_tensor_slices(x_valid)
    # dataset_test = tf.data.Dataset.from_tensor_slices(x_test)
    # # dataset = dataset.shuffle(100000).repeat().batch(batch_size)
    # dataset = dataset.shuffle(100000).batch(batch_size)
    # dataset_valid = dataset_valid.shuffle(100000).batch(batch_size)
    # dataset_test = dataset_test.shuffle(100000).batch(batch_size)
    # return X, dataset_valid, dataset_test
    return X

def get_test_dataset(step,x_test, batch_size):
    # x_train = get_np_data(dataset, split='train')
    # choose random instances
    # ix = np.random.randint(0, x_train.shape[0], batch_size)
    # retrieve selected images
    X = tf.convert_to_tensor(x_test[step*batch_size:(step*batch_size)+batch_size],dtype='float32')
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
    return tf.convert_to_tensor(x_input,dtype='float32')


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


# # dict_mat = loadmat('gen_n_25_100_784.mat')
#
# # sparseDCS_reconstloss = dict_mat.get('sparseDCS_reconstloss')
# # DCS_reconstloss = dict_mat.get('DCS_reconstloss')
# # DCS_valid_reconstloss = dict_mat.get('DCS_valid_reconstloss')
# # sparseDCS_valid_reconstloss = dict_mat.get('sparseDCS_valid_reconstloss')
#
# generator_net = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_20_100/gen_n_20_100_784_0139')
# measure_net = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_20_100/meas_n_20_100_784_0139')
# generatorSparse_net = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_20_100/genSpar_n_20_100_784_0139')
# measureSparse_net = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_20_100/measSpar_n_20_100_784_0139')


make_output_dir(output_dir)
generator_net = nets.MLPGenNet()
measure_net = nets.MLPMesNet(num_measurements)
generatorSparse_net = nets.MLPGenNetSparse()
measureSparse_net = nets.MLPMesNetSparse(num_measurements)

sparseDCS_reconstloss = np.zeros(epochs)
DCS_reconstloss = np.zeros(epochs)
sparseDCS_valid_reconstloss = np.zeros(epochs)
DCS_valid_reconstloss = np.zeros(epochs)

train_dataset = get_np_data(dataset, split='train')
test_dataset = get_np_data(dataset, split='test')
valid_dataset = test_dataset[0:int(test_dataset.shape[0]/2), :, :, :]
bat_per_epo = int(train_dataset.shape[0] / batch_size)
bat_valid_per_epo = int(valid_dataset.shape[0] / batch_size)
bat_test_per_epo = int(test_dataset.shape[0] / batch_size)
data_size = [train_dataset.shape[1],train_dataset.shape[2]]
data_dim = data_size[0]*data_size[1]

for epoch in range(140,epochs):
    sparseDCS_reconstloss_itr = []
    DCS_reconstloss_itr = []

    print('epoch %s: started' % (epoch))
    # step = 0
    for step in range(bat_per_epo):
        x_batch_train = get_train_dataset(train_dataset, batch_size)
        x_batch_test = get_test_dataset(step, test_dataset, batch_size)
        print('> Epoch:%d  Iteration:%d:' % (epoch,step))
        generatorSparse_inputs = get_Sparseprior(batch_size)
        generator_inputs = get_prior(num_latents,batch_size)
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
            generator_loss_Sparse = tf.reduce_mean(
                gen_loss_fn(meas_img_Sparse, optimized_sample_Sparse, measureSparse_net))
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
        # reconstructions_Sparse = generatorSparse_net(optimised_z_Sparse,is_training = False)
        sparseDCS_reconstloss_itr.append(recont_loss_Sparse)

        if step % export_every == 0:
            rescont_sparse_file = 'reconstructions_sparse_%d_%d' % (epoch,step)
            data_file = 'data_%d_%d' % (epoch,step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np_Sparse = postprocess(reconstructions_Sparse)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions_sparse'))
            reconstructions_np_Sparse = tf.reshape(reconstructions_np_Sparse, data_np.shape)
            sample_exporter.save(reconstructions_np_Sparse, rescont_sparse_file)
            sample_exporter.save(data_np, data_file)

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
        # reconstructions = generator_net(optimised_z,is_training = False)
        DCS_reconstloss_itr.append(recont_loss)

        if step % export_every == 0:
            rescont_file = 'reconstructions_%d_%d' % (epoch,step)
            data_file = 'data_%d_%d' % (epoch,step)
            # Create an object which gets data and does the processing.
            data_np = postprocess(x_batch_train)
            reconstructions_np = postprocess(reconstructions)
            sample_exporter = file_utils.FileExporter(
                os.path.join(output_dir, 'reconstructions'))
            reconstructions_np = tf.reshape(reconstructions_np, data_np.shape)
            sample_exporter.save(reconstructions_np, rescont_file)
            sample_exporter.save(data_np, data_file)

    sparseDCS_reconstloss[epoch] = np.mean(np.array(sparseDCS_reconstloss_itr))
    DCS_reconstloss[epoch] = np.mean(np.array(DCS_reconstloss_itr))
    # Validation
    step = 0
    sparseDCS_reconstloss_valid_itr = []
    DCS_reconstloss_valid_itr = []
    for step in range(bat_valid_per_epo):
        x_batch_valid = get_train_dataset(valid_dataset, batch_size)
        # step = step + 1
        # if step == 10:
        #     break
        generatorSparse_inputs_v = get_Sparseprior(batch_size)
        z_i_Sparse_v = tf.identity(generatorSparse_inputs_v)
        x_img_reshape_v = tf.reshape(x_batch_valid, [-1, tf.shape(x_batch_valid)[1] * tf.shape(x_batch_valid)[2]])
        meas_img_Sparse_v = measureSparse_net(x_img_reshape_v)
        optimised_z_Sparse_v = optimise_and_sample_Sparse(z_i_Sparse_v, meas_img_Sparse_v, generatorSparse_net,
                                                          measureSparse_net)
        reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v)
        # reconstructions_Sparse_v = generatorSparse_net(optimised_z_Sparse_v,is_training = False)
        sparseDCS_reconstloss_valid = tf.reduce_mean(tf.norm(reconstructions_Sparse_v - x_img_reshape_v, axis=-1))
        sparseDCS_reconstloss_valid_itr.append(sparseDCS_reconstloss_valid)
        print('Valid_recont_loss_Sparse %s\n' % (sparseDCS_reconstloss_valid))

        # generator_inputs_v = prior.sample(batch_size)
        generator_inputs_v = get_prior(num_latents,batch_size)
        z_i_v = tf.identity(generator_inputs_v)
        meas_img_v = measure_net(x_img_reshape_v)
        optimised_z_v = optimise_and_sample(z_i_v, meas_img_v, generator_net, measure_net)
        reconstructions_v = generator_net(optimised_z_v)
        # reconstructions_v = generator_net(optimised_z_v,is_training = False)
        DCS_reconstloss_valid = tf.reduce_mean(tf.norm(reconstructions_v - x_img_reshape_v, axis=-1))
        DCS_reconstloss_valid_itr.append(DCS_reconstloss_valid)
        print('Valid_recont_loss %s\n' % (DCS_reconstloss_valid))

    DCS_valid_reconstloss[epoch] = np.mean(np.array(DCS_reconstloss_valid_itr))
    sparseDCS_valid_reconstloss[epoch] = np.mean(np.array(sparseDCS_reconstloss_valid_itr))

# Model Saving
    if (epoch + 1) % 10 == 0:
        filename1 = '/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_%d_%d/gen_n_%d_%d_%d_%04d' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent,epoch)
        filename2 = '/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_%d_%d/meas_n_%d_%d_%d_%04d' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent,epoch)
        filename3 = '/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_%d_%d/genSpar_n_%d_%d_%d_%04d' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent,epoch)
        filename4 = '/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_%d_%d/measSpar_n_%d_%d_%d_%04d' %(num_measurements,num_latents,num_measurements,num_latents,dim_latent,epoch)
        generator_net.save(filename1, save_format='tf')
        measure_net.save(filename2, save_format='tf')
        generatorSparse_net.save(filename3, save_format='tf')
        measureSparse_net.save(filename4, save_format='tf')
        mat_file = '/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_%d_%d/saved_var_%d_%d_%d.mat' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent)
        scipy.io.savemat(mat_file, mdict={'sparseDCS_reconstloss': sparseDCS_reconstloss,
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
    x_img_reshape_tst = tf.reshape(x_batch_test, [-1, tf.shape(x_batch_test)[1] * tf.shape(x_batch_test)[2]])
    meas_img_Sparse_tst = measureSparse_net(x_img_reshape_tst)
    optimised_z_Sparse_tst = optimise_and_sample_Sparse(z_i_Sparse_tst, meas_img_Sparse_tst, generatorSparse_net,
                                                        measureSparse_net)
    reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst)
    # reconstructions_Sparse_tst = generatorSparse_net(optimised_z_Sparse_tst,is_training = False)
    sparseDCS_test_reconstloss_itr = tf.reduce_mean(tf.norm(reconstructions_Sparse_tst - x_img_reshape_tst, axis=-1))
    sparseDCS_test_reconstloss.append(sparseDCS_test_reconstloss_itr)
    print('Test_recont_loss_Sparse %s\n' % (sparseDCS_test_reconstloss_itr))

    data_np = postprocess(x_batch_test)
    reconstructions_Sparse_tst_np = postprocess(reconstructions_Sparse_tst)
    sample_exporter = file_utils.FileExporter(
        os.path.join(output_dir, 'reconstructions_Sparse_test'))
    reconstructions_Sparse_tst_np = tf.reshape(reconstructions_Sparse_tst_np, data_np.shape)
    rescont_test_sparse_file = 'reconstructions_Sparse_test_%d_%d_%d' % (num_measurements,num_latents,step)
    data_file = 'data_%d_%d_%d' % (num_measurements,num_latents,step)
    sample_exporter.save(reconstructions_Sparse_tst_np, rescont_test_sparse_file)
    sample_exporter.save(data_np, data_file)


    generator_inputs_tst = get_prior(num_latents,batch_size)
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
    rescont_test_file = 'reconstructions_test_%d_%d_%d' % (num_measurements,num_latents,step)
    data_file = 'data_%d_%d_%d' % (num_measurements,num_latents,step)
    sample_exporter.save(reconstructions_tst_np, rescont_test_file)
    sample_exporter.save(data_np, data_file)

reconstructions_Sparse_tst_np_reshape = tf.reshape(reconstructions_Sparse_tst_np, [batch_size, data_dim])
data_np_reshape = tf.reshape(data_np, [batch_size, data_dim])
reconstructions_tst_np_reshape = tf.reshape(reconstructions_tst_np, [batch_size, data_dim])
mat_file = '/content/drive/My Drive/Colab Notebooks/DCS_MNIST/DeepcsTF2_%d_%d/gen_n_%d_%d_%d.mat' % (num_measurements,num_latents,num_measurements,num_latents,dim_latent)
scipy.io.savemat(mat_file, mdict={'DCS_test_reconstloss': DCS_test_reconstloss,
                                              'sparseDCS_test_reconstloss': sparseDCS_test_reconstloss,
                                              'sparseDCS_reconstloss': sparseDCS_reconstloss,
                                              'DCS_reconstloss': DCS_reconstloss,
                                              'DCS_valid_reconstloss': DCS_valid_reconstloss,
                                              'sparseDCS_valid_reconstloss': sparseDCS_valid_reconstloss,
                                              'data_np_reshape': data_np_reshape.numpy(),
                                              'reconstructions_tst_np_reshape': reconstructions_tst_np_reshape.numpy(),
                                              'reconstructions_Sparse_tst_np_reshape': reconstructions_Sparse_tst_np_reshape.numpy()})
aa = 1