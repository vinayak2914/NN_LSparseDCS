import tensorflow as tf
import numpy as np
from numpy.random import seed
import nets_keras as nets
import tensorflow_probability as tfp
import random
import collections
import tsne
import pylab
import os
# import sonnet as snt
import file_utils
import math
import scipy.io
from scipy.io import loadmat
import glob

def get_intersect_supp(opt_z):
    supp = np.array([],dtype = np.float32)
    for i in range(opt_z.shape[0]):
        temp_supp = (np.array(np.where(opt_z[i, :] != 0)))
        if i == 0:
            supp = temp_supp
        else:
            supp = np.array(np.intersect1d(temp_supp, supp))
    # for batch in range(batch_per_epoch):
    #     opt_z_arr = np.array(opt_z_list[batch])
    #     for i in range(opt_z_arr.shape[0]):
    #         temp_supp = (np.array(np.where(opt_z_arr[i, :] != 0)))
    #         if i == 0:
    #             supp = temp_supp
    #         else:
    #             supp = np.array(np.intersect1d(temp_supp,supp))

    return supp

i = 0
for filename in glob.glob('DeepcsTF2_z5_m50_spr100/mat_z/*.mat'):
    dict_mat = loadmat(filename)
    opt_z_sparse_n = (dict_mat.get('opt_z_sparse_n'))
    opt_z_n = (dict_mat.get('opt_z_n'))
    if i == 0:
        z_ProxDCS = opt_z_sparse_n
        z_DCS = opt_z_n
        labels = np.ones([opt_z_sparse_n.shape[0]])*i
        supp = get_intersect_supp(opt_z_sparse_n)
    else:
        z_ProxDCS = np.concatenate((z_ProxDCS,opt_z_sparse_n),axis = 0)
        z_DCS = np.concatenate((z_DCS, opt_z_n), axis=0)
        labels = np.concatenate((labels, np.ones([opt_z_sparse_n.shape[0]])*i), axis=0)
    i = i+1

Y = tsne.tsne(z_ProxDCS, 2, 50, 20.0)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.savefig('DeepcsTF2_z5_m50_spr100/mat_z/ProxDCS_tSNE_1.png')


