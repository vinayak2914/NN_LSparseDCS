from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import glob
import os

for filename in glob.glob('DCS_DN/*.mat'):
    dict_mat = loadmat(filename)
    meas = int(filename[13:15])
    spr = int(filename[16:19])
    sparseDCS_reconstloss = dict_mat.get('sparseDCS_reconstloss')
    DCS_reconstloss = dict_mat.get('DCS_reconstloss')
    DCS_valid_reconstloss = dict_mat.get('DCS_valid_reconstloss')
    sparseDCS_valid_reconstloss = dict_mat.get('sparseDCS_valid_reconstloss')
    data_np_reshape = dict_mat.get('data_np_reshape')
    DCS_test_reconstloss = dict_mat.get('DCS_test_reconstloss')
    sparseDCS_test_reconstloss = dict_mat.get('sparseDCS_test_reconstloss')
    reconstructions_tst_np_reshape = dict_mat.get('reconstructions_tst_np_reshape')
    reconstructions_Sparse_tst_np_reshape = dict_mat.get('reconstructions_Sparse_tst_np_reshape')

    # data_mse = np.mean(np.square(data_np_reshape_10_100 - data_np_reshape_10_100))
    sparseDCS_mse = np.mean(np.square(data_np_reshape - reconstructions_Sparse_tst_np_reshape))
    DCS_mse = np.mean(np.square(data_np_reshape - reconstructions_tst_np_reshape))
    # data_psnr = 10*np.log10(1/data_mse)
    sparseDCS_psnr = 10 * np.log10(1 / sparseDCS_mse)
    DCS_psnr = 10 * np.log10(1 / DCS_mse)
    print('sparseDCS_PSNR of measuremnt %d and sparsity %d = %f' % (meas,spr,sparseDCS_psnr))
    print('DCS_PSNR  of measuremnt %d and sparsity %d= %f\n' % (meas,spr,DCS_psnr))
    # SSIM
    sparseDCS_ssim = ssim(data_np_reshape, reconstructions_Sparse_tst_np_reshape,
                          data_range=reconstructions_Sparse_tst_np_reshape.max()
                                     - reconstructions_Sparse_tst_np_reshape.min())

    DCS_ssim = ssim(data_np_reshape, reconstructions_tst_np_reshape,
                    data_range=reconstructions_tst_np_reshape.max()
                               - reconstructions_tst_np_reshape.min())

    print('SparseDCS_SSIM of measuremnt %d and sparsity %d = %f' % (meas,spr,sparseDCS_ssim))
    print('DCS_SSIM of measuremnt %d and sparsity %d = %f\n' % (meas,spr,DCS_ssim))
    # Mean Ssuare Error
    sparseDCS_mse = np.linalg.norm(data_np_reshape - reconstructions_Sparse_tst_np_reshape)
    DCS_mse = np.linalg.norm(data_np_reshape - reconstructions_tst_np_reshape)
    print('SparseDCS_MSE of measuremnt %d and sparsity %d = %f' % (meas,spr,sparseDCS_mse))
    print('DCS_MSE of measuremnt %d and sparsity %d = %f\n' % (meas,spr,DCS_mse))

    print ('*******************************************************\n')
    fig, ax = plt.subplots()
    ax_hnd1, = ax.plot(20*np.log10(np.reshape(sparseDCS_reconstloss,-1)))
    ax_hnd2, = ax.plot(20*np.log10(np.reshape(DCS_reconstloss,-1)))
    ax.legend((ax_hnd1, ax_hnd2), ('ProxDCS Reconstruction error', 'DCS Reconstruction error'), loc='upper right', shadow=True)
    fig_title = ('Reconstruction error (latent space with sparsity = %d and meas = %d) ' %(spr,meas))
    ax.set_title(fig_title)
    ax.set_ylabel('Reconstruction error (dB)')
    ax.set_xlabel('epochs')
    # ax.grid(which='major', alpha=0.5)
    # plt.grid(color='b', linestyle='-', linewidth=1,which='minor', alpha=0.2)
    plt.grid(True)
    save_title = ('R_error_m_%dspr_%d' % (meas,spr))
    plt.savefig(os.path.join('DCS_DN', save_title + '.png'),
        bbox_inches='tight')
    # plt.show()