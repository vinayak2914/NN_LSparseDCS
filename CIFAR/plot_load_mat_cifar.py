from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
# dict_mat_10_100 = loadmat('mat_10_100_784/gen_n_10_100_784.mat')
dict_mat_20_100 = loadmat('saved_var_20_100_1024.mat')
# dict_mat_25_100 = loadmat('mat_25_100_784/gen_n_25_100_784.mat')

# test_mat_10_100 = loadmat('mat_10_100_784/test_10_100_784.mat')
# test_mat_20_100 = loadmat('mat_20_100_784/test_20_100_784.mat')
# test_mat_25_100 = loadmat('mat_25_100_784/test_25_100_784.mat')


# Measurement settings y = 10, sparsity = 100
#
# sparseDCS_reconstloss_10_100 = dict_mat_10_100.get('sparseDCS_reconstloss')
# DCS_reconstloss_10_100 = dict_mat_10_100.get('DCS_reconstloss')
# DCS_valid_reconstloss_10_100 = dict_mat_10_100.get('DCS_valid_reconstloss')
# sparseDCS_valid_reconstloss_10_100 = dict_mat_10_100.get('sparseDCS_valid_reconstloss')
#
# data_np_reshape_10_100 = test_mat_10_100.get('data_np_reshape')
# DCS_test_reconstloss_10_100 = test_mat_10_100.get('DCS_test_reconstloss')
# sparseDCS_test_reconstloss_10_100 = test_mat_10_100.get('sparseDCS_test_reconstloss')
# reconstructions_tst_np_reshape_10_100 = test_mat_10_100.get('reconstructions_tst_np_reshape')
# reconstructions_Sparse_tst_np_reshape_10_100 = test_mat_10_100.get('reconstructions_Sparse_tst_np_reshape')
#
# # data_mse = np.mean(np.square(data_np_reshape_10_100 - data_np_reshape_10_100))
# sparseDCS_mse = np.mean(np.square(data_np_reshape_10_100 - reconstructions_Sparse_tst_np_reshape_10_100))
# DCS_mse = np.mean(np.square(data_np_reshape_10_100 - reconstructions_tst_np_reshape_10_100))
# # data_psnr = 10*np.log10(1/data_mse)
# sparseDCS_psnr = 10*np.log10(1/sparseDCS_mse)
# DCS_psnr = 10*np.log10(1/DCS_mse)
# print ('sparseDCS_PSNR = %f\n' %(sparseDCS_psnr))
# print ('DCS_PSNR = %f\n' %(DCS_psnr))
# # SSIM
# sparseDCS_ssim = ssim(data_np_reshape_10_100, reconstructions_Sparse_tst_np_reshape_10_100,
#                  data_range=reconstructions_Sparse_tst_np_reshape_10_100.max() - reconstructions_Sparse_tst_np_reshape_10_100.min())
#
# DCS_ssim = ssim(data_np_reshape_10_100, reconstructions_tst_np_reshape_10_100,
#                   data_range=reconstructions_tst_np_reshape_10_100.max() - reconstructions_tst_np_reshape_10_100.min())
#
# print ('SparseDCS_SSIM = %f\n' %(sparseDCS_ssim))
# print ('DCS_SSIM = %f\n' %(DCS_ssim))
# # Mean Ssuare Error
# sparseDCS_mse = np.linalg.norm(data_np_reshape_10_100 - reconstructions_Sparse_tst_np_reshape_10_100)
# DCS_mse = np.linalg.norm(data_np_reshape_10_100 - reconstructions_tst_np_reshape_10_100)
# print ('SparseDCS_MSE = %f\n' %(sparseDCS_mse))
# print ('DCS_MSE = %f\n' %(DCS_mse))


# Measurement settings y = 20, sparsity = 100

sparseDCS_reconstloss_20_100 = dict_mat_20_100.get('sparseDCS_reconstloss')
DCS_reconstloss_20_100 = dict_mat_20_100.get('DCS_reconstloss')
DCS_valid_reconstloss_20_100 = dict_mat_20_100.get('DCS_valid_reconstloss')
sparseDCS_valid_reconstloss_20_100 = dict_mat_20_100.get('sparseDCS_valid_reconstloss')

# data_np_reshape_20_100 = test_mat_20_100.get('data_np_reshape')
# DCS_test_reconstloss_20_100 = test_mat_20_100.get('DCS_test_reconstloss')
# sparseDCS_test_reconstloss_20_100 = test_mat_20_100.get('sparseDCS_test_reconstloss')
# reconstructions_tst_np_reshape_20_100 = test_mat_20_100.get('reconstructions_tst_np_reshape')
# reconstructions_Sparse_tst_np_reshape_20_100 = test_mat_20_100.get('reconstructions_Sparse_tst_np_reshape')

# # data_mse = np.mean(np.square(data_np_reshape_20_100 - data_np_reshape_20_100))
# sparseDCS_mse = np.mean(np.square(data_np_reshape_20_100 - reconstructions_Sparse_tst_np_reshape_20_100))
# DCS_mse = np.mean(np.square(data_np_reshape_20_100 - reconstructions_tst_np_reshape_20_100))
# # data_psnr = 10*np.log10(1/data_mse)
# sparseDCS_psnr = 10*np.log10(1/sparseDCS_mse)
# DCS_psnr = 10*np.log10(1/DCS_mse)
# print ('sparseDCS_PSNR = %f\n' %(sparseDCS_psnr))
# print ('DCS_PSNR = %f\n' %(DCS_psnr))
# # SSIM
# sparseDCS_ssim = ssim(data_np_reshape_20_100, reconstructions_Sparse_tst_np_reshape_20_100,
#                  data_range=reconstructions_Sparse_tst_np_reshape_20_100.max() - reconstructions_Sparse_tst_np_reshape_20_100.min())
#
# DCS_ssim = ssim(data_np_reshape_20_100, reconstructions_tst_np_reshape_20_100,
#                   data_range=reconstructions_tst_np_reshape_20_100.max() - reconstructions_tst_np_reshape_20_100.min())
#
# print ('SparseDCS_SSIM = %f\n' %(sparseDCS_ssim))
# print ('DCS_SSIM = %f\n' %(DCS_ssim))
# # Mean Ssuare Error
# sparseDCS_mse = np.linalg.norm(data_np_reshape_20_100 - reconstructions_Sparse_tst_np_reshape_20_100)
# DCS_mse = np.linalg.norm(data_np_reshape_20_100 - reconstructions_tst_np_reshape_20_100)
# print ('SparseDCS_MSE = %f\n' %(sparseDCS_mse))
# print ('DCS_MSE = %f\n' %(DCS_mse))
#
# # Measurement settings y = 25, sparsity = 100
#
# sparseDCS_reconstloss_25_100 = dict_mat_25_100.get('sparseDCS_reconstloss')
# DCS_reconstloss_25_100 = dict_mat_25_100.get('DCS_reconstloss')
# DCS_valid_reconstloss_25_100 = dict_mat_25_100.get('DCS_valid_reconstloss')
# sparseDCS_valid_reconstloss_25_100 = dict_mat_25_100.get('sparseDCS_valid_reconstloss')
#
# data_np_reshape_25_100 = test_mat_25_100.get('data_np_reshape')
# DCS_test_reconstloss_25_100 = test_mat_25_100.get('DCS_test_reconstloss')
# sparseDCS_test_reconstloss_25_100 = test_mat_25_100.get('sparseDCS_test_reconstloss')
# reconstructions_tst_np_reshape_25_100 = test_mat_25_100.get('reconstructions_tst_np_reshape')
# reconstructions_Sparse_tst_np_reshape_25_100 = test_mat_25_100.get('reconstructions_Sparse_tst_np_reshape')
#
# # data_mse = np.mean(np.square(data_np_reshape_25_100 - data_np_reshape_25_100))
# sparseDCS_mse = np.mean(np.square(data_np_reshape_25_100 - reconstructions_Sparse_tst_np_reshape_25_100))
# DCS_mse = np.mean(np.square(data_np_reshape_25_100 - reconstructions_tst_np_reshape_25_100))
# # data_psnr = 10*np.log10(1/data_mse)
# sparseDCS_psnr = 10*np.log10(1/sparseDCS_mse)
# DCS_psnr = 10*np.log10(1/DCS_mse)
# print ('sparseDCS_PSNR = %f\n' %(sparseDCS_psnr))
# print ('DCS_PSNR = %f\n' %(DCS_psnr))
# # SSIM
# sparseDCS_ssim = ssim(data_np_reshape_25_100, reconstructions_Sparse_tst_np_reshape_25_100,
#                  data_range=reconstructions_Sparse_tst_np_reshape_25_100.max() - reconstructions_Sparse_tst_np_reshape_25_100.min())
#
# DCS_ssim = ssim(data_np_reshape_25_100, reconstructions_tst_np_reshape_25_100,
#                   data_range=reconstructions_tst_np_reshape_25_100.max() - reconstructions_tst_np_reshape_25_100.min())
#
# print ('SparseDCS_SSIM = %f\n' %(sparseDCS_ssim))
# print ('DCS_SSIM = %f\n' %(DCS_ssim))
# # Mean Ssuare Error
# sparseDCS_mse = np.linalg.norm(data_np_reshape_25_100 - reconstructions_Sparse_tst_np_reshape_25_100)
# DCS_mse = np.linalg.norm(data_np_reshape_25_100 - reconstructions_tst_np_reshape_25_100)
# print ('SparseDCS_MSE = %f\n' %(sparseDCS_mse))
# print ('DCS_MSE = %f\n' %(DCS_mse))


fig, ax = plt.subplots()
ax_hnd1, = ax.plot(20*np.log10(np.reshape(sparseDCS_reconstloss_20_100,-1)))
ax_hnd2, = ax.plot(20*np.log10(np.reshape(DCS_reconstloss_20_100,-1)))
ax.legend((ax_hnd1, ax_hnd2), ('ProxDCS Reconstruction error', 'DCS Reconstruction error'), loc='upper right', shadow=True)
fig_title = ('Reconstruction error (latent space with sparsity = %d and dim = %d) ' %(100,1024))
ax.set_title(fig_title)
ax.set_ylabel('Reconstruction error (dB)')
ax.set_xlabel('epochs')
# ax.grid(which='major', alpha=0.5)
# plt.grid(color='b', linestyle='-', linewidth=1,which='minor', alpha=0.2)
plt.grid(True)
plt.show()



# fig1, axes = plt.subplots(nrows=1, ncols=3,
#                          sharex=True, sharey=True)
# ax = axes.ravel()
#
# label = 'PNSR: {:.2f}, MSE: {:.2f}, SSIM: {:.2f}'
#
# ax[0].imshow(data_np_reshape, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[0].set_xlabel(label.format(data_psnr,0, 1))
# ax[0].set_title('Original image')
#
# ax[1].imshow(reconstructions_tst_np_reshape, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[1].set_xlabel(label.format(DCS_psnr,DCS_mse, DCS_ssim))
# ax[1].set_title('Image with noise')
#
# ax[2].imshow(reconstructions_Sparse_tst_np_reshape, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[2].set_xlabel(label.format(sparseDCS_psnr,sparseDCS_mse, sparseDCS_ssim))
# ax[2].set_title('Image plus constant')
#
# plt.tight_layout()
# plt.show()
#
# aa = 1