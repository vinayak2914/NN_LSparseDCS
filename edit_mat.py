import scipy
from scipy.io import loadmat


# test_dict = loadmat('mat_files/combine_mat/test_10_100_784.mat')
gen_dict = loadmat('mat_files/combine_mat/gen_n_10_100_784.mat')
save_dict = loadmat('mat_files/combine_mat/saved_var_10_100_784.mat')
mat_file = 'mat_files/combine_mat/gen_n_10_100_784.mat'

sparseDCS_reconstloss = gen_dict.get('sparseDCS_reconstloss')
DCS_reconstloss = gen_dict.get('DCS_reconstloss')
DCS_valid_reconstloss = gen_dict.get('DCS_valid_reconstloss')
sparseDCS_valid_reconstloss = gen_dict.get('sparseDCS_valid_reconstloss')
data_np_reshape = gen_dict.get('data_np_reshape')
DCS_test_reconstloss = gen_dict.get('DCS_test_reconstloss')
sparseDCS_test_reconstloss = gen_dict.get('sparseDCS_test_reconstloss')
reconstructions_tst_np_reshape = gen_dict.get('reconstructions_tst_np_reshape')
reconstructions_Sparse_tst_np_reshape = gen_dict.get('reconstructions_Sparse_tst_np_reshape')


sparseDCS_reconstloss_1 = save_dict.get('sparseDCS_reconstloss')
DCS_reconstloss_1 = save_dict.get('DCS_reconstloss')
DCS_valid_reconstloss_1 = save_dict.get('DCS_valid_reconstloss')
sparseDCS_valid_reconstloss_1 = save_dict.get('sparseDCS_valid_reconstloss')
sparseDCS_reconstloss[0,0:170] = sparseDCS_reconstloss_1[0,0:170]
DCS_reconstloss[0,0:170] = DCS_reconstloss_1[0,0:170]
DCS_valid_reconstloss[0,0:170] = DCS_valid_reconstloss_1[0,0:170]
sparseDCS_valid_reconstloss[0,0:170] = sparseDCS_valid_reconstloss_1[0,0:170]


scipy.io.savemat(mat_file, mdict={'DCS_test_reconstloss': DCS_test_reconstloss,
                                              'sparseDCS_test_reconstloss': sparseDCS_test_reconstloss,
                                              'sparseDCS_reconstloss': sparseDCS_reconstloss,
                                              'DCS_reconstloss': DCS_reconstloss,
                                              'DCS_valid_reconstloss': DCS_valid_reconstloss,
                                              'sparseDCS_valid_reconstloss': sparseDCS_valid_reconstloss,
                                              'data_np_reshape': data_np_reshape,
                                              'reconstructions_tst_np_reshape': reconstructions_tst_np_reshape,
                                              'reconstructions_Sparse_tst_np_reshape': reconstructions_Sparse_tst_np_reshape})