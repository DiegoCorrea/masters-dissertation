# 1: User KNN
USERKNN_GRID_PARAMS = {"k": [10, 20, 30], "sim_options": {'name': ['pearson', 'cosine', 'msd'], 'user_based': [True]}}
# 2: Item KNN
ITEMKNN_GRID_PARAMS = {"k": [10, 20, 30], "sim_options": {'name': ['pearson', 'cosine', 'msd'], 'user_based': [False]}}
# 3: SVD
SVD_GRID_PARAMS = {"n_factors": [50, 100, 150], "n_epochs": [10, 30, 50], "lr_all": [0.001, 0.003, 0.005],
                   "reg_all": [0.01, 0.03, 0.05]}
# 4: SVD++
SVDpp_GRID_PARAMS = {"n_factors": [50, 100, 150], "n_epochs": [10, 30, 50], "lr_all": [0.001, 0.003, 0.005],
                     "reg_all": [0.01, 0.03, 0.05]}
# 5: NMF
NMF_GRID_PARAMS = {"n_factors": [50, 100], "n_epochs": [30, 50], "reg_pu": [0.003, 0.005],
                   "reg_qi": [0.03, 0.05], "reg_bu": [0.003, 0.005], "reg_bi": [0.003, 0.005],
                   "lr_bu": [0.003, 0.005], "lr_bi": [0.003, 0.005], "biased": [True]}
# 6: Co Clustering
CLUSTERING_GRID_PARAMS = {"n_cltr_u": [3, 5, 6], "n_cltr_i": [3, 5, 7], "n_epochs": [10, 30, 50]}
