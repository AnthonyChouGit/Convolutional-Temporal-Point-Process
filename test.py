from tester import Tester
import math
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/data/zwt-datasets/data_retweet'
device = 'cuda:1'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 128
max_len = 256

config = {
    'model': 'conv-tpp',
    'embed_dim': 64,
    'hidden_dim': 64,
    'mlp_dim': 128,
    'num_channel':2,
    'horizon':[1000, 2000],
    'num_component':64
}


tester = Tester(config, data_dir, device, 'conv-retweet', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3, display_step=40)
print(tester.data_stats)
print(tester.time_stats)
print(tester.type_stats)
# tester.loadModel()
# tester.train()
# tester.testNll()
# tester.testPred()

# type_stats = tester.type_stats
# key_list = type_stats.keys()
# types = np.sort(key_list)
# # print(type_stats[types].values)
# # plt.bar(types, type_stats[types].values)
# # plt.savefig('temp.jpg', bbox_inches='tight')
# fig, ax = plt.subplots()

# ax.bar(types, type_stats[types].values, color='green')

# ax.set(
#     xlim=(0, 23), 
# xticks=np.arange(1, 23), 
# # ylim=(0, 1200)
# )
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# # ax.pie(type_stats[types].values, labels=(1,2,3), autopct='%.2f')
# # plt.xticks(fontsize=20)
# # plt.yticks(fontsize=20)


# plt.savefig('temp.jpg', bbox_inches='tight')



# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'nhp',
#     'mat_t': 2,
#     'n_samples_pred':100
# }
# tester = Tester(config, data_dir, device, 'nhp-lastfm', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=1, lr_step=40, init_lr=1e-2, display_step=10, step_gamma=0.5)
# # tester.train()
# tester.loadModel()
# tester.testNll()
# tester.testPred()

# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'log-norm-mix',
#     'num_component': 64,
# }
# tester = Tester(config, data_dir, device, 'lognorm-retweet', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=40, patience=100)
# tester.loadModel()
# # tester.train()
# tester.testNll()
# tester.testPred()

# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'log-norm-mix',
#     'num_channel':3,
#     'horizon': [4, 16, 32],
#     'num_component': 64,
# }
# tester = Tester(config, data_dir, device, 'lognorm-retweet', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=40, patience=100)
# # print(tester.data_stats)
# tester.loadModel()
# # tester.model.model.plotKernel()
# # tester.train()
# tester.testNll()
# tester.testPred()

# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'conv-tpp',
#     'num_channel':3,
#     'horizon': [4, 16, 32],
#     'num_component': 64,
# }
# tester = Tester(config, data_dir, device, 'conv-1', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=40, patience=100)
# # tester.train()
# tester.loadModel()
# tester.testNll()
# tester.testPred()

# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'nhp',
#     'num_channel':3,
#     'horizon': [20, 30, 40],
#     'num_component': 64,
#     'max_t': 1,
#     'n_samples_pred':1
# }
# tester = Tester(config, data_dir, device, 'nhp-retweet', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=40, patience=100)
# # tester.train()
# tester.loadModel()
# tester.testNll()
# tester.testPred()
