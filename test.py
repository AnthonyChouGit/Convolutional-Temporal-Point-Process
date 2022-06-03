from tester import Tester
import math

data_dir = '/data/zwt-datasets/data_lastfm'
device = 'cuda:0'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 128
max_len = 256

config = {
    'model': 'rmtpp',
    'embed_dim': 64,
    'hidden_dim': 64,
    'mlp_dim': 128,
}

tester = Tester(config, data_dir, device, 'rmtpp-lastfm', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3, display_step=40)
tester.loadModel()
# tester.train()
tester.testNll()
tester.testPred()

# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'nhp',
#     'mat_t': 10,
#     'n_samples_pred':200
# }
# tester = Tester(config, data_dir, device, 'nhp-so', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-2, display_step=10, step_gamma=0.5)
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
