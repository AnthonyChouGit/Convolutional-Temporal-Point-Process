from tester import Tester
import math

data_dir = '/data/zwt-datasets/data_lastfm'
device = 'cuda:1'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 128
max_len = 256

config = {
    'embed_dim': 64,
    'hidden_dim': 64,
    'model': 'conv-tpp-pred',
    'num_channel':3,
    'horizon': [0.05, 0.1, 0.2],
    'num_component': 64,
    'gamma':0,
    'beta':1.
}
tester = Tester(config, data_dir, device, 'conv-lastfm-pred-type', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=1, patience=85)
tester.train()
# tester.loadModel()
tester.testPred()


# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'log-norm-mix-pred',
#     'num_component': 64,
#     'gamma':0,
#     'beta':1.
# }
# tester = Tester(config, data_dir, device, 'lognorm-lastfm-pred-type', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=1, patience=85)
# tester.train()
# # tester.loadModel()
# tester.testPred()