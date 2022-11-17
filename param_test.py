from tester import Tester
# import math
# import numpy as np
# import matplotlib.pyplot as plt

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
    'num_channel':1,
    'horizon':[2583*10],
    'num_component':64
}

tester = Tester(config, data_dir, device, 'conv-retweet-2583*10', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=0.5e-3, display_step=40)
tester.train()
tester.testNll()

# config = {
#     'model': 'conv-tpp',
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'num_channel':2,
#     'horizon':[0.76*10],
#     'num_component':64
# }

# tester = Tester(config, data_dir, device, 'conv-lastfm-0.76*10-2c', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3, display_step=40)
# tester.train()
# tester.testNll()

# config = {
#     'model': 'conv-tpp',
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'num_channel':3,
#     'horizon':[0.76*10],
#     'num_component':64
# }


# config = {
#     'model': 'conv-tpp',
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'num_channel':1,
#     'horizon':[0.76*3],
#     'num_component':64
# }

# tester = Tester(config, data_dir, device, 'conv-lastfm-0.76*3', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3, display_step=40)
# tester.train()
# tester.testNll()
