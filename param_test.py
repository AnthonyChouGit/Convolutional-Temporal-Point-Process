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

# (1st, 2nd)
for i in [1, 3, 5, 10]:
    for j in [3]:
        horizon = [2583*i, 2583*j]

        config = {
            'model': 'conv-tpp',
            'embed_dim': 64,
            'hidden_dim': 64,
            'num_channel':2,
            'horizon':horizon,
            'num_component':64
        }

        tester = Tester(config, data_dir, device, f'conv-retweet-2583*({i},{j})', max_len, max_epoch=600, batch_size=batch_size, eval_batch_size=eval_batch_size,
                pred_batch_size=pred_batch_size, lr_step=40, init_lr=0.5e-3, display_step=40, patience=600)
        tester.train()
        tester.testNll()

