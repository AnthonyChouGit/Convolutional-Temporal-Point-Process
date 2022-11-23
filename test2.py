from tester import Tester
# import math
# import numpy as np
# import matplotlib.pyplot as plt

data_dir = '/data/zwt-datasets/data_retweet'
device = 'cpu'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 128
max_len = 256
config = {
        'model': 'conv-tpp',
        'embed_dim': 64,
        'hidden_dim': 64,
        'num_channel':2,
        'horizon':[2583*1, 2583*3],
        'num_component':64
    }
tester = Tester(config, data_dir, device, 'conv-retweet-2583*(1,3)', max_len, max_epoch=600, batch_size=batch_size, eval_batch_size=eval_batch_size,
                pred_batch_size=pred_batch_size, lr_step=40, init_lr=0.5e-3, display_step=40, patience=600)
tester.loadModel()
tester.testNll()