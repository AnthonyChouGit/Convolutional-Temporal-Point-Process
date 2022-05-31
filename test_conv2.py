from tester import Tester
import math

data_dir = '/data/zwt-datasets/data_so'
device = 'cuda:1'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 128
max_len = 256


config = {
    'embed_dim': 64,
    'hidden_dim': 64,
    'model': 'conv-tpp',
    'num_channel':3,
    'horizon': [4, 8, 16, 32],
    'num_component': 64,
}
tester = Tester(config, data_dir, device, 'temp', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=1, patience=100)
tester.train()
tester.testNll()
# tester.testPred()

# config = {
#     'embed_dim': 64,
#     'hidden_dim': 64,
#     'model': 'conv-tpp',
#     'num_channel':3,
#     'horizon': [4, 16, 32],
#     'num_component': 64,
# }
# tester = Tester(config, data_dir, device, 'conv-mooc-41632', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
#         pred_batch_size=pred_batch_size, lr_step=40, init_lr=5e-4, display_step=40, patience=100)
# tester.train()
# tester.testNll()
