from tester import Tester
import math

data_dir = '/data/zwt-datasets/data_so'
device = 'cuda:0'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 1
max_len = 256

config = {
    'model': 'rmtpp',
    'embed_dim': 64,
    'hidden_dim': 64,
    'mlp_dim': 128,
    'max_t': 8.,
    'n_samples_pred': 100.
}

tester = Tester(config, data_dir, device, 'rmtpp-1', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3)
tester.train()
tester.testNll()
tester.testPred()

config = {
    'embed_dim': 64,
    'hidden_dim': 64,
    'model': 'nhp',
    'max_t': 8.,
    'n_samples_pred': 100.
}
tester = Tester(config, data_dir, device, 'nhp-1', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3)
tester.train()
tester.testNll()
tester.testPred()

config = {
    'embed_dim': 64,
    'hidden_dim': 64,
    'model': 'log-norm-mix',
    'num_component': 64,
    'max_t': 8.,
    'n_samples_pred': 100.
}
tester = Tester(config, data_dir, device, 'lognorm-1', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=30, init_lr=5e-4)
tester.train()
tester.testNll()
tester.testPred()

config = {
    'embed_dim': 64,
    'hidden_dim': 64,
    'model': 'conv-tpp',
    'num_channel':3,
    'horizon': [[16, 32, math.inf], [16, 32, math.inf], [16, 32, math.inf]],
    'num_component': 64,
    'max_t': 8.,
    'n_samples_pred': 100.
}
tester = Tester(config, data_dir, device, 'lognorm-1', max_len, max_epoch=1000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=30, init_lr=5e-4)
tester.train()
tester.testNll()
tester.testPred()