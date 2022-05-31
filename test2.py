from tester import Tester
import math

data_dir = '/data/zwt-datasets/data_so'
device = 'cuda:1'
batch_size = 32
eval_batch_size = 128
pred_batch_size = 128
max_len = 256

config = {
    'model': 'hp'
}
tester = Tester(config, data_dir, device, 'hp-so', max_len, max_epoch=2000, batch_size=batch_size, eval_batch_size=eval_batch_size,
        pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-2, display_step=40, patience=100)
# tester.train()
tester.loadModel()
tester.testNll()
