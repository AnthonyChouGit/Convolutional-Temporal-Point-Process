from tester import Tester

data_dir = '/data/zwt-datasets/data_so'
device = 'cuda:1'
batch_size = 64
eval_batch_size = 128
pred_batch_size = 128
max_len = 256

for i in [1, 3, 5, 10]:

    config = {
        'model': 'conv-tpp',
        'embed_dim': 64,
        'hidden_dim': 64,
        'num_channel':3,
        'horizon':[0.83*i],
        'num_component':64,
        'omega': 30
    }

    tester = Tester(config, data_dir, device, f'conv-so-083*{i}-3c-omega20)', max_len, max_epoch=400, batch_size=batch_size, eval_batch_size=eval_batch_size,
            pred_batch_size=pred_batch_size, lr_step=40, init_lr=1e-3, display_step=40, patience=600)
    tester.train()
    # tester.loadModel()
    tester.testNll()
    tester.plot(f'conv-so-083*{i}-3c-omega20')