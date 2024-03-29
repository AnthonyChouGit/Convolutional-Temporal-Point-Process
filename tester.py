from statistics import mean
from data_utils.data import prepare_dataloader
from models.model import TPP
from eval_utils.eval import evalNll, evalPred
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from tqdm import tqdm
import math
import os
import pandas as pd

class Tester:
    def __init__(self, config, data_dir, device, model_name, max_len=None, max_epoch=400, 
    batch_size=32, eval_batch_size=128, pred_batch_size=64, init_lr=5e-4, patience=70, threshold=1e-4, lr_step=30,
     step_gamma=0.5, reload_step=10, display_step=30) -> None:
        self.data_dir = data_dir
        self.device = device
        self.model_name=model_name
        self.max_len=max_len
        self.batch_size=batch_size
        self.eval_batch_size = eval_batch_size
        self.init_lr = init_lr
        self.patience=patience
        self.lr_step=lr_step
        self.step_gamma=step_gamma
        self.reload_step=reload_step
        self.display_step=display_step
        self.max_epoch = max_epoch
        self.threshold = threshold
        self.config = config

        self.train_data, self.eval_train_data,  self.dev_data, self.test_data, self.pred_loader, self.num_types = prepare_dataloader(data_dir, batch_size, eval_batch_size, pred_batch_size, max_len)
        config['num_types'] = self.num_types
        self.model = TPP(config).to(device)

    def train(self):
        optimizer = Adam(self.model.parameters(), lr=self.init_lr)
        scheduler = StepLR(optimizer, self.lr_step, gamma=self.step_gamma)
        now = datetime.now().strftime("%D %H:%M:%S")
        with open(f'{self.model_name}.txt', 'a') as f:
            f.write(f'{now}\n')
            f.write(f'{self.config}\n')
        best_loss = math.inf
        impatient = 0
        for epoch in tqdm(range(self.max_epoch)):
            for ind, batch in enumerate(self.train_data):
                event_time, _, event_type = batch
                optimizer.zero_grad()
                loss, _, _ = self.model.compute_loss(event_type, event_time)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.)
                optimizer.step()
            scheduler.step()
            with torch.no_grad():
                dev_loss, _, _ = evalNll(self.model, self.dev_data)
                if (best_loss - dev_loss) < self.threshold: 
                    impatient += 1
                    if dev_loss < best_loss:
                        best_loss = dev_loss
                        torch.save(self.model.state_dict(), f'state_dicts/{self.model_name}')
                else:
                    best_loss = dev_loss
                    torch.save(self.model.state_dict(), f'state_dicts/{self.model_name}')
                    impatient = 0
                if (epoch+1)%self.display_step==0:
                    with open(f'{self.model_name}.txt', 'a') as f:
                        f.write(f'epoch {epoch+1} finished, best_loss={best_loss}.\n')
                    print(f'epoch {epoch+1} finished, best_loss={best_loss}.')
            if impatient >= self.patience:
                break
            if (epoch+1)%self.reload_step==0:
                self.loadModel()
                # self.model.load_state_dict(torch.load(f'state_dicts/{self.model_name}'))
        self.loadModel()
        # torch
        # torch.save(self.model.state_dict(), f'state_dicts/{self.model_name}')
    
    def plot(self,dir_name='default', width=1):
        self.model.plot(dir_name, width)
    
    def loadModel(self):
        self.model.load_state_dict(torch.load(f'state_dicts/{self.model_name}', map_location=self.device))

    def modelExists(self):
        return os.path.exists(f'state_dicts/{self.model_name}')

    @property
    def time_stats(self):
        with torch.no_grad():
            all_dt = torch.empty(0, device=self.device)
            for d in [self.eval_train_data, self.dev_data, self.test_data]:
                for batch in d:
                    time, _, et = batch
                    et = et.to(self.device)
                    time = time.to(self.device)
                    mask = et.ne(0)
                    batch_size = time.shape[0]
                    dt = time[:, 1:] - time[:, :-1]
                    dt = dt.masked_select(mask[:, 1:])
                    all_dt = torch.cat([all_dt, dt])
            dt_mean = all_dt.mean()
            dt_max = all_dt.max()
            dt_min = all_dt.min()
        return dt_mean, dt_max, dt_min

    @property
    def type_stats(self):
        with torch.no_grad():
            all_type = list()
            for d in [self.eval_train_data, self.dev_data, self.test_data]:
                for batch in d:
                    _, _, et = batch
                    et = et.to(self.device)
                    mask = et.ne(0)
                    et = et.masked_select(mask)
                    all_type.extend(list(et.cpu().numpy()))
            all_type = pd.Series(all_type)
            return all_type.value_counts()

    @property
    def data_stats(self):
        with torch.no_grad():
            train_seq_num = 0
            seq_lens = list()
            for batch in self.eval_train_data:
                _, _, event_type = batch
                train_seq_num += event_type.shape[0]
                batch_seq_lens = event_type.ne(0).sum(1).tolist()
                seq_lens.extend(batch_seq_lens)
            dev_seq_num = 0
            for batch in self.dev_data:
                _, _, event_type = batch
                dev_seq_num += event_type.shape[0]
                batch_seq_lens = event_type.ne(0).sum(1).tolist()
                seq_lens.extend(batch_seq_lens)
            test_seq_num = 0
            for batch in self.dev_data:
                _, _, event_type = batch
                test_seq_num += event_type.shape[0]
                batch_seq_lens = event_type.ne(0).sum(1).tolist()
                seq_lens.extend(batch_seq_lens)
        return self.num_types, min(seq_lens), mean(seq_lens), max(seq_lens), \
                train_seq_num, dev_seq_num, test_seq_num

    def testNll(self):
        self.loadModel()
        with torch.no_grad():
            test_loss, type_loss, dt_loss = evalNll(self.model, self.test_data)
        print(f'All done. Test_loss={test_loss}, type_loss={type_loss}, dt_loss={dt_loss}.')
        with open(f'{self.model_name}.txt', 'a') as f:
            f.write(f'All done. Test_loss={test_loss}, type_loss={type_loss}, dt_loss={dt_loss}.\n')
        # TODO: test NLL and prediction error

    def testPred(self):
        self.loadModel()
        with torch.no_grad():
            type_acc, dt_error = evalPred(self.model, self.pred_loader)
        print(f'All done. ACC={type_acc}, RMSE={dt_error}.')
        with open(f'{self.model_name}.txt', 'a') as f:
            f.write(f'All done. ACC={type_acc}, RMSE={dt_error}.')
