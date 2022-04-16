import torch
import torch.utils.data
import numpy as np
import pickle

class EventData(torch.utils.data.Dataset):
    def __init__(self, data, max_len=None) -> None:
        super().__init__()
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        self.event_type = [[elem['type_event']+1 for elem in inst] for inst in data]

        self.length = len(data)
        self.max_len = max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        time, time_gap, event_type =  self.time[idx], self.time_gap[idx], self.event_type[idx]
        if self.max_len is not None:
            time = time[:self.max_len]
            time_gap = time_gap[:self.max_len]
            event_type = event_type[:self.max_len]
        return time, time_gap, event_type

def pad_time(insts):
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [0] * (max_len - len(inst))
        for inst in insts
    ])

    return torch.tensor(batch_seq, dtype=torch.float32)

def pad_type(insts):
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [0] * (max_len - len(inst))
        for inst in insts
    ])

    return torch.tensor(batch_seq, dtype=torch.long)

def collate_fn(insts):
    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)

    # TODO: temp test code
    # time = time[:, :256]
    # event_type = event_type[:, :256]

    return time, time_gap, event_type

def get_dataloader(data, batch_size, shuffle=True, max_len=None):
    ds = EventData(data, max_len)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl

def prepare_dataloader(data_dir, train_batch_size, eval_batch_size, pred_batch_size, max_len=None):
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[dict_name]
        return data, int(num_types)
    
    train_data, num_types = load_data(f'{data_dir}/train.pkl', 'train')
    dev_data, _ = load_data(f'{data_dir}/dev.pkl', 'dev')
    test_data, _ = load_data(f'{data_dir}/test.pkl', 'test')

    train_loader = get_dataloader(train_data, train_batch_size, shuffle=True, max_len=max_len)
    train_eval_loader = get_dataloader(train_data, eval_batch_size, shuffle=False, max_len=max_len)
    dev_loader = get_dataloader(dev_data, eval_batch_size, shuffle=False, max_len=max_len)
    test_loader = get_dataloader(test_data, eval_batch_size, shuffle=False, max_len=max_len)
    pred_loader = get_dataloader(test_data, pred_batch_size, shuffle=False, max_len=max_len)
    return train_loader, train_eval_loader, dev_loader, test_loader, pred_loader, num_types