from data_utils.data import prepare_dataloader
from models.model import TPP
import torch
import math
from sklearn.metrics import f1_score, mean_squared_error


if __name__ == '__main__':
    with torch.no_grad():
        data_dir = '/data/zwt-datasets/data_so'
        device = 'cuda:1'
        batch_size = 32
        eval_batch_size = 128
        embed_dim = 64
        model_name = 'so1'
        hidden_dim = 64
        train_loader, eval_train_loader, dev_loader, test_loader, num_types = prepare_dataloader(data_dir, batch_size,
                                                                        eval_batch_size, 256)# TODO: here
        # mean_log_inter, std_log_inter = get_dtime_stats(eval_train_loader)
        config = {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_types': num_types,
            'model': 'conv-tpp',
            'num_channel':3,
            'horizon': [math.inf, math.inf, math.inf],
            'num_component': 64
        }
        model = TPP(config).to(device)
        model.load_state_dict(torch.load(f'state_dicts/{model_name}'))
        model.eval()
        types_true = list()
        types_pred = list()
        dtimes_true = list()
        dtimes_pred = list()
        for ind, batch in enumerate(test_loader):
                event_time, _, event_type = batch
                for i in range(batch_size):
                    time_seq = event_time[i]
                    type_seq = event_type[i]
                    event_num = type_seq.ne(0).sum()
                    type_seq = type_seq[:event_num]
                    time_seq = time_seq[:event_num]
                    time_in = time_seq[:-1]
                    type_in = type_seq[:-1]
                    types_true.append(type_seq[-1])
                    dtimes_true.append(time_seq[-1]-time_seq[-2])
                    assert type_seq[-1]!=0
                    type_pred, dtime_pred = model.predict(type_in, time_in)
                    types_pred.append(type_pred)
                    dtimes_pred.append(dtime_pred)
                    print(type_pred, type_seq[-1].item(), dtime_pred, (time_seq[-1]-time_seq[-2]).item())
        print(f1_score(types_true, types_pred, average='micro'))
        print(mean_squared_error(dtimes_true, dtimes_pred, squared=False))

