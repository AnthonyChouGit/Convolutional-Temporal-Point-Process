import torch

def get_dtime_stats(dataloader):
    dtime_list = list()
    for batch in dataloader:
        event_time, _, event_type = batch
        event_time = event_time.cuda()
        event_type = event_type.cuda()
        batch_size = event_time.shape[0]
        event_time = event_time - event_time[:, 0].view(-1, 1)
        event_time = torch.cat([torch.zeros(batch_size, 1, device='cuda'), event_time], dim=1)
        dtime = event_time[:, 1:] - event_time[:, :-1]
        mask = event_type.ne(0)
        valid_dtime = dtime.masked_select(mask).tolist()
        dtime_list.extend(valid_dtime)
    log_dtimes = (torch.tensor(dtime_list, device='cuda').clamp(1e-10)).log()
    return log_dtimes.mean().item(), log_dtimes.std().item()
