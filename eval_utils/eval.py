def evaluate(model, dataloader):
    model.eval()
    total_event_num = 0
    total_loss = 0.
    for batch in dataloader:
        event_time, _, event_type = batch
        event_num = event_type.ne(0).sum().item()
        loss = model.compute_loss(event_type, event_time)
        total_event_num += event_num
        total_loss += loss
    model.train()
    return total_loss / total_event_num
