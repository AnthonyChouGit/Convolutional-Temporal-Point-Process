from sklearn.metrics import accuracy_score, mean_squared_error

def evalNll(model, dataloader):
    model.eval()
    total_event_num = 0
    total_loss = 0.
    total_type_loss = 0.
    total_dt_loss = 0.
    for batch in dataloader:
        event_time, _, event_type = batch
        event_num = event_type[:, 1:].ne(0).sum().item()
        loss, type_loss, dt_loss = model.compute_loss(event_type, event_time)
        total_event_num += event_num
        total_loss += loss
        total_type_loss += type_loss
        total_dt_loss += dt_loss
    model.train()
    return total_loss / total_event_num, total_type_loss / total_event_num, total_dt_loss / total_event_num

def evalPred(model, dataloader):
    model.eval()
    type_true = list()
    type_pred = list()
    dtime_true = list()
    dtime_pred = list()
    for batch in dataloader:
        event_time, _, event_type = batch
        estimated_type, estimated_dt = model.predict(event_type[:, :-1], event_time[:, :-1])
        mask = event_type[:, 1:].ne(0)
        estimated_type = estimated_type.masked_select(mask.to(estimated_type.device)).tolist()
        estimated_dt = estimated_dt.masked_select(mask.to(estimated_dt.device)).tolist()
        real_type = event_type[:, 1:].masked_select(mask.to(event_type.device)).tolist()
        dtimes = event_time[:, 1:] - event_time[:, :-1]
        # print(dtimes.masked_select(mask.to(dtimes.device)).mean())
        real_dt = dtimes.masked_select(mask.to(dtimes.device)).tolist()
        type_true.extend(real_type)
        type_pred.extend(estimated_type)
        dtime_true.extend(real_dt)
        dtime_pred.extend(estimated_dt)
    type_acc = accuracy_score(type_true, type_pred)
    dt_error = mean_squared_error(dtime_true, dtime_pred, squared=False)
    model.train()
    return type_acc, dt_error
        