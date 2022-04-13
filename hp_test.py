from data_utils.data import prepare_dataloader
from models.model import TPP
from eval_utils.eval import evaluate
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import math
from tqdm import tqdm

if __name__ == '__main__':
    batch_size = 32
    eval_batch_size = 128
    num_epochs = 400
    lr = 1e-3
    device = 'cuda:1'
    model_name = 'so1-hp'
    patience = 85
    display_step = 1
    lr_step = 40
    step_gamma = 0.5
    reload_step = 10
    data_dir = '/data/zwt-datasets/data_so'
    train_loader, _, dev_loader, test_loader, num_types = prepare_dataloader(data_dir, batch_size,
                                                                     eval_batch_size, 256)# TODO: here
    config = {
        'num_types': num_types,
        'model': 'hp'
    }
    model = TPP(config).to(device)

    # model.load_state_dict(torch.load(f'state_dicts/{model_name}'))
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, lr_step, gamma=step_gamma)
    now = datetime.now().strftime("%D %H:%M:%S")
    with open(f'{model_name}.txt', 'a') as f:
        f.write(f'{now}\n')
        f.write(f'{config}\n')
    best_loss = math.inf
    impatient = 0
    for epoch in tqdm(range(num_epochs)):
        for ind, batch in enumerate(train_loader):
            event_time, _, event_type = batch
            optimizer.zero_grad()
            loss = model.compute_loss(event_type, event_time)
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            dev_loss = evaluate(model, dev_loader)
            if (best_loss - dev_loss) < 1e-4: 
                impatient += 1
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    torch.save(model.state_dict(), f'state_dicts/{model_name}')
            else:
                best_loss = dev_loss
                torch.save(model.state_dict(), f'state_dicts/{model_name}')
                impatient = 0
            if (epoch+1)%display_step==0:
                with open(f'{model_name}.txt', 'a') as f:
                    f.write(f'epoch {epoch+1} finished, best_loss={best_loss}.\n')
                print(f'epoch {epoch+1} finished, best_loss={best_loss}.')
        if impatient >= patience:
            break
        if (epoch+1)%reload_step==0:
            model.load_state_dict(torch.load(f'state_dicts/{model_name}'))

    model.load_state_dict(torch.load(f'state_dicts/{model_name}'))
    with torch.no_grad():
        test_loss = evaluate(model, test_loader)
    print(f'All done. Test_loss={test_loss}.')
    with open(f'{model_name}.txt', 'a') as f:
        f.write(f'All done. Test_loss={test_loss}.\n')

    torch.save(model.state_dict(), f'state_dicts/{model_name}')

