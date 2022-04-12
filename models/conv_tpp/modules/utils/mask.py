import torch

def get_subsequent_mask(length):
    return torch.triu(
        torch.ones(length, length, dtype=torch.uint8), diagonal=1
    ).bool()
