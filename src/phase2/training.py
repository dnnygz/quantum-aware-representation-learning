"""Fixed budget, final epoch evaluation; no validation-driven selection."""
import random
import time
import numpy as np
import torch
from torch import nn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def train(model, x, y, validation, target, config):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.BCELoss()
    curves = []
    started = time.perf_counter()
    for epoch in range(config['epochs']):
        model.train()
        order = torch.randperm(len(x))
        total = 0.
        norms = {'encoder': [], 'head': []}
        for indices in order.split(config['batch_size']):
            optimizer.zero_grad()
            loss = loss_fn(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            for name in norms:
                grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in grads):
                    raise ValueError('Missing/nonfinite gradient')
                norms[name].append(sum(float(g.square().sum()) for g in grads) ** .5)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite parameters')
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(validation), target))
        if not np.isfinite(val_loss):
            raise ValueError('Nonfinite validation loss')
        curves.append(dict(epoch=epoch+1, train_loss=total/len(x), validation_loss=val_loss,
                           **{f'{k}_gradient_norm': float(np.mean(v)) for k,v in norms.items()}))
    return curves, time.perf_counter() - started
