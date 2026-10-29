"""Fixed-budget E0 head fitting; frozen saved embeddings are the only inputs."""
import hashlib
import time
import numpy as np
import torch
from .models import VQC
from .e0_data import initial_weights


CLOSED = dict(protocol_id='E0-v1-approved-closed',folds=list(range(5)),
    encoder_seeds=[1729,1730,1731],head_seeds=[1729,1730,1731],epochs=50,batch_size=32,
    lr=.01,weight_decay=.001,initialization_std=.1,shuffle_seed='encoder_seed',
    primary_e01=['train_log_loss_refit_minus_historical_B','validation_log_loss_refit_minus_historical_B'],
    primary_e02=['validation_roc_auc_LR_angle_minus_LR_z','validation_log_loss_LR_angle_minus_LR_z'],
    diagonal_comparison='prespecified_h_equals_s_subset',selection='fixed_final_epoch_all_initializations',
    phase_closed=True)


def validate_config(config):
    if config!=CLOSED:
        raise ValueError('E0 is closed: configuration changes require an independent approved protocol')


def new_head(seed):
    with torch.random.fork_rng(devices=[]):
        head=VQC()
    with torch.no_grad(): head.weights.copy_(initial_weights(seed))
    return head


def fit_head(z,y,validation,target,encoder_seed,head_seed,config):
    """No encoder exists in this function; optimizer membership is exactly the VQC head."""
    x=torch.tensor(np.array(z,copy=True),dtype=torch.float64)
    yt=torch.tensor(np.array(y,copy=True),dtype=torch.float64)
    xv=torch.tensor(np.array(validation,copy=True),dtype=torch.float64)
    yv=torch.tensor(np.array(target,copy=True),dtype=torch.float64)
    if x.requires_grad or xv.requires_grad: raise ValueError('Embeddings must be constant inputs')
    head=new_head(head_seed)
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(head.weights)}:
        raise ValueError('Optimizer contains unexpected parameters')
    generator=torch.Generator(device='cpu').manual_seed(encoder_seed)
    loss_fn=torch.nn.BCELoss();curves=[];shuffle=hashlib.sha256()
    fit_seconds=0.;validation_seconds=0.;steps=0
    for epoch in range(config['epochs']):
        start=time.perf_counter();head.train()
        order=torch.randperm(len(x),generator=generator)
        shuffle.update(order.numpy().tobytes())
        total=0.;norms=[];squares=[]
        for indices in order.split(config['batch_size']):
            optimizer.zero_grad();loss=loss_fn(head(x[indices]),yt[indices])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            if head.weights.grad is None or not torch.isfinite(head.weights.grad).all():
                raise ValueError('Missing or nonfinite head gradient')
            norms.append(float(head.weights.grad.norm()))
            squares.append(head.weights.grad.detach().square().clone())
            optimizer.step();steps+=1
            if not torch.isfinite(head.weights).all(): raise ValueError('Nonfinite head weights')
            total+=float(loss.detach())*len(indices)
        rms=torch.stack(squares).mean(dim=0).sqrt()
        gradient={f'gradient_l{l}_q{q}_{gate}':float(rms[l,q,g])
                  for l in range(2) for q in range(4) for g,gate in enumerate(['RY','RZ'])}
        fit_seconds+=time.perf_counter()-start
        start=time.perf_counter();head.eval()
        with torch.no_grad(): value=float(loss_fn(head(xv),yv))
        validation_seconds+=time.perf_counter()-start
        if not np.isfinite(value): raise ValueError('Nonfinite validation loss')
        curves.append(dict(epoch=epoch+1,train_loss=total/len(x),validation_loss=value,
            head_gradient_norm=float(np.mean(norms)),optimizer_steps=steps,**gradient))
    timing=dict(fit_update_seconds=fit_seconds,epoch_validation_seconds=validation_seconds,
        training_qnode_calls=head.calls,training_sample_state_evaluations=head.samples,
        optimizer_steps=steps,shuffle_sha256=shuffle.hexdigest(),trainable_parameters=16)
    start=time.perf_counter()
    with torch.no_grad(): predictions=dict(train=head(x).numpy(),validation=head(xv).numpy())
    timing.update(final_prediction_seconds=time.perf_counter()-start,
        total_qnode_calls=head.calls,total_sample_state_evaluations=head.samples)
    return head,initial,predictions,curves,timing
