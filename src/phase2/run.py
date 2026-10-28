"""CLI for paired development-only experiments and controls."""
import argparse
import copy
import importlib.metadata
import json
import pickle
import subprocess
import time
import resource
import sys
import shutil
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from src.data.utils import sha256_file
from .data import load_development, Representation
from .models import Model
from .training import train, seed_all
from .evaluation import metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(config, output, smoke=False):
    started_run = time.perf_counter()
    if (not config['experiments'] or not set(config['experiments']) <= {'A', 'B'}
            or len(set(config['experiments'])) != len(config['experiments'])
            or config['epochs'] < 1 or config['batch_size'] < 1
            or not config['seeds'] or len(set(config['seeds'])) != len(config['seeds'])
            or not config['folds'] or not set(config['folds']) <= set(range(5))
            or config['intermediate_components'] < 4):
        raise ValueError('Invalid experimental configuration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    if smoke:
        config.update(folds=[0], seeds=[1729], epochs=2)
    data_config, root, protocol, expression, labels = load_development(config['data_config'])
    config['preprocessing'] = data_config['preprocessing']
    config['architecture'] = dict(qubits=4, layers=2, rotations=['RY','RZ'],
        entanglement='0->1->2->3', measurement='PauliZ(3)', encoding='RY once',
        angular='pi*tanh(z/2)', probability='clip((1+expectation)/2,1e-7,1-1e-7)',
        encoder_hidden=8, representation_seed=1729, dtype='float64', device='default.qubit',
        shots=None, differentiation='backprop', loss='unweighted BCE', threshold=0.5,
        optimizer='AdamW', selection='fixed final epoch', classical_C=1.0)
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'pennylane', 'numpy', 'pandas', 'scikit-learn', 'pyarrow']}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    diff = subprocess.run(['git', 'diff', 'HEAD'], capture_output=True, text=True).stdout
    (output / 'working_tree.patch').write_text(diff)
    write_json(output / 'config.json', config)
    snapshot = output / 'source'
    snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
    write_json(output / 'provenance.json', dict(versions=versions, python=platform.python_version(),
        environment={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        source_sha256={p.name: sha256_file(p) for p in snapshot.glob('*.py')}, commit=git.stdout.strip(),
        stage='smoke' if smoke else 'development_cv', locked_test_evaluated=False,
        input_sha256={str(p.relative_to(root)): sha256_file(p) for p in [
            root / data_config['phase1c']['inner_cv']['artifact'],
            root / data_config['paths']['processed_dir'] / 'expression.parquet',
            root / data_config['paths']['processed_dir'] / 'labels.parquet']}))
    rows = []
    for fold in protocol['folds']:
        if fold['fold'] not in config['folds']:
            continue
        tr, va = fold['train_case_ids'], fold['validation_case_ids']
        y, v = labels.loc[tr].to_numpy(), labels.loc[va].to_numpy()
        for experiment in config['experiments']:
            components = 4 if experiment == 'A' else config['intermediate_components']
            started = time.perf_counter()
            representation = Representation(data_config['preprocessing'], components).fit(expression.loc[tr])
            x, xv = representation.transform(expression.loc[tr]), representation.transform(expression.loc[va])
            preprocessing_seconds = time.perf_counter() - started
            folder = output / f'fold_{fold["fold"]}_{experiment}'
            folder.mkdir()
            with (folder / 'representation.pkl').open('wb') as stream:
                pickle.dump(representation, stream)
            write_json(folder / 'representation.json', dict(train_ids=tr, validation_ids=va,
                explained_variance_ratio=representation.pca.explained_variance_ratio_.tolist(),
                preprocessing_seconds=preprocessing_seconds))
            xt, vt = torch.tensor(x), torch.tensor(xv)
            yt, vyt = torch.tensor(y, dtype=torch.float64), torch.tensor(v, dtype=torch.float64)
            for seed in config['seeds']:
                def save(name, probability, seconds, model=None, curves=None, estimator=None):
                    target = folder / f'{name}_seed_{seed}'
                    target.mkdir()
                    result = dict(experiment=name, fold=fold['fold'], seed=seed,
                                  **metrics(v, probability), seconds=seconds,
                                  preprocessing_seconds=preprocessing_seconds)
                    if model is not None:
                        result['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        result['head_parameters'] = sum(p.numel() for p in model.head.parameters())
                        result['qnode_calls'] = getattr(model.head, 'calls', 0)
                        result['circuit_sample_evaluations'] = getattr(model.head, 'samples', 0)
                        torch.save(model.state_dict(), target / 'model.pt')
                    if estimator is not None:
                        with (target / 'model.pkl').open('wb') as stream:
                            pickle.dump(estimator, stream)
                    if curves is not None:
                        pd.DataFrame(curves).to_csv(target / 'curves.csv', index=False)
                    pd.DataFrame(dict(case_id=va, er_label=v, probability_er_positive=probability)).to_csv(
                        target / 'validation_predictions.csv', index=False)
                    write_json(target / 'metrics.json', result)
                    rows.append(result)
                    print(name, fold['fold'], seed, result['roc_auc'], flush=True)
                for quantum in [True, False]:
                    seed_all(seed)
                    if experiment == 'A' and not quantum:
                        start = time.perf_counter()
                        lr = LogisticRegression(C=1., max_iter=2000, random_state=seed).fit(x, y)
                        save('A_classical', lr.predict_proba(xv)[:, 1], time.perf_counter()-start, estimator=lr)
                        continue
                    model = Model(components, encoder=experiment == 'B', quantum=quantum)
                    curves, seconds = train(model, xt, yt, vt, vyt, config)
                    with torch.no_grad():
                        probability = model(vt).numpy()
                    save(experiment if quantum else 'B_classical', probability, seconds, model, curves)
                    if experiment == 'B' and quantum:
                        # Encoder has never fitted validation. Head C is prespecified, no tuning.
                        model.eval()
                        with torch.no_grad():
                            z, zv = model.encoder(xt).numpy(), model.encoder(vt).numpy()
                        start = time.perf_counter()
                        probe = LogisticRegression(C=1., max_iter=2000, random_state=seed).fit(z, y)
                        save('B_frozen_probe', probe.predict_proba(zv)[:, 1],
                             time.perf_counter()-start, estimator=probe)
                        write_json(folder / f'B_frozen_probe_seed_{seed}' / 'encoder_reference.json',
                                   {'state_dict': f'../B_seed_{seed}/model.pt', 'representation': '../representation.pkl'})
    if not rows:
        raise ValueError('No experiments selected')
    table = pd.DataFrame(rows)
    table.to_csv(output / 'comparison.csv', index=False)
    write_json(output / 'resources.json', dict(wall_seconds=time.perf_counter()-started_run,
        process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_unit='bytes' if sys.platform == 'darwin' else 'KiB',
        device='CPU', torch_threads=torch.get_num_threads()))
    return table


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='configs/phase2_ab.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.output, args.smoke)


if __name__ == '__main__':
    main()
