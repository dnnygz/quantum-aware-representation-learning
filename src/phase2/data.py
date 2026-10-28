"""Strict partition gate and training-only shared transforms."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.data.preprocessing import _build_preprocessor
from src.data.utils import load_config


def validate_partitions(protocol):
    dev = protocol['development_case_ids']
    locked = protocol['locked_test_case_ids']
    if len(set(dev)) != len(dev) or len(set(locked)) != len(locked) or set(dev) & set(locked):
        raise ValueError('Invalid development/locked partition')
    seen = []
    for fold in protocol['folds']:
        tr, va = fold['train_case_ids'], fold['validation_case_ids']
        if (len(set(tr)) != len(tr) or len(set(va)) != len(va)
                or set(tr) & set(va) or set(tr + va) != set(dev)
                or set(tr + va) & set(locked)):
            raise ValueError('Invalid fold partition')
        seen.extend(va)
    if sorted(seen) != sorted(dev):
        raise ValueError('Validation folds must cover development exactly once')


def load_development(path='configs/data.yaml'):
    config, root = load_config(path)
    protocol = json.loads((root / config['phase1c']['inner_cv']['artifact']).read_text())
    validate_partitions(protocol)
    processed = root / config['paths']['processed_dir']
    # Read only development rows through Arrow, never materialize locked expression/labels.
    import pyarrow.dataset as ds
    def read_dev(name, field):
        dataset = ds.dataset(processed / name, format='parquet')
        return dataset.to_table(filter=ds.field(field).isin(protocol['development_case_ids'])).to_pandas()
    import pyarrow.parquet as pq
    schema = pq.read_schema(processed / 'expression.parquet')
    metadata = json.loads(schema.metadata[b'pandas'])
    index_field = metadata['index_columns'][0]
    expression = read_dev('expression.parquet', index_field)
    expression.index = expression.index.astype(str)
    labels = read_dev('labels.parquet', 'case_id').set_index('case_id')['er_label'].astype(int)
    if not set(labels.unique()) <= {0, 1}:
        raise ValueError('Expected ER+=1, ER-=0')
    return config, root, protocol, expression, labels


class Representation:
    def __init__(self, preprocessing, components, seed=1729):
        self.rna = _build_preprocessor(preprocessing).set_params(scale=True)
        self.pca = PCA(n_components=components, svd_solver='randomized', random_state=seed)
        self.scaler = StandardScaler()

    def fit(self, training):
        self.training_ids = list(training.index)
        x = self.rna.fit_transform(training)
        self.pca.fit(x)
        self.scaler.fit(self.pca.transform(x))
        return self

    def transform(self, x):
        values = self.scaler.transform(self.pca.transform(self.rna.transform(x)))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite representation')
        return values
