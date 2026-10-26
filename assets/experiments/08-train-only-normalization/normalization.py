"""Training-source state statistics and immutable CPU standardization.

No windows enter fitting. Every (trajectory_id,state_index) is counted once.
Only qpos/qvel are transformed; each feature retains its own physical unit.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from source_utils import MODEL_KEYS, file_sha256, numeric_fingerprint, read_numeric, source_path, validate_split_assignment

ALGORITHM = 'train-unique-states-ddof0-eps-v1'
EPSILON = 1e-6
FIELDS = ('qpos', 'qvel')
JOINTS = tuple([f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2'])
UNITS = {'qpos': ['rad']*7 + ['m']*2, 'qvel': ['rad/s']*7 + ['m/s']*2}
STAT_NAMES = ('mean', 'std', 'min', 'max', 'epsilon', 'scale', 'protected_mask', 'exact_constant_mask', 'near_constant_mask', 'count')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def context(input_dir):
    root = Path(input_dir)
    manifest = read_json(root / 'dataset_manifest.json')
    splits = read_json(root / 'splits.json')
    validate_split_assignment(manifest, splits)
    schema = read_json(root / 'schema.json')
    for field in FIELDS:
        if schema['fields']['observations/' + field]['units'] != UNITS[field]:
            raise ValueError('Source field units/order do not match the verified panda schema.')
    return manifest, splits


def compute_statistics(states):
    """Float64 per-feature population statistics; useful for small numeric tests."""
    output = {}
    if set(states) != set(FIELDS):
        raise ValueError('Exactly qpos and qvel are required.')
    for field in FIELDS:
        x = np.array(states[field], dtype=np.float64, copy=True)
        if x.ndim != 2 or x.shape[1] != 9 or x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError('Expected nonempty finite (states,9): ' + field)
        mean = x.mean(axis=0, dtype=np.float64)
        std = x.std(axis=0, dtype=np.float64, ddof=0)
        lo, hi = x.min(axis=0), x.max(axis=0)
        epsilon = np.full(9, EPSILON, dtype=np.float64)
        protected = std <= epsilon
        exact = lo == hi  # Identical stored values, rather than rounded std == 0.
        values = {'mean': mean, 'std': std, 'min': lo, 'max': hi, 'epsilon': epsilon,
                  'scale': np.maximum(std, epsilon), 'protected_mask': protected,
                  'exact_constant_mask': exact, 'near_constant_mask': protected & ~exact,
                  'count': np.array(x.shape[0], dtype=np.int64)}
        output.update({f'{field}/{name}': value for name, value in values.items()})
    if output['qpos/count'] != output['qvel/count']:
        raise ValueError('State field counts differ.')
    return output


def fit_training_statistics(input_dir):
    """Reads numerical values only from splits['train']; validation is excluded."""
    root = Path(input_dir)
    manifest, splits = context(root)
    records = {row['trajectory_id']: row for row in manifest['trajectories']}
    chunks = {field: [] for field in FIELDS}
    sources = []
    for label in splits['train']:
        row = records[label]
        path = source_path(root / 'trajectories', row['file'])
        if file_sha256(path) != row['file_sha256']:
            raise ValueError('Training source file SHA mismatch: ' + label)
        arrays = read_numeric(path)
        if (numeric_fingerprint(arrays) != row['numeric_content_fingerprint'] or
                numeric_fingerprint(arrays, MODEL_KEYS) != row['model_content_fingerprint']):
            raise ValueError('Training numerical fingerprint mismatch: ' + label)
        n = arrays['actions'].shape[0]
        if n != row['N'] or row['observations'] != n + 1:
            raise ValueError('Source N/N+1 mismatch: ' + label)
        for field in FIELDS:
            value = arrays['obs/agent/' + field]
            if value.shape != (n+1, 1, 9) or value.dtype != np.float32:
                raise ValueError('Unexpected source shape/dtype: ' + field)
            chunks[field].append(value[:, 0, :].astype(np.float64, copy=True))
        sources.append({key: row[key] for key in ('trajectory_id', 'source_id', 'file', 'file_sha256',
                                                'numeric_content_fingerprint', 'model_content_fingerprint', 'N')})
        sources[-1].update(state_indices=list(range(n+1)), count=n+1)
    values = compute_statistics({field: np.concatenate(chunks[field], axis=0) for field in FIELDS})
    info = {'algorithm': ALGORITHM, 'fit_split': 'train', 'environment_index': 0,
            'train_trajectory_ids': list(splits['train']), 'sources': sources,
            'sample_count': sum(row['count'] for row in sources),
            'state_identity': '(trajectory_id,state_index), including initial/final state; no value deduplication',
            'field_order': list(JOINTS), 'fields': list(FIELDS), 'raw_units': UNITS,
            'statistics_dtype': 'float64', 'output_dtype': 'float32', 'ddof': 0,
            'epsilon_each_raw_unit': EPSILON,
            'formula': 'scale=max(std,epsilon); z=(x-mean)/scale',
            'mask_definition': 'protected:std<=epsilon; exact_constant:min==max; near_constant:protected and not exact',
            'validation_used_for_fitting': False, 'windows_used_for_fitting': False,
            'future_visibility_note': 'Offline training records; not a claim that future states are visible at a window start.'}
    return values, info


def save_statistics(directory, values, info):
    directory = Path(directory)
    if any((directory / name).exists() for name in ('train_stats.npz', 'train_stats.json')):
        raise FileExistsError('Frozen statistics already exist; choose a new output directory.')
    np.savez_compressed(directory / 'train_stats.npz', **values)
    result = copy.deepcopy(info)
    result['npz_sha256'] = file_sha256(directory / 'train_stats.npz')
    result['statistics'] = {key: value.tolist() for key, value in values.items()}
    (directory / 'train_stats.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class FrozenNormalizer:
    """Bytes-backed statistics cannot be made writable. Public access returns copies."""
    def __init__(self, values, info):
        expected = {f'{field}/{name}' for field in FIELDS for name in STAT_NAMES}
        if set(values) != expected:
            raise ValueError('Statistics fields do not match the algorithm schema.')
        if (info['algorithm'] != ALGORITHM or info['field_order'] != list(JOINTS) or
                info['fields'] != list(FIELDS) or info['raw_units'] != UNITS or info['ddof'] != 0 or
                info['epsilon_each_raw_unit'] != EPSILON or info['fit_split'] != 'train' or
                info['validation_used_for_fitting'] or info['windows_used_for_fitting']):
            raise ValueError('Wrong algorithm, field order, units or fitting source.')
        self._info = copy.deepcopy(info)
        self._values = {}
        for key, value in values.items():
            field, name = key.split('/')
            expected_dtype = np.int64 if name == 'count' else np.bool_ if name.endswith('_mask') else np.float64
            expected_shape = () if name == 'count' else (9,)
            if value.shape != expected_shape or value.dtype != expected_dtype or not np.isfinite(value).all():
                raise ValueError('Wrong statistic shape/dtype/finite value: ' + key)
            self._values[key] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        for field in FIELDS:
            v = {name: self._values[f'{field}/{name}'] for name in STAT_NAMES}
            if (not np.all(v['std'] >= 0) or not np.all(v['epsilon'] == EPSILON) or
                    not np.array_equal(v['scale'], np.maximum(v['std'], v['epsilon'])) or
                    not np.array_equal(v['protected_mask'], v['std'] <= v['epsilon']) or
                    not np.array_equal(v['exact_constant_mask'], v['min'] == v['max']) or
                    not np.array_equal(v['near_constant_mask'], v['protected_mask'] & ~v['exact_constant_mask']) or
                    not np.all(v['min'] <= v['mean']) or not np.all(v['mean'] <= v['max']) or
                    int(v['count']) != info['sample_count'] or int(v['count']) <= 0):
                raise ValueError('Inconsistent statistic values: ' + field)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        info = read_json(directory / 'train_stats.json')
        if file_sha256(directory / 'train_stats.npz') != info['npz_sha256']:
            raise ValueError('Frozen NPZ SHA mismatch.')
        values = read_numeric(directory / 'train_stats.npz')
        if {key: value.tolist() for key, value in values.items()} != info['statistics']:
            raise ValueError('JSON and numeric statistics differ.')
        return cls(values, info)

    @property
    def arrays(self):
        return {key: value.copy() for key, value in self._values.items()}

    @property
    def metadata(self):
        return copy.deepcopy(self._info)

    def verify_context(self, manifest, splits):
        validate_split_assignment(manifest, splits)
        if self._info['train_trajectory_ids'] != splits['train']:
            raise ValueError('Statistics were fitted from different training sources.')
        records = {row['trajectory_id']: row for row in manifest['trajectories']}
        if [row['trajectory_id'] for row in self._info['sources']] != splits['train']:
            raise ValueError('Statistics provenance source list mismatch.')
        count = 0
        for source in self._info['sources']:
            row = records[source['trajectory_id']]
            for name in ('source_id', 'file', 'file_sha256', 'numeric_content_fingerprint', 'model_content_fingerprint', 'N'):
                if source[name] != row[name]:
                    raise ValueError('Statistics source fingerprint/identity mismatch: ' + name)
            if source['state_indices'] != list(range(row['N']+1)) or source['count'] != row['N']+1:
                raise ValueError('Statistics state index/count mismatch.')
            count += source['count']
        if count != self._info['sample_count']:
            raise ValueError('Statistics total state count mismatch.')

    @staticmethod
    def _input(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu':
                raise ValueError('Only CPU tensors are accepted.')
            value = value.detach().numpy()
        value = np.array(value, dtype=np.float64, copy=True)
        if value.ndim < 1 or value.shape[-1] != 9 or not np.isfinite(value).all():
            raise ValueError('Expected finite values with final feature dimension9.')
        return value

    def transform(self, field, value):
        if field not in FIELDS:
            raise ValueError('Only qpos and qvel may be standardized.')
        x = self._input(value)
        z = (x - self._values[field+'/mean']) / self._values[field+'/scale']
        if not np.isfinite(z).all():
            raise ValueError('Standardized values are not finite.')
        with np.errstate(over='ignore', invalid='ignore'):
            encoded = z.astype(np.float32, copy=True)
        if not np.isfinite(encoded).all():
            raise ValueError('Float32 standardized output is not finite.')
        return torch.from_numpy(encoded)

    def inverse(self, field, value):
        if field not in FIELDS:
            raise ValueError('Only qpos and qvel may be inverted.')
        z = self._input(value)
        return z * self._values[field+'/scale'] + self._values[field+'/mean']
