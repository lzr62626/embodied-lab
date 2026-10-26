"""Fit from train once, freeze, then export CPU batches and real state statistics."""
import argparse
from collections import Counter
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_trajectory_dataset import MultiTrajectoryDataset
from normalization import FIELDS, JOINTS, UNITS, FrozenNormalizer, context, fit_training_statistics, save_statistics
from normalized_window_dataset import NormalizedWindowDataset
from source_utils import file_sha256, read_numeric, source_path

GENERATED = ('train_stats.npz', 'train_stats.json', 'train_batch.npz', 'validation_batch.npz',
             'batch_summary.json', 'normalization_report.json', 'schema.json', 'environment.json', 'requirements.txt')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def numeric_batch(tree, prefix=''):
    arrays, strings = {}, {}
    for key, value in tree.items():
        name = prefix + key
        if isinstance(value, dict):
            a, s = numeric_batch(value, name + '/')
            arrays.update(a); strings.update(s)
        elif isinstance(value, torch.Tensor) and value.device.type == 'cpu':
            arrays[name] = value.detach().numpy().copy()
        elif name == 'metadata/trajectory_id' and isinstance(value, (list, tuple)):
            strings[name] = list(value)
        else:
            raise TypeError('Unsupported batch value: ' + name)
    return arrays, strings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent / 'input')
    parser.add_argument('--output-dir', type=Path, default=Path('result'))
    args = parser.parse_args()
    out = args.output_dir
    if any((out / name).exists() for name in GENERATED):
        raise FileExistsError('Existing output is preserved; choose a new output directory.')
    manifest, splits = context(args.input_dir)
    values, info = fit_training_statistics(args.input_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_statistics(out, values, info)
    frozen_sha = {name: file_sha256(out / name) for name in ('train_stats.npz', 'train_stats.json')}
    normalizer = FrozenNormalizer.load(out)
    # Only now, after statistics are frozen, load validation values and windows.
    source_arrays = {row['trajectory_id']: read_numeric(source_path(args.input_dir / 'trajectories', row['file']))
                     for row in manifest['trajectories']}
    summary = {'configuration': {'batch_size': 4, 'shuffle': False, 'drop_last': False,
                                'num_workers': 0, 'pin_memory': False, 'device': 'cpu', 'generator_seed': 42},
               'statistics_fitted_once': True, 'statistics_sha256': frozen_sha, 'splits': {}}
    for split in ('train', 'validation'):
        base = MultiTrajectoryDataset(args.input_dir / 'dataset_manifest.json', args.input_dir / 'splits.json',
                                      args.input_dir / 'trajectories', split)
        dataset = NormalizedWindowDataset(base, normalizer)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0,
                            pin_memory=False, generator=torch.Generator(device='cpu').manual_seed(42))
        batches = []; first_shapes = None
        for index, batch in enumerate(loader):
            numeric, labels = numeric_batch(batch)
            ids = numeric['metadata/source_id']
            state_indices = numeric['metadata/state_indices']
            for field in FIELDS:
                numeric['raw_observations/' + field] = np.stack([
                    source_arrays[base.names[int(source)]]['obs/agent/' + field][states, 0, :].copy()
                    for source, states in zip(ids, state_indices)])
            shapes = {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in numeric.items()}
            batches.append({'size': len(ids), 'global_window_indices': numeric['metadata/global_window_index'].tolist(),
                            'source_ids': ids.tolist(), 'trajectory_ids': labels['metadata/trajectory_id'],
                            'shapes': shapes})
            if index == 0:
                np.savez_compressed(out / f'{split}_batch.npz', **numeric)
                first_shapes = shapes
        summary['splits'][split] = {'windows': len(dataset), 'batch_sizes': [b['size'] for b in batches],
                                    'first_batch': first_shapes, 'batches': batches,
                                    'last_batch_shapes': batches[-1]['shapes']}
    write_json(out / 'batch_summary.json', summary)
    report = {'fit_state_count': info['sample_count'], 'fitting_sources': splits['train'],
              'statistics_frozen_before_validation': True, 'feature_rows': [],
              'unique_state_counts': {}, 'window_state_occurrences': {},
              'inverse_tolerance': {'rtol': 1e-6, 'atol_each_raw_unit': 1e-7},
              'normalized_mean_std_definition': 'Computed over unique original states, not repeated window appearances.',
              'limitations': 'Three short fragments; no model training, independence proof or large-dataset distribution claim.'}
    transformed = {}
    for split in ('train', 'validation'):
        pooled = {field: np.concatenate([source_arrays[label]['obs/agent/' + field][:, 0, :] for label in splits[split]])
                  for field in FIELDS}
        report['unique_state_counts'][split] = len(pooled['qpos'])
        report['window_state_occurrences'][split] = summary['splits'][split]['windows'] * (manifest['K']+1)
        transformed[split] = {}
        for field, x in pooled.items():
            z32 = normalizer.transform(field, x).numpy()
            inverse_error = np.abs(normalizer.inverse(field, z32) - x.astype(np.float64))
            transformed[split][field] = {'mean': z32.astype(np.float64).mean(axis=0),
                                        'std': z32.astype(np.float64).std(axis=0, ddof=0),
                                        'inverse_max_abs': inverse_error.max(axis=0),
                                        'min': z32.min(axis=0), 'max': z32.max(axis=0)}
    for field in FIELDS:
        for j, joint in enumerate(JOINTS):
            row = {'field': field, 'index': j, 'joint': joint, 'raw_unit': UNITS[field][j], 'count': info['sample_count']}
            for name in ('mean', 'std', 'min', 'max', 'epsilon', 'scale', 'protected_mask', 'exact_constant_mask', 'near_constant_mask'):
                row[name] = values[f'{field}/{name}'][j].item()
            row['train_transformed'] = {name: vector[j].item() for name, vector in transformed['train'][field].items()}
            row['validation_transformed'] = {name: vector[j].item() for name, vector in transformed['validation'][field].items()}
            report['feature_rows'].append(row)
    write_json(out / 'normalization_report.json', report)
    source_schema = json.loads((args.input_dir / 'schema.json').read_text(encoding='utf-8'))
    schema = source_schema.copy()
    schema.update(format='article8-frozen-train-standardized-windows-v1', statistics='train_stats.json + train_stats.npz',
                  fitting_state_count=info['sample_count'], raw_field_order=list(JOINTS), raw_units=UNITS)
    for field in FIELDS:
        schema['fields']['observations/' + field] = {'shape': [4, 5, 9], 'dtype': 'float32',
            'axes': ['B', 'K+1', 'J'], 'units': 'dimensionless per-feature standard score',
            'transform': '(raw-mean)/max(std,epsilon)', 'statistics': f'{field}/* in train_stats.npz'}
        schema['fields']['raw_observations/' + field] = {'shape': [4, 5, 9], 'dtype': 'float32',
            'axes': ['B', 'K+1', 'J'], 'units': UNITS[field],
            'role': 'Export-only audit copy; absent from Dataset model observations.'}
    schema['observation_subset'] = ['standardized qpos', 'standardized qvel', 'RGB float32 CHW /255']
    write_json(out / 'schema.json', schema)
    environment = {'python_version': platform.python_version(),
                   'packages': {name: metadata.version(name) for name in ('numpy', 'torch', 'pillow')},
                   'torch_runtime_version': torch.__version__, 'torch_cuda_build': torch.version.cuda,
                   'processing_device': 'cpu', 'simulator_created': False, 'reset_calls': 0, 'step_calls': 0,
                   'training': False, 'test_set': False, 'normalization_fit_split': 'train',
                   'loader': summary['configuration'], 'input_configuration': 'Unchanged article7 saved trajectories.',
                   'helper_sha256': {name: file_sha256(Path(__file__).parent / name) for name in
                       ('source_utils.py', 'trajectory_dataset.py', 'trajectory_window_dataset.py', 'multi_trajectory_dataset.py')}}
    write_json(out / 'environment.json', environment)
    (out / 'requirements.txt').write_text('\n'.join(f'{name}=={metadata.version(name)}' for name in ('numpy', 'torch', 'pillow'))+'\n', encoding='utf-8')
    for name in ('dataset_manifest.json', 'splits.json', 'window_index.json'):
        source, destination = args.input_dir / name, out / name
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise FileExistsError('Existing provenance differs: ' + name)
        else:
            shutil.copy2(source, destination)
    assert all(file_sha256(out / name) == value for name, value in frozen_sha.items())
    print(json.dumps({'result': 'EXPORTED', 'fit_states': info['sample_count'],
                      'train': summary['splits']['train']['batch_sizes'],
                      'validation': summary['splits']['validation']['batch_sizes'],
                      'protected': {field: values[field+'/protected_mask'].tolist() for field in FIELDS},
                      'note': 'Independent acceptance is separate; this is not a validation PASS.'}))


if __name__ == '__main__':
    main()
