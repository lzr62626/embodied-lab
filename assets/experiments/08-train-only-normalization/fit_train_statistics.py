"""Fit once from training-source original states. This never opens a simulator."""
import argparse
from pathlib import Path
from normalization import fit_training_statistics, save_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent / 'input')
    parser.add_argument('--output-dir', type=Path, default=Path('fitted-statistics'))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new directory; existing statistics are never overwritten.')
    values, info = fit_training_statistics(args.input_dir)
    args.output_dir.mkdir(parents=True)
    save_statistics(args.output_dir, values, info)
    print(f"Fitted {info['sample_count']} unique training-state records; validation excluded.")


if __name__ == '__main__':
    main()
