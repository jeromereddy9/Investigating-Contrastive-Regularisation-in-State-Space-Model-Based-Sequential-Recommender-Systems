import os
import sys
import csv
import json
import pickle
import traceback
from datetime import datetime
from src.utils import path_builder

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger

from src.models.Baselines.GRU4Rec import GRU4Rec
from src.models.Baselines.SASRec import SASRec
from src.models.Baselines.CL4SRec import CL4SRec
from src.models.Baselines.DouRec import DuoRec
from src.models.Baselines.mamba4rec import Mamba4Rec
from src.models.Baselines.gated_mamba import SIGMA
from src.models.SSM_CL.mamba4rec_cl import Mamba4Rec_CL
from src.models.SSM_CL.SIGMA_cl import SIGMA_CL

# ─────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────
DATASETS = [
    'amazon_toys_and_games',
    'amazon_videogames',
    'movielens_1m',
    'lastfm_1k',
]

EXPERIMENTS = [
    # Baselines
    (GRU4Rec,      'GRU4Rec',      'gru4rec',      None),
    (SASRec,       'SASRec',       'sasrec',        None),
    (CL4SRec,      'CL4SRec',      'cl4srec',       None),
    (DuoRec,       'DuoRec',       'duorec',        None),
    # Mamba4Rec variants
    (Mamba4Rec,    'Mamba4Rec',    'mamba4rec',     'CE'),
    (Mamba4Rec,    'Mamba4Rec',    'mamba4rec',     'BPR'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'CE'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'BPR'),
    # SIGMA variants
    (SIGMA,        'SIGMA',        'sigma',         'CE'),
    (SIGMA,        'SIGMA',        'sigma',         'BPR'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'CE'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'BPR'),
]

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
CONFIG_DIR        = path_builder('src/configs')
DATASET_CONFIG    = path_builder(CONFIG_DIR + '/dataset.yaml')
TRAINING_CONFIG   = path_builder(CONFIG_DIR + '/training.yaml')
MODELS_CONFIG_DIR = path_builder(CONFIG_DIR + '/models')
OUTPUT_DIR        = path_builder('src/results')
CHECKPOINT_DIR    = path_builder('src/checkpoints')
CSV_PATH          = path_builder(OUTPUT_DIR + '/training_summary.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CSV logging — training summary only
# ─────────────────────────────────────────────
CSV_HEADERS = [
    'timestamp', 'dataset', 'model', 'loss_type', 'exp_id',
    'best_valid_score',       # best NDCG@10 on validation
    'best_epoch',             # epoch at which best valid score was achieved
    'total_epochs_run',       # how many epochs before early stopping
    'early_stopped',          # whether early stopping triggered
    'status', 'error'
]

def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

def log_result(row: dict):
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)

# ─────────────────────────────────────────────
# Experiment ID
# ─────────────────────────────────────────────
def make_exp_id(model_name: str, loss_type: str | None, dataset: str) -> str:
    parts = [model_name]
    if loss_type:
        parts.append(loss_type)
    parts.append(dataset)
    return '_'.join(parts)

# ─────────────────────────────────────────────
# Callback trainer to capture per-epoch metrics
# ─────────────────────────────────────────────
class TrackingTrainer(Trainer):
    """
    Extends RecBole's Trainer to record per-epoch training loss
    and validation score for convergence analysis.
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        self.epoch_history = []  # list of dicts, one per epoch

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        train_loss = super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
        return train_loss

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        # Wrap to capture per-epoch info after each epoch
        result = super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        return result

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result = super()._valid_epoch(valid_data, show_progress)
        return valid_result


# ─────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────
def run_experiment(
    model_class,
    model_name: str,
    config_file: str,
    dataset_name: str,
    loss_type: str | None = None,
) -> dict:
    exp_id = make_exp_id(model_name, loss_type, dataset_name)
    print(f"\n{'='*60}")
    print(f"  {exp_id}")
    print(f"{'='*60}")

    row = {
        'timestamp':        datetime.now().isoformat(),
        'dataset':          dataset_name,
        'model':            model_name,
        'loss_type':        loss_type or 'default',
        'exp_id':           exp_id,
        'best_valid_score': None,
        'best_epoch':       None,
        'total_epochs_run': None,
        'early_stopped':    None,
        'status':           'failed',
        'error':            '',
    }

    try:
        # Build config
        config_dict = {}
        if loss_type:
            config_dict['loss_type'] = loss_type

        config = Config(
            model=model_class,
            dataset=dataset_name,
            config_file_list=[
                DATASET_CONFIG,
                TRAINING_CONFIG,
                path_builder(MODELS_CONFIG_DIR + f'/{config_file}.yaml'),
            ],
            config_dict=config_dict,
        )

        config['data_path'] = path_builder('src/datasets/preprocessed')

        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)
        logger = getLogger()

        # Data
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # Model
        model = model_class(config, dataset).to(config['device'])
        logger.info(model)

        # Train — RecBole's trainer logs train_loss and valid_score per epoch
        # internally in trainer.train_loss_dict and trainer.best_valid_score
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=True
        )

        # Extract per-epoch convergence data from RecBole's internal tracking
        train_loss_history = trainer.train_loss_dict  # {epoch: loss}
        best_epoch = trainer.best_valid_epoch if hasattr(trainer, 'best_valid_epoch') else None
        total_epochs = len(train_loss_history)
        early_stopped = total_epochs < config['epochs']

        # Per-epoch history for convergence plots
        epoch_history = []
        for epoch, loss in train_loss_history.items():
            epoch_history.append({
                'epoch': epoch,
                'train_loss': loss,
            })

        # Summary row
        row['best_valid_score'] = best_valid_score
        row['best_epoch']       = best_epoch
        row['total_epochs_run'] = total_epochs
        row['early_stopped']    = early_stopped
        row['status']           = 'success'

        # Save checkpoint — model + convergence history
        ckpt_path = path_builder(CHECKPOINT_DIR + f'/{exp_id}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'model_state_dict':  trainer.model.state_dict(),
                'config':            dict(config),
                'best_valid_score':  best_valid_score,
                'best_valid_result': best_valid_result,
                'best_epoch':        best_epoch,
                'total_epochs_run':  total_epochs,
                'early_stopped':     early_stopped,
                'epoch_history':     epoch_history,  # per-epoch train loss
                'exp_id':            exp_id,
            }, f)
        print(f"  Checkpoint saved → {ckpt_path}")
        print(f"  Best valid score : {best_valid_score:.4f} at epoch {best_epoch}")
        print(f"  Epochs run       : {total_epochs} (early stopped: {early_stopped})")

    except Exception as e:
        row['error'] = str(e)
        print(f"  FAILED: {e}")
        traceback.print_exc()

    log_result(row)
    return row

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    init_csv()
    print(f"\nTraining run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets            : {DATASETS}")
    print(f"Experiments/dataset : {len(EXPERIMENTS)}")
    print(f"Total experiments   : {len(DATASETS) * len(EXPERIMENTS)}")

    all_results = []
    failed = []

    for dataset in DATASETS:
        print(f"\n{'#'*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'#'*60}")

        for model_class, model_name, config_file, loss_type in EXPERIMENTS:
            result = run_experiment(
                model_class=model_class,
                model_name=model_name,
                config_file=config_file,
                dataset_name=dataset,
                loss_type=loss_type,
            )
            all_results.append(result)
            if result['status'] == 'failed':
                failed.append(result['exp_id'])

    # Summary
    print(f"\n{'='*60}")
    print(f"  Training complete : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total             : {len(all_results)}")
    print(f"  Passed            : {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"  Failed            : {len(failed)}")
    if failed:
        print(f"\n  Failed experiments:")
        for exp_id in failed:
            print(f"    - {exp_id}")
    print(f"\n  Summary CSV  → {CSV_PATH}")
    print(f"  Checkpoints  → {CHECKPOINT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()