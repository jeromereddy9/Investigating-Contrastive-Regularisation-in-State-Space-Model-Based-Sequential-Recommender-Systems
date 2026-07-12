import os
import sys
import csv
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


DATASETS = [
    'amazon_musical_instruments',
    'amazon_videogames',
    'movielens_1m',
    'lastfm_1k',
]

# (model_class, model_name, config_file, loss_type or None)
EXPERIMENTS = [
    # Baselines
    (GRU4Rec,       'GRU4Rec',       'gru4rec',       None),
    (SASRec,        'SASRec',        'sasrec',        None),
    (CL4SRec,       'CL4SRec',       'cl4srec',       None),
    (DuoRec,        'DuoRec',        'duorec',        None),
    # Mamba4Rec variants
    (Mamba4Rec,     'Mamba4Rec',     'mamba4rec',     'CE'),
    (Mamba4Rec,     'Mamba4Rec',     'mamba4rec',     'BPR'),
    (Mamba4Rec_CL,  'Mamba4Rec_CL',  'mamba4rec_cl',  'CE'),
    (Mamba4Rec_CL,  'Mamba4Rec_CL',  'mamba4rec_cl',  'BPR'),
    # SIGMA variants
    (SIGMA,         'SIGMA',         'sigma',         'CE'),
    (SIGMA,         'SIGMA',         'sigma',         'BPR'),
    (SIGMA_CL,      'SIGMA_CL',      'sigma_cl',      'CE'),
    (SIGMA_CL,      'SIGMA_CL',      'sigma_cl',      'BPR'),
]


CONFIG_DIR        = path_builder('src/configs')
DATASET_CONFIG    = path_builder(CONFIG_DIR+'/dataset.yaml')
TRAINING_CONFIG   = path_builder(CONFIG_DIR+'/training.yaml')
MODELS_CONFIG_DIR = path_builder(CONFIG_DIR+'/models')
OUTPUT_DIR        = path_builder('src/results')
CHECKPOINT_DIR    = path_builder('src/checkpoints')
CSV_PATH          = path_builder(OUTPUT_DIR+'/results.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


CSV_HEADERS = [
    'timestamp', 'dataset', 'model', 'loss_type', 'exp_id',
    'hit@5', 'hit@10', 'hit@20',
    'ndcg@5', 'ndcg@10', 'ndcg@20',
    'mrr@5', 'mrr@10', 'mrr@20',
    'best_valid_score', 'status', 'error'
]

def init_csv():
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

def log_result(row: dict):
    """Append a result row to the CSV."""
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)


def make_exp_id(model_name: str, loss_type: str | None, dataset: str) -> str:
    parts = [model_name]
    if loss_type:
        parts.append(loss_type)
    parts.append(dataset)
    return '_'.join(parts)


def run_experiment(
    model_class,
    model_name: str,
    config_file: str,
    dataset_name: str,
    loss_type: str | None = None,
) -> dict:
    exp_id = make_exp_id(model_name, loss_type, dataset_name)
    print(f"  {exp_id}")

    row = {
        'timestamp':  datetime.now().isoformat(),
        'dataset':  dataset_name,
        'model':  model_name,
        'loss_type': loss_type or 'default',
        'exp_id':  exp_id,
        'hit@5': None,  'hit@10': None,  'hit@20': None,
        'ndcg@5': None, 'ndcg@10': None, 'ndcg@20': None,
        'mrr@5': None,  'mrr@10': None,  'mrr@20': None,
        'best_valid_score': None,
        'status': 'failed',
        'error':  '',
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

        # Point RecBole at our preprocessed data
        config['data_path'] = path_builder('src/datasets/preprocessed')

        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)
        logger = getLogger()

        # Data
        dataset   = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # Model
        model = model_class(config, dataset).to(config['device'])
        logger.info(model)

        # Train
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=True
        )

        # Evaluate
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)

        # Parse metrics
        row['best_valid_score'] = best_valid_score
        row['hit@5'] = test_result.get('hit@5')
        row['hit@10'] = test_result.get('hit@10')
        row['hit@20'] = test_result.get('hit@20')
        row['ndcg@5'] = test_result.get('ndcg@5')
        row['ndcg@10'] = test_result.get('ndcg@10')
        row['ndcg@20'] = test_result.get('ndcg@20')
        row['mrr@5'] = test_result.get('mrr@5')
        row['mrr@10'] = test_result.get('mrr@10')
        row['mrr@20'] = test_result.get('mrr@20')
        row['status'] = 'success'

        # Save model checkpoint
        ckpt_path = path_builder(CHECKPOINT_DIR + f'/{exp_id}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'model_state_dict': trainer.model.state_dict(),
                'config':           dict(config),
                'test_result':      test_result,
                'best_valid_score': best_valid_score,
                'best_valid_result': best_valid_result,
                'exp_id':           exp_id,
            }, f)
        print(f"  Checkpoint saved → {ckpt_path}")

    except Exception as e:
        row['error'] = str(e)
        print(f"  FAILED: {e}")
        traceback.print_exc()

    log_result(row)
    return row

def main():
    init_csv()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nExperiment run started: {timestamp}")
    print(f"Datasets  : {DATASETS}")
    print(f"Experiments per dataset: {len(EXPERIMENTS)}")
    print(f"Total experiments: {len(DATASETS) * len(EXPERIMENTS)}")

    all_results = []
    failed = []

    for dataset in DATASETS:
        print(f"  Dataset: {dataset}")

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
    print(f"  Run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total : {len(all_results)}")
    print(f"  Passed: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"\n  Failed experiments:")
        for exp_id in failed:
            print(f"    - {exp_id}")
    print(f"\n  Results saved → {CSV_PATH}")
    print(f"  Checkpoints  → {CHECKPOINT_DIR}")

if __name__ == '__main__':
    main()