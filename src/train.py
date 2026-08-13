import os
import sys
import csv
import pickle
import traceback
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('recbole').setLevel(logging.ERROR)

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger

from src.utils import path_builder
from src.models.Baselines.GRU4Rec import GRU4Rec
from src.models.Baselines.SASRec import SASRec
from src.models.Baselines.CL4SRec import CL4SRec
from src.models.Baselines.DouRec import DuoRec
from src.models.Baselines.mamba4rec import Mamba4Rec
from src.models.Baselines.gated_mamba import SIGMA
from src.models.SSM_CL.mamba4rec_cl import Mamba4Rec_CL
from src.models.SSM_CL.SIGMA_cl import SIGMA_CL


VALID_STAGES = ['1', '2', '3', '4', '5', '6']
STAGE = os.environ.get('STAGE', '1')
assert STAGE in VALID_STAGES, f"STAGE must be one of {VALID_STAGES}"

STAGE_EXPERIMENTS = {
    '1': [
        (Mamba4Rec, 'Mamba4Rec', 'mamba4rec', 'CE',  None),
        (Mamba4Rec, 'Mamba4Rec', 'mamba4rec', 'BPR', None),
        (SIGMA,     'SIGMA',     'sigma',     'CE',  None),
        (SIGMA,     'SIGMA',     'sigma',     'BPR', None),
    ],
    '2': [
        (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl', 'CE', 'info_nce'),
        (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl', 'CE', 'dcl'),
    ],
    '3': [
        (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl', 'BPR', 'info_nce'),
        (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl', 'BPR', 'dcl'),
    ],
    '4': [
        (SIGMA_CL, 'SIGMA_CL', 'sigma_cl', 'CE', 'info_nce'),
        (SIGMA_CL, 'SIGMA_CL', 'sigma_cl', 'CE', 'dcl'),
    ],
    '5': [
        (SIGMA_CL, 'SIGMA_CL', 'sigma_cl', 'BPR', 'info_nce'),
        (SIGMA_CL, 'SIGMA_CL', 'sigma_cl', 'BPR', 'dcl'),
    ],
    '6': [
        (GRU4Rec, 'GRU4Rec', 'gru4rec', None, None),
        (SASRec,  'SASRec',  'sasrec',  None, None),
        (CL4SRec, 'CL4SRec', 'cl4srec', None, None),
        (DuoRec,  'DuoRec',  'duorec',  None, None),
    ],
}

EXPERIMENTS = STAGE_EXPERIMENTS[STAGE]

DATASETS = [
    'amazon_videogames',
    'amazon_toys_and_games',
    'movielens_1m',
    'lastfm_1k',
]


# Paths

CONFIG_DIR        = path_builder('src/configs')
DATASET_CONFIG    = path_builder(CONFIG_DIR + '/dataset.yaml')
TRAINING_CONFIG   = path_builder(CONFIG_DIR + '/training.yaml')
MODELS_CONFIG_DIR = path_builder(CONFIG_DIR + '/models')
OUTPUT_DIR        = path_builder('src/results')
CHECKPOINT_DIR    = path_builder('src/checkpoints')
CSV_PATH          = path_builder(OUTPUT_DIR + f'/stage{STAGE}_results.csv')

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# CSV

CSV_HEADERS = [
    'timestamp', 'dataset', 'model', 'loss_type', 'cl_loss_type', 'exp_id',
    'best_valid_score', 'best_epoch', 'total_epochs_run', 'early_stopped',
    'status', 'error'
]

def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()

def log_result(row):
    with open(CSV_PATH, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)

def make_exp_id(model_name, loss_type, cl_loss_type, dataset):
    parts = [model_name]
    if loss_type:    parts.append(loss_type)
    if cl_loss_type: parts.append(cl_loss_type)
    parts.append(dataset)
    return '_'.join(parts)


# Single experiment

def run_experiment(model_class, model_name, config_file,
                   dataset_name, loss_type=None, cl_loss_type=None):
    import time
    exp_id = make_exp_id(model_name, loss_type, cl_loss_type, dataset_name)
    
    print(f"  {exp_id}")

    row = {
        'timestamp':        datetime.now().isoformat(),
        'dataset':          dataset_name,
        'model':            model_name,
        'loss_type':        loss_type or 'default',
        'cl_loss_type':     cl_loss_type or 'none',
        'exp_id':           exp_id,
        'best_valid_score': None,
        'best_epoch':       None,
        'total_epochs_run': None,
        'early_stopped':    None,
        'status':           'failed',
        'error':            '',
    }

    try:
        config_dict = {}
        if loss_type:    config_dict['loss_type']    = loss_type
        if cl_loss_type: config_dict['cl_loss_type'] = cl_loss_type
        if loss_type == 'BPR':
            config_dict['train_neg_sample_args'] = {
                'distribution': 'uniform',
                'sample_num': 1,
                'alpha': 1.0,
                'dynamic': False,
                'candidate_num': 0
            }

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

        print(f"  Device : {config['device']}")
        print(f"  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = model_class(config, dataset).to(config['device'])
        logger.info(model)

        trainer = Trainer(config, model)
        start = time.perf_counter()
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=True
        )
        elapsed = time.perf_counter() - start

        train_loss_history = trainer.train_loss_dict
        best_epoch    = getattr(trainer, 'best_valid_epoch', None)
        total_epochs  = len(train_loss_history)
        early_stopped = total_epochs < config['epochs']

        epoch_history = [
            {'epoch': e, 'train_loss': l}
            for e, l in train_loss_history.items()
        ]

        row.update({
            'best_valid_score': best_valid_score,
            'best_epoch':       best_epoch,
            'total_epochs_run': total_epochs,
            'early_stopped':    early_stopped,
            'status':           'success',
        })

        ckpt_path = os.path.join(CHECKPOINT_DIR, f'{exp_id}.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'model_state_dict':  trainer.model.state_dict(),
                'config': {k: v for k, v in config.final_config_dict.items()},
                'best_valid_score':  best_valid_score,
                'best_valid_result': best_valid_result,
                'best_epoch':        best_epoch,
                'total_epochs_run':  total_epochs,
                'early_stopped':     early_stopped,
                'epoch_history':     epoch_history,
                'exp_id':            exp_id,
            }, f)

        print(f"  Training time  : {elapsed/60:.1f} min ({elapsed/total_epochs:.0f}s/epoch)")
        print(f"  Best valid     : {best_valid_score:.4f} at epoch {best_epoch}")
        print(f"  Epochs run     : {total_epochs} (early stopped: {early_stopped})")
        print(f"  Checkpoint     : {ckpt_path}")

    except Exception as e:
        row['error'] = str(e)
        print(f"  FAILED: {e}")
        traceback.print_exc()

    log_result(row)
    return row


# Main

def main():
    init_csv()
    run_start = datetime.now()
    print(f"\nStage {STAGE} training started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets    : {DATASETS}")
    print(f"Experiments : {len(EXPERIMENTS)} per dataset")
    print(f"Total       : {len(DATASETS) * len(EXPERIMENTS)}")

    all_results, failed = [], []

    for dataset in DATASETS:
        print(f"\n  Dataset: {dataset}\n")
        for model_class, model_name, config_file, loss_type, cl_loss_type in EXPERIMENTS:
            result = run_experiment(
                model_class=model_class,
                model_name=model_name,
                config_file=config_file,
                dataset_name=dataset,
                loss_type=loss_type,
                cl_loss_type=cl_loss_type,
            )
            all_results.append(result)
            if result['status'] == 'failed':
                failed.append(result['exp_id'])

    elapsed_total = (datetime.now() - run_start).total_seconds() / 3600
    print(f"  Stage {STAGE} complete : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time : {elapsed_total:.1f} hours")
    print(f"  Total      : {len(all_results)}")
    print(f"  Passed     : {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"\n  Failed experiments:")
        for exp_id in failed:
            print(f"    - {exp_id}")
    print(f"\n  Results: {CSV_PATH}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")

if __name__ == '__main__':
    main()
