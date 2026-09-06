import os
import sys
import warnings
import logging
import csv
from datetime import datetime
warnings.filterwarnings('ignore')
logging.getLogger('recbole').setLevel(logging.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import traceback
import time
import numpy as np
import torch

from src.utils import path_builder
from recbole.config.configurator import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


from src.models.Baselines.mamba4rec import Mamba4Rec
from src.models.Baselines.gated_mamba import SIGMA
from src.models.SSM_CL.Model_Level.mamba4rec_cl import Mamba4Rec_CL
from src.models.SSM_CL.Model_Level.SIGMA_cl import SIGMA_CL

# Geometry Functions
def compute_isotropy(embeddings):
    embeddings = embeddings - embeddings.mean(dim=0)
    cov = torch.mm(embeddings.T, embeddings) / embeddings.shape[0]
    eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=1e-10)
    eigenvalues = eigenvalues / eigenvalues.sum()
    entropy = -(eigenvalues * torch.log(eigenvalues)).sum()
    return (torch.exp(entropy) / embeddings.shape[1]).item()

def compute_effective_rank(embeddings):
    embeddings = embeddings - embeddings.mean(dim=0)
    _, singular_values, _ = torch.linalg.svd(embeddings, full_matrices=False)
    singular_values = singular_values.clamp(min=1e-10)
    singular_values = singular_values / singular_values.sum()
    entropy = -(singular_values * torch.log(singular_values)).sum()
    return torch.exp(entropy).item()

def extract_embeddings(model, n_items):
    with torch.no_grad():
        return model.item_embedding.weight[:n_items].cpu().float()


# Config

TEST_DATASET  = 'amazon_toys_and_games'
TEST_EPOCHS   = 30
TEST_BATCH    = 512

CONFIG_DIR        = path_builder('src/configs')
DATASET_CONFIG    = path_builder(CONFIG_DIR + '/dataset.yaml')
TRAINING_CONFIG   = path_builder(CONFIG_DIR + '/training.yaml')
MODELS_CONFIG_DIR = path_builder(CONFIG_DIR + '/models')

# Output CSV
RESULTS_CSV = path_builder(f'src/test_results_{TEST_DATASET}_{TEST_EPOCHS}.csv')


# Models to Test
TESTS = [
    (Mamba4Rec,    'Mamba4Rec',    'mamba4rec',     'CE',    None),
    (Mamba4Rec,    'Mamba4Rec',    'mamba4rec',     'BPR',   None),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'CE',    'info_nce'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'CE',    'dcl'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'BPR',   'info_nce'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'BPR',   'dcl'),
    (SIGMA,        'SIGMA',        'sigma',         'CE',    None),
    (SIGMA,        'SIGMA',        'sigma',         'BPR',   None),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'CE',    'info_nce'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'CE',    'dcl'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'BPR',   'info_nce'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'BPR',   'dcl'),
]

CSV_HEADERS = [
    'timestamp', 'dataset', 'model', 'loss_type', 'cl_loss_type',
    'status', 'error',
    'hit@5', 'hit@10', 'hit@20',
    'ndcg@5', 'ndcg@10', 'ndcg@20',
    'mrr@5', 'mrr@10', 'mrr@20',
    'isotropy', 'effective_rank', 'embedding_dim', 'n_items',
    'total_time_sec'
]


# Test Runner
def run_test(model_class, model_name, config_file, loss_type, cl_loss_type):
    label = f"{model_name}_{loss_type}" if loss_type else model_name
    if cl_loss_type:
        label += f"_{cl_loss_type}"

    print(f"  {label}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")


    result = {
        'timestamp': datetime.now().isoformat(),
        'dataset': TEST_DATASET,
        'model': model_name,
        'loss_type': loss_type or 'default',
        'cl_loss_type': cl_loss_type or 'none',
        'status': 'failed',
        'error': '',
        'hit@5': None, 'hit@10': None, 'hit@20': None,
        'ndcg@5': None, 'ndcg@10': None, 'ndcg@20': None,
        'mrr@5': None, 'mrr@10': None, 'mrr@20': None,
        'isotropy': None, 'effective_rank': None,
        'embedding_dim': None, 'n_items': None,
        'total_time_sec': None,
    }

    try:
        config_dict = {
            'epochs': TEST_EPOCHS,
            'train_batch_size': TEST_BATCH,
            'eval_batch_size': TEST_BATCH,
            'stopping_step': TEST_EPOCHS,
        }
        if loss_type:
            config_dict['loss_type'] = loss_type
        if cl_loss_type:
            config_dict['cl_loss_type'] = cl_loss_type
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
            dataset=TEST_DATASET,
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

        print(f"  Device: {config['device']}")

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = model_class(config, dataset).to(config['device'])
        trainer = Trainer(config, model)

        # Train
        start = time.perf_counter()
        trainer.fit(train_data, valid_data, saved=False, show_progress=True)
        train_time = time.perf_counter() - start

        # Evaluate on test set
        print(f"\n  Evaluating on test set...")
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

        # Geometry
        print(f" Computing geometry...")
        n_items = config['n_items'] if 'n_items' in config else dataset.item_num
        embeddings = extract_embeddings(model, n_items)
        isotropy = compute_isotropy(embeddings)
        eff_rank = compute_effective_rank(embeddings)

        # Fill results
        result.update({
            'status': 'success',
            'total_time_sec': train_time,
            'hit@5': test_result.get('hit@5'),
            'hit@10': test_result.get('hit@10'),
            'hit@20': test_result.get('hit@20'),
            'ndcg@5': test_result.get('ndcg@5'),
            'ndcg@10': test_result.get('ndcg@10'),
            'ndcg@20': test_result.get('ndcg@20'),
            'mrr@5': test_result.get('mrr@5'),
            'mrr@10': test_result.get('mrr@10'),
            'mrr@20': test_result.get('mrr@20'),
            'isotropy': isotropy,
            'effective_rank': eff_rank,
            'embedding_dim': embeddings.shape[1],
            'n_items': embeddings.shape[0],
        })

        print(f"\n  PASS")
        print(f"  NDCG@10: {result['ndcg@10']:.4f}")
        print(f"  Isotropy: {result['isotropy']:.4f}")
        print(f"  Eff. Rank: {result['effective_rank']:.2f}/{result['embedding_dim']}")
        print(f"  Time: {result['total_time_sec']/60:.1f} min")

    except Exception as e:
        result['error'] = str(e)
        print(f"  FAIL: {e}")
        traceback.print_exc()

    return result

# Main
def main():
    print(f"\nTest dataset : {TEST_DATASET}")
    print(f"Test epochs  : {TEST_EPOCHS}")
    print(f"Test batch   : {TEST_BATCH}")
    print(f"Output       : {RESULTS_CSV}")

    results = []
    for model_class, model_name, config_file, loss_type, cl_loss_type in TESTS:
        result = run_test(model_class, model_name, config_file, loss_type, cl_loss_type)
        results.append(result)

    # Save CSV
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k in CSV_HEADERS}
            writer.writerow(row)

    # Summary
    passed = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Total: {len(results)}")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Results: {RESULTS_CSV}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()