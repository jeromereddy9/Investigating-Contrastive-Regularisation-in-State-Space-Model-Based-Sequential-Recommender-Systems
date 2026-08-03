import os
import sys
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('recbole').setLevel(logging.ERROR)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import traceback
from src.utils import path_builder

import torch
from recbole.config.configurator import Config
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


# Test config
TEST_DATASET  =  'amazon_videogames'
TEST_EPOCHS   = 3
TEST_BATCH    = 64

CONFIG_DIR        = path_builder('src/configs')
DATASET_CONFIG    = path_builder(CONFIG_DIR + '/dataset.yaml')
TRAINING_CONFIG   = path_builder(CONFIG_DIR + '/training.yaml')
MODELS_CONFIG_DIR = path_builder(CONFIG_DIR + '/models')


# All models to test
TESTS = [
    (GRU4Rec,      'GRU4Rec',      'gru4rec',      None),
    (SASRec,       'SASRec',       'sasrec',        None),
    (CL4SRec,      'CL4SRec',      'cl4srec',       None),
    (DuoRec,       'DuoRec',       'duorec',        None),
    (Mamba4Rec,    'Mamba4Rec',    'mamba4rec',     'CE'),
    (Mamba4Rec,    'Mamba4Rec',    'mamba4rec',     'BPR'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'CE'),
    (Mamba4Rec_CL, 'Mamba4Rec_CL', 'mamba4rec_cl',  'BPR'),
    (SIGMA,        'SIGMA',        'sigma',         'CE'),
    (SIGMA,        'SIGMA',        'sigma',         'BPR'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'CE'),
    (SIGMA_CL,     'SIGMA_CL',     'sigma_cl',      'BPR'),
]


# CL loss types to test for CL variants
CL_LOSS_TYPES = ['info_nce', 'dcl']

def run_test(model_class, model_name, config_file, loss_type, cl_loss_type=None):
    label = model_name
    if loss_type:
        label += f'_{loss_type}'
    if cl_loss_type:
        label += f'_{cl_loss_type}'

    try:
        config_dict = {
            'epochs': TEST_EPOCHS,
            'train_batch_size': TEST_BATCH,
            'eval_batch_size': TEST_BATCH,
            'stopping_step': TEST_EPOCHS,  # disable early stopping for test
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
        print("RecBole device:", config["device"])
        print("CUDA available:", torch.cuda.is_available())
        config['data_path'] = path_builder('src/datasets/preprocessed/amazon_videogames')

        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = model_class(config, dataset).to(config['device'])
        print("=" * 50)
        print(f"Config device      : {config['device']}")
        print(f"Model device       : {next(model.parameters()).device}")
        print(f"CUDA available     : {torch.cuda.is_available()}")
        print(f"Current GPU        : {torch.cuda.get_device_name(0)}")
        print("=" * 50)
        trainer = Trainer(config, model)
        trainer.fit(train_data, valid_data, saved=False, show_progress=False)

        print(f"  PASS  {label}")
        return True

    except Exception as e:
        print(f"  FAIL  {label}")
        print(f"        {e}")
        traceback.print_exc()
        return False


def main():
    passed = []
    failed = []

    print(f"\nTest dataset : {TEST_DATASET}")
    print(f"Test epochs  : {TEST_EPOCHS}")

    for model_class, model_name, config_file, loss_type in TESTS:
        is_cl = model_class in (Mamba4Rec_CL, SIGMA_CL)

        if is_cl:
            # Test all three CL loss types for CL variants
            for cl_loss_type in CL_LOSS_TYPES:
                ok = run_test(model_class, model_name, config_file, loss_type, cl_loss_type)
                label = f"{model_name}_{loss_type}_{cl_loss_type}" if loss_type else f"{model_name}_{cl_loss_type}"
                (passed if ok else failed).append(label)
        else:
            ok = run_test(model_class, model_name, config_file, loss_type)
            label = f"{model_name}_{loss_type}" if loss_type else model_name
            (passed if ok else failed).append(label)

    # Summary
    print(f"  Passed : {len(passed)}/{len(passed)+len(failed)}")
    print(f"  Failed : {len(failed)}")
    if failed:
        print(f"\n  Failed tests:")
        for f in failed:
            print(f"    - {f}")

if __name__ == '__main__':
    main()