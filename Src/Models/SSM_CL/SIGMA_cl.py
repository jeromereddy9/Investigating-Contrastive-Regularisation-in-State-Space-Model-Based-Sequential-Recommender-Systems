import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from external.libraries.SIGMA.model.gated_mamba import SIGMA

class SIGMA_CL(SIGMA):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def calculate_loss(self, interaction):
        pass

