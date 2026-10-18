import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.models.Baselines.mamba4rec import Mamba4Rec
from src.models.SSM_CL.cl_logic import CL_Logic
import torch

class Mamba4Rec_CL(CL_Logic,Mamba4Rec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_CL(config)
        self._init_embedding()

    def calculate_loss(self, interaction):
        """
         Calculate combined loss: main recommendation loss + contrastive loss (CL applied at the Model-level)
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        # Main recommendation loss
        # First forward pass recommendation (train mode, dropout active)
        seq_output = self.forward(item_seq, item_seq_len)

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            main_loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight[:self.n_items]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # Model-level CL
        # Two additional forward passes dropout creates different views
        seq_output1 = self.forward(item_seq, item_seq_len)
        seq_output2 = self.forward(item_seq, item_seq_len)

        # Project to CL space
        z1 = self.cl_projector(seq_output1)
        z2 = self.cl_projector(seq_output2)

        if self.cl_loss_type == 'info_nce':
            nce_logits, nce_labels = self._info_nce(
                z1, z2,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim
            )
            cl_loss = self.cl_fct(nce_logits, nce_labels)
        elif self.cl_loss_type == 'dcl':
            cl_loss = self._decoupled_contrastive_loss(
                z1, z2,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim
            )
        else:
            cl_loss = torch.tensor(0.0, device=seq_output.device)

        with torch.no_grad():
            alignment, uniformity = self._decompose(
                seq_output1, seq_output2, seq_output,
                batch_size=item_seq_len.shape[0]
            )

        total_loss = main_loss + self.lmd * cl_loss

        self._last_cl_metrics = {
            'alignment': alignment.item(),
            'uniformity': uniformity.item(),
            f'{self.cl_loss_type}': cl_loss.item(),
            'main_loss': main_loss.item(),
        }

        return total_loss