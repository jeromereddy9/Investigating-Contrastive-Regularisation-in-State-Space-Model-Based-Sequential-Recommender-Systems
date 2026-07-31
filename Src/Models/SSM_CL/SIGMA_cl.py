import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.models.Baselines.gated_mamba import SIGMA
from src.models.SSM_CL.cl_logic import CL_Logic
import torch

class SIGMA_CL(CL_Logic,SIGMA):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_CL(config)
        self._init_embedding()

    def calculate_loss(self, interaction):
        """
        Calculate combined loss: main recommendation loss + contrastive loss
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # main recommendation loss
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            main_loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type == 'CE'
            test_item_emb = self.item_embedding.weight[:self.n_items]  # exclude mask token
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # Contrastive Learning
        # Generate augmented sequences
        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = self._augment(item_seq, item_seq_len)

        # Get representations for augmented views
        seq_output1 = self.forward(aug_item_seq1, aug_len1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)

        if self.cl_loss_type == 'info_nce':
            # Calculate InfoNCE loss
            nce_logits, nce_labels = self._info_nce(
                seq_output1, seq_output2,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim
            )
            cl_loss = self.cl_fct(nce_logits, nce_labels)
        elif self.cl_loss_type == 'dcl':
            # Calculate DCL loss
            cl_loss = self._decoupled_contrastive_loss(
                seq_output1, seq_output2,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim
            )
        else:
            cl_loss = torch.tensor(0.0, device=seq_output.device)

        # Calculate alignment and uniformity for monitoring
        with torch.no_grad():
            alignment, uniformity = self._decompose(
                seq_output1, seq_output2, seq_output,
                batch_size=item_seq_len.shape[0]
            )

        # Combined loss
        total_loss = main_loss + self.lmd * cl_loss

        self._last_cl_metrics = {
            'alignment': alignment.item(),
            'uniformity': uniformity.item(),
            f'{self.cl_loss_type}': cl_loss.item(),
            'main_loss': main_loss.item()
        }

        return total_loss