import math
import random
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

class CL_Logic:
    def _init_CL(self, config):
        self._last_cl_metrics = None
        self.lmd = config['lmd'] if 'lmd' in config else 0.1
        self.tau = config['tau'] if 'tau' in config else 0.2
        self.sim = config['sim'] if 'sim' in config else 'dot'
        self.cl_loss_type = config['cl_loss_type'] if 'cl_loss_type' in config else 'info_nce'

        # CL loss function
        self.cl_fct = torch.nn.CrossEntropyLoss()
        self.batch_size = config['train_batch_size']
        self.mask_default = self._mask_correlated_samples(batch_size=self.batch_size)


    def _init_embedding(self):
        current_size = self.item_embedding.num_embeddings
        expected_size = self.n_items + 1  # +1 for mask token

        if current_size < expected_size:
            old_weight = self.item_embedding.weight.data
            self.item_embedding = nn.Embedding(
                expected_size, self.hidden_size, padding_idx=0
            )
            # Preserve existing weights, only add new mask token row
            self.item_embedding.weight.data[:current_size] = old_weight
            # Initialize mask token embedding
            self.item_embedding.weight.data[current_size:].normal_(0.0, 0.02)

    def _mask_correlated_samples(self, batch_size):
        """Create mask for negative samples in contrastive learning"""
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """Calculate InfoNCE loss"""
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim_matrix = torch.mm(z, z.T) / temp
        else:
            raise NotImplementedError("Make sure 'sim' in ['cos', 'dot']!")

        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self._mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim_matrix[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def _decoupled_contrastive_loss(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        Calculate DCL loss.
        Differs from InfoNCE by excluding the positive pair from the denominator,
        giving a cleaner gradient signal.
        """
        z_i = F.normalize(z_i,dim=-1)
        z_j = F.normalize(z_j,dim=-1)

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim_matrix = torch.mm(z, z.T) / temp
        else:
            raise NotImplementedError("Make sure 'sim' in ['cos', 'dot']!")

        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)

        # Positive similarities
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # shape (N,)

        # Negative mask same as InfoNCE, excludes diagonal and positive pairs
        if batch_size != self.batch_size:
            mask = self._mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default

        # Negatives only in denominator is key difference from InfoNCE
        negative_samples = sim_matrix[mask].reshape(N, -1)  # shape (N, N-2)

        # DCL loss: -positive + log(sum(exp(negatives)))
        loss = (-positive_samples + torch.logsumexp(negative_samples, dim=1)).mean()

        return loss

    def _decompose(self, z_i, z_j, origin_z, batch_size):
        """Calculate alignment and uniformity metrics"""
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distance
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distance for original representations
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    #Augmentation Methods
    def _augment(self, item_seq, item_seq_len):
        """Generate two augmented views entirely on the GPU."""

        device = item_seq.device
        batch_size = item_seq.size(0)

        # Each sequence gets two DIFFERENT augmentations.
        # 0 = crop, 1 = mask, 2 = reorder
        #
        # Generate two random values and derive a pair without replacement.
        rand = torch.rand(batch_size, 2, device=device)

        first = torch.floor(rand[:, 0] * 3).long()
        second = torch.floor(rand[:, 1] * 2).long()

        # Map second choice so it cannot equal first.
        second = second + (second >= first).long()

        # Sequences with length <= 1 receive no augmentation.
        valid = item_seq_len > 1

        first = torch.where(valid, first, torch.full_like(first, 3))
        second = torch.where(valid, second, torch.full_like(second, 3))

        # Start with unchanged copies.
        aug_seq1 = item_seq.clone()
        aug_seq2 = item_seq.clone()

        aug_len1 = item_seq_len.clone()
        aug_len2 = item_seq_len.clone()

        # Apply each augmentation to the appropriate samples.
        for op in range(3):

            mask1 = first == op
            mask2 = second == op

            if mask1.any():
                seq, length = self._apply_augmentation(
                    item_seq[mask1],
                    item_seq_len[mask1],
                    op
                )
                aug_seq1[mask1] = seq
                aug_len1[mask1] = length

            if mask2.any():
                seq, length = self._apply_augmentation(
                    item_seq[mask2],
                    item_seq_len[mask2],
                    op
                )
                aug_seq2[mask2] = seq
                aug_len2[mask2] = length

        return aug_seq1, aug_len1, aug_seq2, aug_len2

    def _apply_augmentation(self, item_seq, item_seq_len, operation):
        """Apply one augmentation to a batch."""

        if operation == 0:
            return self._item_crop(item_seq, item_seq_len)

        elif operation == 1:
            return self._item_mask(item_seq, item_seq_len)

        elif operation == 2:
            return self._item_reorder(item_seq, item_seq_len)

        return item_seq, item_seq_len

    def _item_crop(self, item_seq, item_seq_len, eta=0.6):
        """Randomly crop each sequence in the batch."""

        device = item_seq.device
        batch_size, max_len = item_seq.shape

        num_left = torch.floor(item_seq_len.float() * eta).long()

        # Start with unchanged sequences.
        cropped_seq = item_seq.clone()
        cropped_len = item_seq_len.clone()

        valid = (num_left > 0) & (num_left < item_seq_len)

        if not valid.any():
            return cropped_seq, cropped_len

        # Random starting position for every sequence.
        max_start = item_seq_len - num_left + 1

        random_values = torch.rand(batch_size,device=device)

        crop_begin = (random_values * max_start.float()).long()

        # Position indices [batch, max_len]
        positions = torch.arange( max_len,device=device).unsqueeze(0)

        source_positions = crop_begin.unsqueeze(1) + positions

        # Valid positions inside the cropped subsequence.
        crop_mask = positions < num_left.unsqueeze(1)

        # Prevent out-of-bounds indexing.
        safe_positions = source_positions.clamp(max=max_len - 1)

        gathered = torch.gather(item_seq,1,safe_positions)

        cropped_seq = torch.where(valid.unsqueeze(1) & crop_mask,gathered,cropped_seq)

        # Positions after the cropped sequence must be zero.
        cropped_seq = torch.where(valid.unsqueeze(1) & (positions >= num_left.unsqueeze(1)),torch.zeros_like(cropped_seq),cropped_seq)

        cropped_len = torch.where(valid,num_left,item_seq_len)

        return cropped_seq, cropped_len

    def _item_mask(self, item_seq, item_seq_len, gamma=0.3):
        """Randomly mask items in each sequence."""

        device = item_seq.device
        batch_size, max_len = item_seq.shape

        num_mask = torch.floor(item_seq_len.float() * gamma).long()

        if not (num_mask > 0).any():
            return item_seq, item_seq_len

        # Random scores for every position.
        random_scores = torch.rand(batch_size,max_len,device=device)

        # Positions outside the actual sequence receive infinity,
        # ensuring they are never selected for masking.
        positions = torch.arange(max_len,device=device).unsqueeze(0)

        valid_positions = positions < item_seq_len.unsqueeze(1)

        random_scores = random_scores.masked_fill(~valid_positions,float("inf"))

        # Select the lowest random scores.
        mask_indices = torch.argsort(random_scores,dim=1)

        mask_positions = (torch.arange(max_len, device=device).unsqueeze(0)< num_mask.unsqueeze(1))

        mask_indices = mask_indices.masked_fill(~mask_positions,0)

        masked_seq = item_seq.clone()

        # Build a [batch, max_len] boolean mask.
        batch_indices = torch.arange(batch_size,device=device).unsqueeze(1)

        mask = torch.zeros(batch_size,max_len,dtype=torch.bool,device=device)

        mask.scatter_(1,mask_indices,mask_positions)

        masked_seq = torch.where(mask,torch.tensor(self.n_items,dtype=item_seq.dtype,device=device),masked_seq)

        return masked_seq, item_seq_len

    def _item_reorder(self, item_seq, item_seq_len, beta=0.6):
        """Randomly reorder a subsequence in each sequence."""

        device = item_seq.device
        batch_size, max_len = item_seq.shape

        num_reorder = torch.floor(item_seq_len.float() * beta).long()

        valid = ((num_reorder > 1) &(num_reorder < item_seq_len))

        if not valid.any():
            return item_seq, item_seq_len

        # Random starting position.
        max_start = item_seq_len - num_reorder + 1

        random_values = torch.rand(batch_size,device=device)

        reorder_begin = (random_values * max_start.float()).long()

        positions = torch.arange(max_len,device=device).unsqueeze(0)

        # Relative positions within the reorder window.
        relative_positions = positions - reorder_begin.unsqueeze(1)

        inside = (valid.unsqueeze(1) &(relative_positions >= 0) &(relative_positions < num_reorder.unsqueeze(1)))

        # Random values used to generate a permutation.
        random_order = torch.rand(batch_size,max_len,device=device)

        random_order = random_order.masked_fill(~inside,float("inf"))

        permutation = torch.argsort(random_order,dim=1)

        # We need the source positions corresponding to each destination
        # position inside the reorder window.
        sorted_positions = permutation

        reordered_seq = item_seq.clone()

        batch_indices = torch.arange(batch_size,device=device).unsqueeze(1)

        # Extract the randomly ordered values.
        gathered = torch.gather(item_seq,1,sorted_positions)

        # Only write back into the reorder region.
        reordered_seq = torch.where(inside,gathered,reordered_seq)

        return reordered_seq, item_seq_len


    def get_cl_metrics(self):
        """Get the latest contrastive learning metrics"""
        if hasattr(self, '_last_cl_metrics'):
            return self._last_cl_metrics
        return None

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight[:self.n_items]  # exclude mask token
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores