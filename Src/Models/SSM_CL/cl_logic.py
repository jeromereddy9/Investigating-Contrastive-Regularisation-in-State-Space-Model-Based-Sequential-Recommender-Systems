import math
import random
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

class CL_Logic:
    def _init_CL(self, config):
        self._last_cl_metrics = None
        self.lmd = config.get('lmd', 0.1)  # CL loss weight
        self.tau = config.get('tau', 0.2)  # temperature
        self.sim = config.get('sim', 'dot')  # similarity type
        self.cl_loss_type = config.get('cl_loss_type', 'info_nce') # CL loss type

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
        # computed manually rather than via CrossEntropyLoss
        # since positive is NOT included in the denominator
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
        """Generate two augmented views of the input sequence"""
        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []

        for seq, length in zip(item_seq, item_seq_len):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]

            if switch[0] == 0:
                aug_seq, aug_len = self._item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = self._item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = self._item_reorder(seq, length)
            else:
                aug_seq, aug_len = seq, length

            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)

            if switch[1] == 0:
                aug_seq, aug_len = self._item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = self._item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = self._item_reorder(seq, length)
            else:
                aug_seq, aug_len = seq, length

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)

        return torch.stack(aug_seq1), torch.stack(aug_len1), torch.stack(aug_seq2), torch.stack(aug_len2)

    def _item_crop(self, item_seq, item_seq_len, eta=0.6):
        """Randomly crop a subsequence"""
        num_left = math.floor(item_seq_len * eta)
        if num_left == 0 or num_left >= item_seq_len:
            return item_seq, item_seq_len
        crop_begin = random.randint(0, item_seq_len - num_left)
        cropped_item_seq = np.zeros(item_seq.shape[0], dtype=np.int64)
        if crop_begin + num_left < item_seq.shape[0]:
            cropped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            cropped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(cropped_item_seq, dtype=torch.long, device=item_seq.device), \
            torch.tensor(num_left, dtype=torch.long, device=item_seq.device)

    def _item_mask(self, item_seq, item_seq_len, gamma=0.3):
        """Randomly mask items"""
        num_mask = math.floor(item_seq_len * gamma)
        if num_mask == 0:
            return item_seq, item_seq_len
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        # Use n_items as mask token (since embedding size is n_items+1)
        masked_item_seq[mask_index] = self.n_items
        return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

    def _item_reorder(self, item_seq, item_seq_len, beta=0.6):
        """Randomly reorder a subsequence"""
        num_reorder = math.floor(item_seq_len * beta)
        if num_reorder <= 1 or num_reorder >= item_seq_len:
            return item_seq, item_seq_len
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = item_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len


    def get_cl_metrics(self):
        """Get the latest contrastive learning metrics"""
        if hasattr(self, '_last_cl_metrics'):
            return self._last_cl_metrics
        return None