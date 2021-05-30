"""
Modified based on SCL by Author: Yonglong Tian (yonglong@mit.edu)
to achieve domain contrastive loss
zhan liming
1.12.2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLossD(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossD, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, args, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # labels = labels.contiguous().view(-1, 1)
            labels = torch.zeros_like(labels.contiguous().view(-1, 1)) \
                .scatter_(1, torch.zeros(args.train_batch_size, dtype=torch.long).view(-1, 1).cuda(), 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # print(labels)
            mask = torch.eq(labels, labels.T).float()  # change to [11111,00000]
            # print(mask)
        else:
            mask = mask.float().to(device)

        contrast_count = 1  # change to negtive num
        # contrast_feature = F.normalize(features, p=2, dim=1)  # slice along the contrstt dim
        contrast_feature = features  # slice along the contrstt dim

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            # anchor_count = contrast_count
            anchor_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)

        # compute mean of log-likelihood over positive
        # print(mask.sum(1))
        if torch.min(mask.sum(1)) == 0:
            raise ValueError('sum is 0, nan is coming')
        # print(mask * log_prob)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print(mean_log_prob_pos)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        # loss = loss.sum()
        # print(loss)

        return loss


class SupConLossC(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossC, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, args, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # labels = torch.zeros_like(labels.contiguous().view(-1, 1)) \
            #     .scatter_(1, torch.zeros(args.train_batch_size, dtype=torch.long).view(-1, 1).cuda(), 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # print(labels)
            mask = torch.eq(labels, labels.T).float()  # change to [11111,00000]
            # print(mask)
        else:
            mask = mask.float().to(device)

        contrast_count = 1  # change to negtive num
        # contrast_feature = F.normalize(features, p=2, dim=1)  # slice along the contrstt dim
        contrast_feature = features  # slice along the contrstt dim

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            # anchor_count = contrast_count
            anchor_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        # print(mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)

        # compute mean of log-likelihood over positive
        # print(mask.sum(1))
        # if torch.min(mask.sum(1)) == 0:
        #     raise ValueError('sum is 0, nan is coming')
        # print(mask * log_prob)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-8)
        # print(mean_log_prob_pos)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        # loss = loss.sum()
        # print(loss)

        return loss
