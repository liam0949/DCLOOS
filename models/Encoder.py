import torch.nn as nn
import torch
from transformers import BertModel, BertForSequenceClassification
# from transformers.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from transformers import DistilBertModel
import numpy as np
import math
import torch.nn.functional as F


class BertEncoder(nn.Module):
    def __init__(self, bert_path=None):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x, x_mask):
        return self.encoder(x, attention_mask=x_mask)[1]  # [bst, max_seq, 768]


class HeadBuilder(nn.Module):
    def __init__(self, in_dim, feat_dim, dropout=0.5):
        super(HeadBuilder, self).__init__()
        self.reduction = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            # nn.ReLU(),
            # nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
            # nn.Linear(in_dim, feat_dim)
        )

    def forward(self, emb):
        emb = self.reduction(emb)
        return emb


# eps = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0)).cuda()
#             B = eps * var + mean
class AVAHead(nn.Module):
    def __init__(self, in_dim, n_way, latent_dim=128, feat_dim=1024, dropout=0.6):
        super(AVAHead, self).__init__()
        self.latent_dim = latent_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            # nn.ReLU(),
            # nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2 * latent_dim)
        )
        self.logits = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            # nn.ReLU()
            # nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_way)
        )

    def forward(self, embs):
        # eps = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))
        # torch.split(self.projector(embs), [bsz, bsz], dim=0)
        #mean, log_var = torch.split(self.projector(embs), [self.latent_dim, self.latent_dim], dim=1)
        #std = torch.exp(0.5 * log_var)
        #eps = torch.randn_like(std)
        #theta = eps * std + mean
        #theta = mean
        logit = self.logits(embs)
        return logit, 0.0, 0.0


class ConvexSampler(nn.Module):
    def __init__(self, args):
        super(ConvexSampler, self).__init__()
        #self.num_convex = round(args.n_oos/5)
        self.num_convex = args.num_convex
        self.num_convex_val = args.num_convex_val
        self.oos_num = args.n_oos
        self.oos_label_id = args.oos_label_id


    def forward(self, z, label_ids, mode=None):
        convex_list = []
        # print(z)
        # print(label_ids)
        if mode =='train':
            if label_ids.size(0)>2:
                while len(convex_list) < self.num_convex:
                    cdt = np.random.choice(label_ids.size(0)-self.oos_num, 2, replace=False)
                    # print(cdt)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(self.num_convex, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.oos_label_id]*self.num_convex).cuda()), dim=0)
        elif mode=='val':
            if label_ids.size(0) > 2:
                print('convex in')
                print(label_ids.size(0))
                val_num = self.num_convex_val
                while len(convex_list) < val_num:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    # if label_ids.size(0) >10:
                    #     print(len(convex_list))
                    # print(cdt)
                    # print(cdt)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(val_num, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat((label_ids, torch.tensor([self.oos_label_id]*val_num).cuda()), dim=0)
                print('convex done')
        # print(z)
        # print(label_ids)
        return z, label_ids


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.mapping = BertEncoder()
        # self.MLP = HeadBuilder(args.in_dim, 128)
        # self.classifier = nn.Linear(128, args.n_way)
        self.sampler = ConvexSampler(args)
        self.head = AVAHead(args.in_dim, args.n_way)

    def forward(self, x, x_mask, label_ids, mode=None):
        feat = self.mapping(x, x_mask)
        if mode is not None:
            feat, label_ids = self.sampler(feat, label_ids, mode=mode)
        logits, mean, log_var = self.head(feat)
        # feat = self.MLP(feat)
        # # feat = F.normalize(feat, p=2, dim=1)
        # logits = self.classifier(feat)
        return logits, mean, log_var, label_ids
