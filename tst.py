import sys
import os
import csv
from dataloader import *
from configs import *
import torch

# print(sys.version_info[0])
# # i = 0
# with open(os.path.join(r'E:\projects\Question-Answering-based-on-SQuAD-master\SQuAD\train', 'train.question'), "r",
#           encoding="utf-8") as context:
#     context_file = context.readlines()
#
#     with open('squad.tsv', mode='w', encoding='utf-8', newline='') as csv_file:
#         fieldnames = ['text', 'label']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')
#
#         writer.writeheader()
#         for i in context_file:
#             q = i[:-2]
#             writer.writerow({'text':q.lower(), 'label': "oos"})
#
#     # print('\n')
import json
import nltk
from csv import writer
# from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
# # Opening JSON file
# with open(r'E:\projects\COSINE-main\data\Yelp\train_data.json') as json_file:
#     data = json.load(json_file)
#     with open("squad.tsv", 'a+', encoding="utf-8") as csv_file:
#         # Create a writer object from csv module
#         fieldnames = ['text', 'label']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')
#         # Add contents of list as last row in the csv file
#         for i in data:
#             for j in sent_tokenize(i['text']):
#                 writer.writerow({"text": j.lower(), 'label':'oos'})
# for i in data:
#     for j in sent_tokenize(i['text']):


# Print the type of data variable

import torch

# args = parse_args('train')
# datas = Data(args)
# a = datas.train_dataloader
# b = datas.neg_dataloader
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# for i in zip(a, b):
#     # a = torch.stack([i[0],j[0]])
#     print(i)
#
#     break
# print(torch.zeros(10))
# labels = torch.arange(10)
# labels = torch.zeros_like(labels.contiguous().view(-1, 1)) \
#                 .scatter_(1, torch.zeros(5, dtype=torch.long).view(-1, 1), 1)
# print(labels)
# labels = torch.rand(25).view(5,5)
# print(torch.min(labels))
# if torch.min(labels)!=0:
#     print("hh")


# with open(os.path.join(r'data', 'test.tsv'), "r",
#           encoding="utf-8") as context:
#     reader = csv.reader(context, delimiter="\t", quotechar=None)
#     count = 0
#     valset = []
#     testset = []
#
#     # for line in reader:
#     #     lines.append(line)
#     for i, tet in enumerate(reader):
#         if i == 0:
#             continue
#         if len(tet) != 2:
#             continue
#         # print(tet[0])
#         # break
#         # print(tet)
#         if tet[1] == 'oos':
#             if count > 99:
#                 testset.append(tet)
#             else:
#                 valset.append(tet)
#             count += 1
#         else:
#             testset.append(tet)
#     with open('test.tsv', mode='w', encoding='utf-8', newline='') as csv_file:
#         fieldnames = ['text', 'label']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')
#
#         writer.writeheader()
#         for i in testset:
#             writer.writerow({'text': i[0].lower(), 'label': i[1]})
#     with open('data/dev.tsv', mode='a+', encoding='utf-8', newline='') as csv_file:
#         fieldnames = ['text', 'label']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')
#
#         # writer.writeheader()
#         for i in valset:
#             writer.writerow({'text': i[0].lower(), 'label': i[1]})

#     #
#
#     print(valset[:10])
#     print(len(testset))
# ids = torch.tensor([1,2,3]).view(-1,1)
# idsl = []
# for i in range(10):
#     idsl.append(ids.numpy())
# print(len(idsl))
# print(np.concatenate(idsl, axis=0))

# known_label_list = [item for item in [1,2,34,5,6] if item not in [1,34]]
# print(known_label_list)
from losses import SupConLossC
logits = torch.randn([8,20])
labels = torch.tensor([1,2,1,5,4,2,5,4])

loss = SupConLossC()
loss(logits, args=None, labels=labels)