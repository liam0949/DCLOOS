import os
import numpy as np
import torch
import random
import csv
import sys
from transformers import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


class Data:

    def __init__(self, args):
        max_seq_lengths = {'oos': 30, 'stackoverflow': 45, 'banking': 55}
        args.max_seq_length = max_seq_lengths[args.dataset_pos]

        self.in_neg_list = ["book_flight",
                            "book_hotel",
                            "car_rental",
                            "travel_suggestion",
                            "travel_alert",
                            "travel_notification",
                            "carry_on",
                            "timezone",
                            "vaccines",
                            "translate",
                            "flight_status",
                            "international_visa",
                            "lost_luggage",
                            "plug_type",
                            "exchange_rate", "direct_deposit",
                            "pto_request",
                            "taxes",
                            "payday",
                            "w2",
                            "pto_balance",
                            "pto_request_status",
                            "next_holiday",
                            "insurance",
                            "insurance_change",
                            "schedule_meeting",
                            "pto_used",
                            "meeting_schedule",
                            "rollover_401k",
                            "income"]
        processor = DatasetProcessor()
        self.data_dir_pos = os.path.join(args.data_dir, args.dataset_pos)
        self.data_dir_neg = os.path.join(args.data_dir, args.dataset_pos)

        self.all_label_list_pos = processor.get_labels(self.data_dir_pos)
        self.n_known_cls = round(len(self.all_label_list_pos) * args.known_cls_ratio)
        self.known_label_list = list(
            np.random.choice(np.array(self.all_label_list_pos), self.n_known_cls, replace=False))
        self.known_label_list = [item for item in self.known_label_list if item not in self.in_neg_list]

        self.num_labels = len(self.known_label_list)

        if args.dataset_pos == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'

        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]
        self.train_examples = self.get_examples(processor, args, 'train')
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        self.neg_examples = self.get_examples(processor, args, 'neg')

        self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
        self.neg_dataloader = self.get_loader(self.neg_examples, args, 'neg')

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir_pos, mode)

        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)

        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    examples.append(example)
        elif mode == 'test':
            for example in ori_examples:
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    examples.append(example)
        elif mode == 'neg':
            """adding negative set"""
            for example in ori_examples:
                if (example.label in self.in_neg_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    example.label = self.unseen_token
                    examples.append(example)

        return examples

    def get_loader(self, examples, args, mode='train'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer)

        input_ids = torch.stack([f.input_ids for f in features])
        input_mask = torch.stack([f.input_mask for f in features])
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            # sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.eval_batch_size)
        elif mode == 'neg':
            # sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, batch_size=args.n_oos, shuffle=True, drop_last=True)

        return dataloader


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'neg':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "neg")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        # tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = tokenizer.encode_plus(
            example.text_a,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )

        segment_ids = [0] * len(tokens_a['input_ids'][0])
        input_ids = tokens_a['input_ids'][0]
        input_mask = tokens_a['attention_mask'][0]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
