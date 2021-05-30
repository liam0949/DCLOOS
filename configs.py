import numpy as np
import os
import glob
import argparse

save_dir = '/data1/liming/CloserLookFewShot/'
data_dir = {}
data_dir['bert'] = './filelists/CUB/'
data_dir['oos'] = './filelists/miniImagenet/'
data_dir['squadQ'] = ''


def parse_args(script):
    parser = argparse.ArgumentParser(description='DCLOOS script %s' % (script))
    parser.add_argument("--data_dir", default=r'E:\projects\datasets', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--resume_file", default='/data1/liming', type=str,
                        help="resume file name.")
    parser.add_argument("--datetime", default='/data1/liming', type=str,
                        help="resume file name.")
    parser.add_argument('--dataset_pos', default='oos', help='OOS/SMIPS/others')
    parser.add_argument('--dataset_neg', default='squad', help='OOS/SMIPS/others')
    parser.add_argument('--model', default='bert',
                        help='model: bert')
    # parser.add_argument('--method', default='DCLOOS',
    #                     help='baseline/DCLOOS')
    parser.add_argument('--dl_large', action='store_true',
                        help='is using self built OOD data')
    parser.add_argument('--class_cts', action='store_true',
                        help='enable class_cts')
    parser.add_argument('--domain_cts', action='store_true',
                        help='enable domain_cts')
    # parser.add_argument('--n_in_domain', default=64, type=int,
    #                     help='in domian examples in a batch')

    parser.add_argument('--max_seq', default=30, type=int,
                        help='max length of a sents')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument("--save_results_path", type=str, default='/data1/liming/DCLOOS', help="the path to save results")

    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--bert_model", default="/pretrained_models/uncased_L-12_H-768_A-12", type=str,
                        help="The path for the pre-trained bert model.")

    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--beta", default=0.1, type=float, help="beta for cts loss")
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    parser.add_argument('--num_convex', type=int, default=23, help="random seed for initialization")
    parser.add_argument('--num_convex_val', type=int, default=23, help="random seed for initialization")
    # parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    # parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT")

    parser.add_argument("--save_model", default=False, type=bool, help="save trained-model")
    #
    parser.add_argument("--save_results", default=False, type=bool, help="save test results")
    parser.add_argument("--loss_ce_only", action='store_true', help="save test results")
    parser.add_argument("--know_only", action='store_true', help="save test results")
    parser.add_argument("--resume", action='store_true', help="load from path")

    parser.add_argument("--known_cls_ratio", default=1.0, type=float,
                        help="The number of known classes")

    parser.add_argument("--labeled_ratio", default=1.0, type=float,
                        help="The ratio of labeled samples in the training set")

    parser.add_argument("--method", type=str, default='SupCon', help="which method to use")

    parser.add_argument('--seed', type=int, default=888, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--lr", default=2e-5, type=float,
                        help="The learning rate of BERT.")
    parser.add_argument("--lr_mlp", type=float, default=1e-4, help="The learning rate of the decision boundary.")

    parser.add_argument("--num_train_epochs", default=300, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument('--n_oos', default=64, type=int,
                        help='oos numbers in a batch')

    parser.add_argument("--eval_batch_size", default=100, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=10, type=int,
                        help="Patient steps for Early Stop.")
    parser.add_argument('--temp', type=float, default=0.2,
                        help='temperature for loss function')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', default=5, type=int, help='Save frequency')
    parser.add_argument('--patient', type=int, default=15, help="random seed for initialization")

    if script == 'train':

        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=-1, type=int,
                            help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        #parser.add_argument('--resume', default=True, type=bool,
         #                   help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup', action='store_true',
                            help='continue from baseline, neglected if resume is true')  # never used in the paper
    elif script == 'test':
        parser.add_argument('--test_num', default=-1, type=int,
                            help='saved feature from the model trained in x epoch, use the best model if x is -1')
        # parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        print('loading success')
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
