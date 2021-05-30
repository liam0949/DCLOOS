from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from utils import *
from losses import SupConLossD, SupConLossC
import configs
import random
import numpy as np
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from models.Encoder import *
import dataloader as dllarge
#import dataloader_in as dlin
from sklearn.metrics import f1_score
import datetime

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def set_model(opt):
    opt.in_dim = opt.feat_dim
    model = Encoder(opt)
    criterionD = SupConLossD(temperature=opt.temp)
    criterionC = SupConLossC(temperature=opt.temp)
    criterion = [criterionD.cuda(), criterionC.cuda()]
    loss_ce = nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, loss_ce


def train(dataset, model, criterion, loss_ce, optimizer_bert, optimizer_mlp, scheduler_bert, epoch, opt,
          annealing_kl=None):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_ce = AverageMeter()
    losses_sct = AverageMeter()
    losses = AverageMeter()
    laccs = AverageMeter()

    end = time.time()
    for idx, (pst, ngt) in enumerate(zip(dataset.train_dataloader, dataset.neg_dataloader)):
        # for idx, pst in enumerate(dataset.train_dataloader):
        # breaker += 1
        global_steps = epoch * opt.steps_per_epoch + idx
        if annealing_kl is not None:
            anneal_kl_this_step = annealing_kl[global_steps]
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            # pst = pst.cuda(non_blocking=True)
            # ngt = ngt.cuda(non_blocking=True)
            # if not opt.know_only:
            input_ids = torch.cat([pst[0], ngt[0]], dim=0).cuda()

            input_mask = torch.cat([pst[1], ngt[1]], dim=0).cuda()

            label_ids = torch.cat([pst[3], ngt[3]], dim=0).cuda()

            bsz = label_ids.shape[0]
            # print(label_ids)
            # break
            # print(label_ids)
            # else:
            #     input_ids = pst[0].cuda()
            #
            #     input_mask = pst[1].cuda()
            #
            #     label_ids = pst[3].cuda()
            #     # print(label_ids)
            #
            #     bsz = label_ids.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        logits, mean, log_var, label_ids = model(input_ids, input_mask, label_ids, mode="train")

        # print(label_ids)

        loss_ce_l = loss_ce(torch.div(logits, opt.temp), label_ids)
        # if opt.method == 'SupCon' and not opt.know_only:
        #     loss_sct = criterion(features=features, args=opt, labels=label_ids)
        #     # loss_sct = torch.tensor([0]).cuda()
        # elif opt.method == 'SimCLR':
        #     loss = criterion(features)
        # else:
        #     raise ValueError('contrastive method not supported: {}'.
        #                      format(opt.method))

        # update metric
        if opt.loss_ce_only or opt.know_only:
            #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)
            # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)
            #if annealing_kl is not None:
                # kld_loss = anneal_kl_this_step * kld_loss
               # kld_loss = 0.0 * kld_loss
           # loss_all = loss_ce_l + 0.01*kld_loss
           loss_all = loss_ce_l

        #elif opt.class_cts or opt.domain_cts:
         #   loss_sct = 0.0
         #   if opt.class_cts:
         #       loss_sct = loss_sct + criterion[1](features=features, args=opt, labels=label_ids)
         #   if opt.domain_cts:
         #       loss_sct = loss_sct + criterion[0](features=features, args=opt, labels=label_ids)
          #  loss_all = (1 - opt.beta) * loss_ce_l + opt.beta * loss_sct
          #  losses_sct.update(loss_sct.item(), bsz)

        losses_ce.update(loss_ce_l.item(), bsz)
        losses.update(loss_all.item(), bsz)
        tb_acc = accuracy(logits, label_ids)
        laccs.update(tb_acc.item(), bsz)

        # admw and adm
        optimizer_bert.zero_grad()
        optimizer_mlp.zero_grad()
        loss_all.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_bert.step()
        optimizer_mlp.step()
        # scheduler_bert.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss_ce {loss_ce.val:.3f} ({loss_ce.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, idx + 1, len(dataset.train_dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_ce=losses_ce, acc=laccs))
            print(anneal_kl_this_step)
            sys.stdout.flush()

        # if breaker == 1:
        #     break
    # break

    return losses.avg, laccs.avg


def evaluation(args, model, data, mode="eval"):
    model.eval()
    with torch.no_grad():
        acc_avg = AverageMeter()
        acc_oos_avg = AverageMeter()
        f1_avg = AverageMeter()
        if mode == 'eval':
            for x in data.eval_dataloader:
                if torch.cuda.is_available():
                    # pst = pst.cuda(non_blocking=True)
                    # ngt = ngt.cuda(non_blocking=True)
                    input_ids = x[0].cuda()

                    input_mask = x[1].cuda()

                    label_ids = x[3].cuda()

                    bsz = label_ids.shape[0]

                # warm-up learning rate
                # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
                oos_ids = label_ids.view(-1).eq(torch.tensor(data.num_labels))

                # print(label_ids[oos_ids])
                # compute loss

                if oos_ids.sum() != 0:

                    logits_oos, _, _, label_ids_oos = model(input_ids[oos_ids], input_mask[oos_ids], label_ids[oos_ids],
                                                            mode=None)
                    acc_oos = accuracy(logits_oos, label_ids_oos)
                    acc_oos_avg.update(acc_oos, n=oos_ids.sum().item())
                    logits_all = logits_oos
                    label_ids_all = label_ids_oos
                    if oos_ids.sum() != bsz:
                        logits_in, _, _, label_ids_in = model(input_ids[~oos_ids], input_mask[~oos_ids],
                                                              label_ids[~oos_ids], mode='val')
                        logits_all = torch.cat([logits_in, logits_oos], dim=0)
                        label_ids_all = torch.cat([label_ids_in, label_ids_oos], dim=0)

                else:
                    logits_all, _, _, label_ids_all = model(input_ids, input_mask, label_ids, mode="val")
                acc_all = accuracy(logits_all, label_ids_all)
                acc_avg.update(acc_all, n=label_ids_all.shape[0])
            return acc_avg.avg, acc_oos_avg.avg

        elif mode == 'test':
            labels = []
            preds = []
            for x in data.test_dataloader:
                if torch.cuda.is_available():
                    # pst = pst.cuda(non_blocking=True)
                    # ngt = ngt.cuda(non_blocking=True)
                    input_ids = x[0].cuda()

                    input_mask = x[1].cuda()

                    label_ids = x[3].cuda()

                    bsz = label_ids.shape[0]

                # warm-up learning rate
                # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

                # compute loss
                oos_ids = label_ids.view(-1).eq(torch.tensor(data.num_labels))

                # compute loss
                logits, _, _, _ = model(input_ids, input_mask, None)
                if oos_ids.sum() != 0:
                    acc_oos = accuracy(logits[oos_ids], label_ids[oos_ids])
                    acc_oos_avg.update(acc_oos, n=oos_ids.sum().item())
                acc_all = accuracy(logits, label_ids)
                acc_avg.update(acc_all, n=bsz)

                _, pred = logits.topk(1, 1, True, True)
                labels.append(label_ids.cpu().numpy())
                preds.append(pred.cpu().numpy())

            f1_scores = f1_score(np.concatenate(labels, axis=0), np.concatenate(preds, axis=0), average=None)
            # f1_scores_oos = f1_score(np.concatenate(labels_oss, axis=0), np.concatenate(preds_oos, axis=0),
            #                          average=None)

            return acc_avg.avg, acc_oos_avg.avg, f1_scores


def main():
    opt = configs.parse_args("train")

    # check if dataset is path that passed required arguments

    # set the path according to the environment
    opt.model_path = opt.save_results_path + '/SupCon/{}_models'.format(opt.dataset_pos)
    opt.tb_path = opt.save_results_path + '/SupCon/{}_tensorboard'.format(opt.dataset_pos)

    currentDT = datetime.datetime.now()

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_ce_{}_know_{}_know_only_{}_beta_{}_dl_large{}'. \
        format(opt.dataset_pos, opt.dataset_neg, opt.model, opt.lr,
               '0', opt.train_batch_size, opt.temp, opt.datetime, str(opt.loss_ce_only),
               str(opt.known_cls_ratio), str(opt.know_only), str(opt.beta), str(opt.dl_large))

    print(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    def set_loader(opt):
        pass

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # build data loader
    if opt.dl_large:
        dataset = dllarge.Data(opt)
        # print(dataset.label_list)
        # exit(1)

    else:
        dataset = dlin.Data(opt)
    opt.n_way = len(dataset.label_list)
    opt.oos_label_id = dataset.num_labels
    # build model and criterion
    model, criterion, loss_ce = set_model(opt)

    # build optimizer
    optimizer_bert, optimizer_mlp = set_optimizer(opt, model)
    opt.steps_per_epoch = len(dataset.train_dataloader)

    total_steps = opt.steps_per_epoch * opt.num_train_epochs
    annealing_kl = frange_cycle_linear(total_steps)
    scheduler_bert = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer_bert,
                                                                        num_warmup_steps=0,
                                                                        # Default value in run_glue.py
                                                                        num_training_steps=total_steps, num_cycles=6)
    start_epoch = 1
    if opt.resume:
        resume_file = get_resume_file(opt.save_folder, opt.resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['model'])
            print("loaded from " + resume_file)
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    max_acc = 0
    train_log = {}
    train_log["test_acc"] = []
    train_log["val_acc"] = []
    train_log["f1_score"] = []
    patient = 0
    # training routine
    for epoch in range(start_epoch, opt.num_train_epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)
        # opt.epoch_step = epoch
        # train for one epoch
        time1 = time.time()
        loss, t_acc = train(dataset, model, criterion, loss_ce, optimizer_bert, optimizer_mlp, scheduler_bert, epoch,
                            opt, annealing_kl=annealing_kl)
        time2 = time.time()
        print('epoch {}, total train time {:.2f}'.format(epoch, time2 - time1))

        # eval for one epoch
        time1 = time.time()
        val_acc, val_oos_acc = evaluation(opt, model, dataset, mode="eval")
        time2 = time.time()
        print('epoch {}, total val time {:.2f}, val acc: {:.3f}, val oss acc: {:.3f}'.format(epoch, time2 - time1,
                                                                                             val_acc.item(),
                                                                                             val_oos_acc.item()))

        # test for best val acc
        if val_acc > max_acc:
            time1 = time.time()
            test_acc, test_oos_acc, f1_scores = evaluation(opt, model, dataset, mode="test")
            time2 = time.time()
            # train_log["test_acc"].append(test_acc)
            # train_log["val_acc"].append(val_acc)
            # train_log["f1_score"].append(val_acc)
            # torch.save(train_log, os.path.join(opt.save_folder, 'trlog'))
            print(
                'epoch {}, total test time {:.2f}, test acc: {:.3f}, test oss acc: {:.3f}'.format(epoch, time2 - time1,
                                                                                                  test_acc.item(),
                                                                                                  test_oos_acc.item()))
            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_oos_acc', test_oos_acc, epoch)
            logger.log_value('f1_scores_mean', np.mean(f1_scores), epoch)
            logger.log_value('f1_scores_oos', f1_scores[-1], epoch)
            logger.log_value('f1_known', np.mean(f1_scores[:-1]), epoch)
            save_file = os.path.join(
                # opt.save_folder, 'best_model_{}.pth'.format(epoch))
                opt.save_folder, 'best_model.pth'.format(epoch))
           # save_model(model, optimizer_bert, opt, epoch, save_file)
            max_acc = val_acc
            patient=0
        else:
            patient += 1
        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', t_acc, epoch)
        logger.log_value('val_acc', val_acc, epoch)
        logger.log_value('val_oos_acc', val_oos_acc, epoch)
        logger.log_value('learning_rate', optimizer_bert.param_groups[0]['lr'], epoch)

       # if epoch % opt.save_freq == 0:
        #    save_file = os.path.join(
        #        opt.save_folder, 'ckpt.pth')
        #    save_model(model, optimizer_bert, opt, epoch, save_file)
        if patient > opt.patient:
            break

    # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer_bert, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
