import torch
from config import Config
import os
import random
import glob
import shutil
from utils import utils
from dataset import dataset
from transforms import transforms
from evaluation.eval import auc_eval_func
from evaluation.evaluator import ALASKAEvaluator

from loss.loss import LabelSmoothing
import argparse

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
from pytorch_pfn_extras.training.triggers import MaxValueTrigger
import cv2
cv2.setNumThreads(0)

from apex import amp

def train_func(manager, model, criterion, optimizer, train_loader, device, scheduler=None, metric_learning=False,
               evaluator=None, val_loader=None, eval_trigger=None, fp16=True):
    while not manager.stop_trigger:
        model.train()
        for x, t in train_loader:
            with manager.run_iteration():
                x, t = x.to(device), t.to(device)
                optimizer.zero_grad()
                if metric_learning:
                    loss = criterion(model(x, t), t)
                else:
                    loss = criterion(model(x), t)
                ppe.reporting.report({'train/loss':loss.item()})
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--snapmodel', type=str)
    args = parser.parse_args()

    # get config
    config = Config()

    # set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # create model
    model = utils.create_model(config)
    device = 'cuda'
    model.cuda()

    # define transforms
    train_trans = transforms.train_transform(resize=(config.input_size_h, config.input_size_w),
                                             normalize=config.normalize)
    val_trans = transforms.eval_transform(resize=(config.input_size_h, config.input_size_w),
                                          normalize=config.normalize)

    # copy config and src
    if not os.path.exists(os.path.join(config.result, 'src')):
        os.makedirs(os.path.join(config.result, 'src'), exist_ok=True)
    for src_file in glob.glob('/work/*.py') + glob.glob('/work/*/*.py'):
        shutil.copy(src_file, os.path.join(config.result, 'src', os.path.basename(src_file)))

    # create dataset
    train_dataset = dataset.Alaska2Dataset(root=config.data, transforms=train_trans, train=True,
                                           batchsize=config.batchsize, uniform=config.batch_uniform,
                                           )
    val_dataset = dataset.Alaska2Dataset(root=config.data, transforms=val_trans, train=False,
                                         uniform=False)

    # create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.batchsize,
        num_workers = config.num_workers,
        shuffle = True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.batchsize,
        num_workers = config.num_workers,
        shuffle = False
    )

    # set optimizer
    #    optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params':metrics_fc.parameters()}],
    #                                 lr=config.lr
    #                                 )
    #else:
    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=config.lr
                                )
    #optimizer = torch.optim.SGD(model.parameters(),
    #                              lr=config.lr,
    #                            momentum=0.9)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if config.fp16:
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=opt_level
                                          #keep_batchnorm_fp32=True
                                          )

    # set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, threshold_mode='abs',
        min_lr=1e-8, eps=1e-08
    )

    # set criterion
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothing().cuda()
    num_epochs = config.num_epochs

    # set manager
    iters_per_epoch = len(train_loader)


    manager = ppe.training.ExtensionsManager(
        model, optimizer, num_epochs,
        iters_per_epoch=iters_per_epoch,
        out_dir=config.result,
        stop_trigger=None
    )

    log_interval = (100, 'iteration')
    #eval_interval = (500, 'iteration')
    eval_interval = (1, 'epoch')

    manager.extend(extensions.snapshot(filename='best_snapshot'), trigger=MaxValueTrigger('validation/auc', trigger=eval_interval))
    if config.fp16:
        manager.extend(extensions.snapshot_object(amp, filename='amp.ckpt'), trigger=MaxValueTrigger('validation/auc', trigger=eval_interval))

    manager.extend(extensions.LogReport(trigger=log_interval))

    manager.extend(extensions.PlotReport(['train/loss', 'validation/loss'],
                                         'epoch', filename='loss.png'), trigger=(1, 'epoch'))

    manager.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'train/loss', 'validation/loss', 'validation/auc', 'lr', 'elapsed_time']
    ), trigger=log_interval)
    

    manager.extend(extensions.ProgressBar(update_interval=100))
    manager.extend(extensions.observe_lr(optimizer=optimizer), trigger=log_interval)
    #manager.extend(extensions.ParnnameterStatistics(model, prefix='model'))
    #manager.extend(extensions.VariableStatisticsPlot(model))

    manager.extend(
        ALASKAEvaluator(
            val_loader,
            model,
            eval_hook=None, eval_func=None,
            loss_criterion=criterion,
            auc_criterion=auc_eval_func,
            device=device,
            scheduler=scheduler,
            metric_learning=config.metric_learning
        ),
        trigger=eval_interval
    )

    # Lets load the snapshot
    if args.snapshot is not None:
        state = torch.load(args.snapshot)
        manager.load_state_dict(state)
        #amp = torch.load('amp.ckpt')
    elif args.snapmodel is not None:
        print('load snapshot model {}'.format(args.snapmodel))
        state = torch.load(args.snapmodel)
        manager._models['main'].load_state_dict(state['models']['main'])

    train_func(manager, model, criterion, optimizer, train_loader, device, metric_learning=config.metric_learning,
               fp16=config.fp16)


if __name__ == '__main__':
    main()

