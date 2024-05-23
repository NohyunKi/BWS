from pathlib import Path
import datetime

import torch
import utils.args
import utils.train

import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.load_data import load_data
from utils.train import CosineAnnealingWarmUpRestarts, accuracy

from model.timm_vit import TimmViT
from model.resnet import ResNet18, ResNet34, ResNet50, ImageResNet18, ImageResNet50
from model.fc import BasicFC
from model.cnn import BasicCNN
from model.densenet import DenseNetBC_100_12, DenseNetBC_190_40
from model.efficientnet import EfficientNetB0
from model.convnet import ConvNet

class BaseTrainer:
    def __init__(self):
        ## initialize
        self._pre_setting()
        self._load_data()
        self._get_model()
        self._get_optimizer_and_scheduler()
        self._get_logger()

    def _additional_parser(self, parser):
        return parser

    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.args = utils.args.get_args(self._additional_parser)
        print(self.args)

        # Set device and Fix seed
        self.device = 'cuda' if (torch.cuda.is_available() and not self.args.no_cuda) else 'cpu'
        utils.train.fix_seed(self.args.seed)

        self.max_test_acc = 0
        self.max_epoch = 0

    def _load_data(self):
        is_timm = False

        if self.args.model == "vit_timm":
            is_timm = True

        dataset_list, loader_list, size_list = \
            load_data(self.args.dataset, self.args.batch_size, is_timm=is_timm)

        self.train_dataset, self.test_dataset, self.eval_dataset = dataset_list
        self.train_loader, self.test_loader = loader_list
        self.input_size, self.label_size = size_list

    def _get_model(self):
        first_conv_stride = 1 if self.args.dataset != "TinyImageNet" else 2

        if self.args.dataset == "ImageNet":
            if self.args.model == "resnet18":
                model = ImageResNet18(in_channels=self.input_size[-1], 
                                    num_classes=self.label_size).to(self.device)
            elif self.args.model == "resnet50":
                model = ImageResNet50(in_channels=self.input_size[-1], 
                                    num_classes=self.label_size).to(self.device)
            
            self.model = model
            return 

        if self.args.model == "resnet18":
            model = ResNet18(in_channels=self.input_size[-1], 
                             num_classes=self.label_size,
                             first_conv_stride=first_conv_stride).to(self.device)
        elif self.args.model == "resnet34":
            model = ResNet34(in_channels=self.input_size[-1], 
                             num_classes=self.label_size,
                             first_conv_stride=first_conv_stride).to(self.device)
        elif self.args.model == "resnet50":
            model = ResNet50(in_channels=self.input_size[-1], 
                             num_classes=self.label_size,
                             first_conv_stride=first_conv_stride).to(self.device)
        elif self.args.model == "densenet100":
            model = DenseNetBC_100_12(
                in_channels=self.input_size[-1], num_classes=self.label_size).to(self.device)
        elif self.args.model == "densenet190":
            model = DenseNetBC_190_40(
                in_channels=self.input_size[-1], num_classes=self.label_size).to(self.device)
        elif self.args.model == "cnn":
            model = BasicCNN(self.input_size, self.label_size).to(self.device)
        elif self.args.model == "fc":
            model = BasicFC(self.input_size, self.label_size).to(self.device)
        elif self.args.model == "vit_timm":
            model = TimmViT(pretrained=True, num_classes=self.label_size).to(self.device)
        elif self.args.model == "eff":
            model = EfficientNetB0(num_classes=self.label_size).to(self.device)
        elif self.args.model == "convnet":
            model = ConvNet(in_channels=self.input_size[-1], num_classes=self.label_size).to(self.device)
        else:
            raise NotImplementedError

        self.model = model

    def _get_optimizer_and_scheduler(self):
        self.CE_loss = nn.CrossEntropyLoss()
        model_params = self.model.parameters()
        
        # learning rate 
        if self.args.scheduler == "CosineOurs":
            lr = self.args.lr*0.001
        else:
            lr = self.args.lr

        # optimizer
        if self.args.optim == "SGD":
            optimizer = optim.SGD(
                model_params, 
                lr=lr, 
                weight_decay=self.args.regularizer
            )
        elif self.args.optim == "Momentum":
            optimizer = optim.SGD(
                model_params,
                lr=lr, 
                weight_decay=self.args.regularizer, 
                momentum=0.9
            )
        elif self.args.optim == "ADAM":
            optimizer = optim.Adam(
                model_params, 
                lr=lr, 
                weight_decay=self.args.regularizer
            )
        else:
            raise NotImplementedError

        #scheduler
        if self.args.scheduler == "CosineOurs":
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer, 
                T_0=self.args.epochs if self.args.epochs < 5000 else 200, 
                T_mult=1, 
                eta_max=self.args.lr, 
                T_up=5, 
                gamma=1
            )
        elif self.args.scheduler == "CosineTorch":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                eta_min=0
            )
        elif self.args.scheduler == "Cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.args.epochs
            )
        elif self.args.scheduler == "Step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 30, gamma = 0.1
            )
        elif self.args.scheduler == "MultiStep":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, [150, 225], gamma = 0.1
            )
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def _get_logger(self):
        self.log_dir = os.path.join(
            "test" if self.args.test_exp else "runs", 
            self.execute_time,
            self.args.dataset, 
            str(self.args.epochs)
        )
        if self.args.exp_name is not None:
            self.log_dir = os.path.join(self.log_dir, self.args.exp_name)

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(self.log_dir, 'args.txt'), 'w') as f:
            json.dump(vars(self.args), f, indent=4) #f.write(str(args))

        # figures
        self.fig_dir = os.path.join(self.log_dir, "figures")
        Path(self.fig_dir).mkdir(parents=True, exist_ok=True)
        
        # models
        self.models_dir = os.path.join(self.log_dir, "models")
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _loss(self, logit, label, index=None, epoch=None):
        return self.CE_loss(logit, label)

    def _train_single_epoch(self, epoch):
        self.model.train()

        train_total = 0
        train_correct = 0
        loader_len = len(self.train_loader)

        for j, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(data)

            prec, = accuracy(outputs, labels, topk=(1, ))
            train_total += labels.size(0)
            train_correct += prec

            loss = self._loss(outputs, labels, epoch)

            #backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if j % 500 == 499:
                print(f'Epoch [{epoch}/{self.args.epochs}][{j+1}/{loader_len}] train accuracy: {float(train_correct)/float(train_total):.2f}%, time: {time.time() - self.t_epoch}')

        self.scheduler.step()
       
        return float(train_correct)/float(train_total)
    
    def _evaluate_single_epoch(self):
        self.model.eval()
        
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(data)
                prec, = accuracy(logits, labels, topk=(1, ))
                total += labels.size(0)
                correct += prec
                        
        acc = float(correct) / float(total)
        return acc
    
    def _print_and_write(self, epoch, acc_list, mode='train', t_epoch=None):
        prt_str = f'Epoch [{epoch}/{self.args.epochs}] {mode} accuracy: {acc_list:.2f}%'
        if t_epoch is not None:
            elapsed_time = time.time()- t_epoch
            prt_str += f', time: {elapsed_time:.2f}s'
        print(prt_str)
        
        acc = acc_list
        self.writer.add_scalar(f"Accuracy/{mode}", acc, epoch)
        if mode == 'test':
            self.curr_test_acc = acc
            self.curr_epoch = epoch
            if acc > self.max_test_acc:
                self.max_test_acc = acc
                self.max_epoch = epoch

    def _train_summary(self):
        print(f"max/min:{self.max_test_acc:.2f}%({self.max_epoch})/{self.curr_test_acc:.2f}")
        print(self.args)

    def save_model(self, epoch):
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.models_dir, "model_" + str(epoch) + ".pt")
        )

    def train(self):
        # epoch starts from 1!
        t_training = time.time()

        for epoch in range(1, self.args.epochs+1):
            t_epoch = time.time()
            self.t_epoch = t_epoch
            # train accuracy
            train_acc_list = self._train_single_epoch(epoch)
            self._print_and_write(epoch, train_acc_list, t_epoch=t_epoch)

            # test accuracy
            test_acc_list = self._evaluate_single_epoch()
            self._print_and_write(epoch, test_acc_list, mode='test')

            if self.args.model_save:
                self.save_model(epoch)

            # record time
            self.writer.add_scalar("Time", time.time()-t_epoch, epoch)
        
        print("Total Learning time: {:2f}s".format(time.time() - t_training))
        self._train_summary()

class BaseTrainerEpoch(BaseTrainer):
    def __init__(self):
        super().__init__()
    
    def _train_single_epoch(self, epoch):
        self.model.train()

        train_total = 0
        train_correct = 0

        t_time = time.time()

        for j, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(data)

            prec, = accuracy(outputs, labels, topk=(1, ))
            train_total += labels.size(0)
            train_correct += prec

            loss = self._loss(outputs, labels, epoch)

            #backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (j+1) % 500 == 0:
                print(f"Epoch [{epoch}/{self.args.epochs}] Batch [{j+1}/{len(self.train_loader)}] : time: {time.time()-t_time:.2f}s")
        
        self.scheduler.step()
       
        return float(train_correct)/float(train_total)


def main():
    trainer = BaseTrainerEpoch()
    trainer.train()

if __name__ == '__main__':
    main()
