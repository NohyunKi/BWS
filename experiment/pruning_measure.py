from collections import defaultdict
import datetime
import csv
import time

import multiprocessing 

import numpy as np

import torch
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from base_trainer import BaseTrainer
import utils.args

class Pruning(BaseTrainer):
    def __init__(self, args, GPU_num, pruning, lock):
        self.args = args
        self.device = f'cuda:{GPU_num}'
        self.lock = lock
        self.pruning = pruning
        super().__init__()

        if self.args.noise:
            class NoisyDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, ratio, label_size, noise_label=None):
                    self.dataset = dataset
                    self.data_num = len(self.dataset)
                    self.noise_num = int(self.data_num*ratio)
                    self.label_size = label_size

                    if noise_label is None:
                        self.noise = np.random.randint((self.label_size-1), size=self.noise_num)
                    else:
                        self.noise = noise_label

                    self.targets = [(label+self.noise[i])%self.label_size if i < self.noise_num else label
                                        for i, label in enumerate(dataset.targets)]
                    
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    if idx < self.noise_num:
                        label = (label+self.noise[idx])%self.label_size
                    return img, label
                
                def __len__(self):
                    return self.data_num

                def get_noise_label(self):
                    return self.noise
            
            noise = np.load("data/noise_CIFAR10_0.4.npy")
            self.train_dataset = NoisyDataset(self.train_dataset, 0.4, 10, noise_label=noise)

        if self.args.measure == "lcmat":
            pruning_idx = np.around(np.load(
                f"data/lcmat_idx_{self.args.dataset}{('_noise_0.4' if self.args.noise else '')}.npz", allow_pickle=True)["idx_set"][()][self.pruning]).astype(int)
        elif self.args.measure == "adacore":
            pruning_idx = np.around(np.load(
                f"data/adacore_idx_{self.args.dataset}{('_noise_0.4' if self.args.noise else '')}.npz", allow_pickle=True)["idx_set"][()][self.pruning]).astype(int)
        elif self.args.measure == "CCS":
            measure = self._load_measure()()
            
            labels = defaultdict(list)
            for i in range(len(self.train_dataset)):
                labels[int(self.train_dataset.targets[i])].append(i)

            pruning_idx = []
            for _, idx in labels.items():
                self.beta = 0
                label_measure = measure[idx]
                label_num = len(label_measure)
                label_measure += np.random.normal(0, 0.0001, len(label_measure))
                label_measure_rank = label_measure.argsort()

                s_measure = label_measure[label_measure_rank[0]]
                f_measure = label_measure[label_measure_rank[round((1-self.beta)*label_num)-1]] + 0.0001
                bins = np.linspace(s_measure, f_measure, num=51)

                bins_idx = []
                bins_num_list = []
                for i in range(50):
                    bins_idx.append(
                        ((bins[i] <= label_measure) &
                        (label_measure < bins[i+1])).nonzero()[0]
                    )
                    bins_num_list.append(len(bins_idx[i]))
                bins_num_list = np.array(bins_num_list)

                bins_num_sort = bins_num_list.argsort()
                max_min = label_num * self.pruning / 50
                for i, idx_ in enumerate(bins_num_sort):
                    if max_min > bins_num_list[idx_]:
                        max_min = (max_min*(50-i) - bins_num_list[idx_]) / (49-i)
                    else:
                        bins_num_list[idx_] = round(max_min)
                        max_min = (max_min*(50-i) - round(max_min)) / (49-i) if i != 49 else 0
                
                for i, idices in enumerate(bins_idx):
                    pruning_idx += list(np.array(idx)[np.random.choice(idices, bins_num_list[i], replace=False)])
                    
        
        else:
            measure = self._load_measure()()

            labels = defaultdict(list)
            for i in range(len(self.train_dataset)):
                labels[int(self.train_dataset.targets[i])].append(i)
            
            pruning_idx = []
            for _, idx in labels.items():
                label_measure = measure[idx]
                label_measure += np.random.normal(0, 0.0001, len(label_measure))
                label_measure_rank = label_measure.argsort()

                if self.args.measure == "mds":
                    thres_idx_s = round(len(label_measure)*(0.5+(self.pruning/2)))
                    thres_idx_f = thres_idx_s - round(len(label_measure)*(self.pruning))

                    pruning_class_idx = (
                        (label_measure[label_measure_rank[thres_idx_f]] <= label_measure) &
                        (label_measure < label_measure[label_measure_rank[thres_idx_s]])
                        ).nonzero()[0]
                else:
                    thres_idx = round(len(label_measure)*(1-self.pruning))
                    pruning_class_idx = (
                        label_measure >= label_measure[label_measure_rank[thres_idx]]
                        ).nonzero()[0]

                pruning_idx += list(np.array(idx)[pruning_class_idx])
        
        if self.args.noise:
            self.noise_rate = ((np.array(pruning_idx) < self.train_dataset.noise_num).sum())/len(pruning_idx) 
            print(len(pruning_idx), self.noise_rate)
        else:
            print(len(pruning_idx))

        data_num = len(self.train_dataset)

        self.train_dataset = torch.utils.data.Subset(self.train_dataset, pruning_idx)
        if self.args.no_iter:
            sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=False)
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=data_num)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        sampler=sampler,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        drop_last=True)


    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        # Set device and Fix seed
        utils.train.fix_seed(self.args.seed)

        self.max_test_acc = 0
        self.max_epoch = 0

    def _load_measure(self):
        if self.args.dataset == "ImageNet":
            if self.args.measure in ["forgetting", "EL2N", "CCS", "SSL", "mem"]:
                measure = np.zeros(len(self.train_dataset))

                name = {"forgetting": "forgetting", "EL2N": "EL2N-1-model", "CCS": "forgetting", "SSL": "self-supervised-prototypes", "mem": "memorization"}

                with open(f'data/ImageNet-1K_{name[self.args.measure]}.csv', newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        measure[self.train_dataset.name_to_idx[row[1]]] = float(row[2])
                print(f"load measure {self.args.measure}")
                return lambda: measure
            elif self.args.measure == "random":
                return lambda: np.random.rand(len(self.train_dataset))
            else:
                raise

        measure_file_name = ""
        measure_file_name += self.args.dataset
        measure_file_name += "_" + (self.args.model if self.args.model not in ["vit_timm", "cnn", "eff"] else "resnet18")
        measure_file_name += "_0_40.0" if self.args.noise else ""
        print("noise_40")

        return {
            "EL2N": lambda: np.load(
                f"data/measures_{measure_file_name}_simple.npz", allow_pickle=True
                )['EL2N'],
            "forgetting": lambda: np.load(
                f"data/measures_{measure_file_name}_simple.npz", allow_pickle=True
                )['forgetting'],
            "CCS": lambda: np.load(
                f"data/measures_{measure_file_name}_simple.npz", allow_pickle=True
                )['forgetting'],
            "mds": lambda: -1 * np.load(
                f'data/mds_{self.args.dataset}_'+
                f'{(self.args.model if self.args.model not in ["vit_timm", "cnn", "eff"] else "resnet18")}'+
                f'_0{"_noise_40.0" if self.args.noise else ""}.npy'),
            "random": lambda: np.random.rand(len(self.train_dataset))
        }.get(self.args.measure, lambda: print(self.args.measure))
    
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
            if (epoch*self.pruning*5).is_integer() or epoch == self.args.epochs:
                test_acc_list = self._evaluate_single_epoch()
                self._print_and_write(epoch, test_acc_list, mode='test')

            if self.args.model_save:
                self.save_model(epoch)

            # record time
            self.writer.add_scalar("Time", time.time()-t_epoch, epoch)
        
        print("Total Learning time: {:2f}s".format(time.time() - t_training))
        self._train_summary()

    
    def _additional_parser(self, parser):
        parser.add_argument("--measure", type=str, default="random")
        parser.add_argument("--noise", action='store_true', default=False)
        parser.add_argument("--no_iter", action='store_true', default=False)
        
        return parser

    def _train_summary(self):
        super()._train_summary()

        no_iter = "_no_iter" if self.args.no_iter else ""
        noise = "_noise40" if self.args.noise else ""

        if not self.args.noise:
            with open(f'results_text/pruning_measure_{self.args.dataset}_{self.args.model}{no_iter}{rev}{noise}.txt', 'a') as f:
                f.write(f"{self.args.measure},{self.pruning}: {self.max_test_acc:.2f}\{self.curr_test_acc:.2f}\n")
        else:
            with open(f'results_text/pruning_measure_{self.args.dataset}_{self.args.model}{no_iter}{rev}{noise}.txt', 'a') as f:
                f.write(f"{self.args.measure},{self.pruning}: {self.max_test_acc:.2f}\{self.curr_test_acc:.2f}\{self.noise_rate:.3f}\n")


def _additional_parser(parser):
    parser.add_argument("--measure", type=str, default="random")
    parser.add_argument("--noise", action='store_true', default=False)
    parser.add_argument("--no_iter", action='store_true', default=False)
    parser.add_argument("--para_gpu", type=int, default=0)
    parser.add_argument("--window", type=float, default=0)
    
    return parser

window_list = None

def get_window(window_list, lock):
    with lock:
        if window_list: # not empty
            window = window_list.pop(0)
        else:
            window = None
    return window

def main():
    global window_list
    args = utils.args.get_args(_additional_parser)
    print(args)
    if args.model == "vit_timm":
        window_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9]
    elif args.noise:
        window_list = [0.1, 0.2, 0.3, 0.4]
    else:
        window_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9][::-1]

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    window_list = manager.list(window_list)

    def worker(GPU_num, window_list, lock, args=args):
        while True:
            window = get_window(window_list, lock)
            print(window, window_list)
            if window is None:
                break

            trainer = Pruning(args, GPU_num, window, lock)
            trainer.train()
    
    procs = []
    for i in range(2):
        proc = multiprocessing.Process(target=worker, args=(i+args.para_gpu, window_list, lock))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

if __name__ == '__main__':
    main()
