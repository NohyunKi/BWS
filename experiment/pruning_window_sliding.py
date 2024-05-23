from collections import defaultdict
import datetime
import time
import csv

import multiprocessing 

import numpy as np

import torch

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from base_trainer import BaseTrainer
import utils.args

class Pruning(BaseTrainer):
    def __init__(self, args, GPU_num, pruning, window_s, lock):
        self.args = args
        self.device = f'cuda:{GPU_num}'
        self.lock = lock
        self.pruning = pruning
        self.window_s = window_s
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
                    
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    if idx < self.noise_num:
                        label = (label+self.noise[idx])%self.label_size
                    return img, label
                
                def __len__(self):
                    return self.data_num

                def get_noise_label(self):
                    return self.noise
            
            print("noise 0.4")
            noise = np.load(f"data/noise_{self.args.dataset}_0.4.npy")
            self.train_dataset = NoisyDataset(self.train_dataset, 0.4, self.label_size, noise_label=noise)

        measure = self._load_measure()()

        labels = defaultdict(list)
        for i in range(len(self.train_dataset)):
            labels[int(self.train_dataset[i][1])].append(i)
        
        np.random.seed(42)
        measure += np.random.normal(0, 0.0001, len(measure))
        np.random.seed(self.args.seed)

        pruning_idx = []
        for label, idx in labels.items():
            label_measure = measure[idx]
            label_measure_rank = label_measure.argsort()

            thres_idx_s = round(len(label_measure)*(1-window_s))
            thres_idx_f = round(len(label_measure)*(1-window_s-self.pruning))
            
            if thres_idx_s != len(label_measure):
                window_list = (
                    (label_measure[label_measure_rank[thres_idx_f]] <= label_measure) &
                    (label_measure < label_measure[label_measure_rank[thres_idx_s]])
                    ).nonzero()[0]
            else:
                window_list = (
                    label_measure[label_measure_rank[thres_idx_f]] <= label_measure
                    ).nonzero()[0]
            
            pruning_idx += list(np.array(idx)[window_list])

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
                                                        num_workers=8,
                                                        drop_last=True)

    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        # Set device and Fix seed
        utils.train.fix_seed(self.args.seed)

        self.max_test_acc = 0
        self.max_epoch = 0

    def _load_measure(self):
        if self.args.dataset == "ImageNet":
            if self.args.measure == "forgetting":
                forgetting_value = np.zeros(len(self.train_dataset))

                with open('data/ImageNet-1K_forgetting.csv', newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        forgetting_value[self.train_dataset.name_to_idx[row[1]]] = int(row[2])
                print("load measure")
                return lambda: forgetting_value
            elif self.args.measure == "random":
                return lambda: np.random.rand(len(self.train_dataset))
            else:
                raise

        measure_file_name = ""
        measure_file_name += self.args.dataset
        measure_file_name += "_" + (self.args.model if self.args.model not in ["cnn", "vit_timm", "eff", "convnet"] else "resnet18")
        measure_file_name += "_noise_40.0" if self.args.noise else ""
        # print("noise 0.4")

        return {
            "EL2N": lambda: np.load(
                f"data/measures_{measure_file_name}_simple.npz", allow_pickle=True
                )['EL2N'],
            "forgetting": lambda: np.load(
                f"data/measures_{measure_file_name}_simple.npz", allow_pickle=True
                )['forgetting'],
        }.get(self.args.measure, lambda: np.random.rand(len(self.train_dataset)))

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
            if not self.args.no_iter:
                test_acc_list = self._evaluate_single_epoch()
                self._print_and_write(epoch, test_acc_list, mode='test')
                
            elif self.args.no_iter \
                and ((epoch*self.pruning*5).is_integer() or epoch == self.args.epochs):
                
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

        with open(f'results_text/pruning_sliding_{self.args.dataset}_{self.args.model}{no_iter}.txt', 'a') as f:
            f.write(f"{self.args.measure},{self.pruning},{self.window_s}: {self.max_test_acc:.2f}\{self.curr_test_acc:.2f}\n")

def _additional_parser(parser):
    parser.add_argument("--measure", type=str, default="random")
    parser.add_argument("--noise", action='store_true', default=False)
    parser.add_argument("--no_iter", action='store_true', default=False)
    
    return parser

window_list = None

def get_pruning_window(window_list, lock):
    with lock:
        if window_list: # not empty
            window = window_list.pop(0)
        else:
            window = None
            return None, None

    pruning = int(window) / 100.
    window = window - (pruning * 100)
    return pruning, window

def main():
    global window_list
    args = utils.args.get_args(_additional_parser)
    print(args)

    window_list = []
    if args.model == "vit_timm":
        list_ = [1, 5, 10, 20]
    elif args.noise:
        list_ = [10, 20, 30, 40]
    else:
        list_ = [1, 5, 10, 20, 30, 40, 50, 75, 90]

    for i in list_:
        window_list += np.arange((i), (i+1.0001-(i/100)), 0.05).tolist()
        if i % 5 != 0:
            window_list += [i+1-(i/100)]
    print(len(window_list))

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    window_list = manager.list(window_list)

    def worker(GPU_num, window_list, lock, args=args):
        while True:
            pruning, window = get_pruning_window(window_list, lock)
            print(pruning, window, window_list)
            if window is None:
                break

            trainer = Pruning(args, GPU_num, pruning, window, lock)
            trainer.train()
    
    procs = []
    for i in range(12):
        proc = multiprocessing.Process(target=worker, args=(i%6, window_list, lock))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

if __name__ == '__main__':
    main()
