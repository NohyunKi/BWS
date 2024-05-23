import numpy as np
import torch
import os
import sys
import time
import datetime
from collections import defaultdict
import multiprocessing 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.train import accuracy
from torchvision.models.feature_extraction import create_feature_extractor

from base_trainer import BaseTrainer
from utils.train import accuracy
import utils

class SaveRegressionacc(BaseTrainer):
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
            self.eval_dataset = NoisyDataset(self.eval_dataset, 0.4, 10, noise_label=noise)

        self.eval_loader = torch.utils.data.DataLoader(dataset=self.eval_dataset,
                                batch_size=self.args.batch_size*4,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)

        self.labels = defaultdict(list)
        for i in range(len(self.train_dataset)):
            self.labels[int(self.train_dataset.targets[i])].append(i)
        self.label_set = torch.tensor(self.train_dataset.targets)

        if self.args.measure == "forgetting":
            self.measure = np.load(
                f"data/measures_{self.args.dataset}_{self.args.model}_simple.npz", allow_pickle=True
                )['forgetting']
        elif self.args.measure == "EL2N":
            self.measure = np.load(
                f"data/measures_{self.args.dataset}_{self.args.model}_simple.npz", allow_pickle=True
                )['EL2N']
        else:
            raise

        pruning_idx = []
        for _, idx in self.labels.items():
            pruning_idx += list(np.random.choice(idx, round(self.pruning*len(idx)), replace=False))
        print(len(pruning_idx))

        self.step_num = len(self.train_loader)

        self.train_dataset = torch.utils.data.Subset(self.train_dataset, pruning_idx)
        sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        sampler=sampler,
                                                        num_workers=4,
                                                        drop_last=True)

        self.check_epoch = [20]

    def _save_regression_acc(self, epoch):
        eval_loader = self.eval_loader

        if self.args.model in ["resnet18", "resnet50"]:
            layer = 'avgpool'
        elif self.args.model == "vit_timm":
            layer = 'model.head_drop'
        elif self.args.model == "eff":
            layer = "adaptive_avg_pool2d"
        else:
            layer = 'mlp_layers.1.3'
        
        self.model.eval()
        feature_model = create_feature_extractor(self.model, {layer: "result"})
        feature_model.eval()

        feature_list = []
        for i, (data, _) in enumerate(eval_loader):
            data = data.to(self.device)
            results = feature_model(data)
            feature_list.append(results["result"].flatten(1))
        feature_list = torch.cat(feature_list, 0)
        del feature_model

        one = torch.ones((len(feature_list), 1), device=self.device)
        feature_list = torch.cat((one, feature_list), dim=1)
        
        ratio = self.pruning

        acc_list = []
        for window in np.arange(0, 1.0001-self.pruning, 0.05).tolist():
            window_idx = self.get_window_idx(window)

            window_feature_set = feature_list[window_idx]
            window_label_set = self.label_set[window_idx]

            w_window = self.cal_w_regression(window_feature_set, window_label_set, 1)

            feature_idx = self.get_easy_idx() if self.args.noise else np.arange(feature_list.shape[0])

            acc = self.check_acc(w_window, feature_list[feature_idx], self.label_set[feature_idx])
            acc_list.append(acc)

        text = "_noise_40.0" if self.args.noise else "" 
        text += f"_{self.args.model}" if self.args.model not in ['resnet18', 'resnet50'] else ""
        
        np.save(f'data/threshold/{self.args.dataset}{text}_{self.args.measure}_epoch{epoch}_ratio{ratio}_acc_{self.args.seed}', acc_list)

    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

        # Set device and Fix seed
        utils.train.fix_seed(self.args.seed)

        self.max_test_acc = 0
        self.max_epoch = 0

    def cal_w_regression(self, featureset, labelset, lambda_):
        ######## x shape : (n, d), n is number of data and d is dimension of data ########
        x = featureset
        labelset = labelset.to(self.device)
        
        label_num = int(max(labelset)+1)
        w = []
        for label in range(label_num):
            y = torch.where(labelset == label, 1, 0).type(torch.float32)
            
            if x.shape[0] > x.shape[1]:
                I = torch.eye(x.shape[1], device=self.device)
                H = (x.mT@x + lambda_*I)
                invH = torch.inverse(H)
                w.append(invH@x.mT@y)
            else:
                I = torch.eye(x.shape[0], device=self.device)
                H = (x@x.mT + lambda_*I)
                invH = torch.inverse(H)
                w.append(x.mT@invH@y)

        w = torch.stack(w, 0)
        return w


    def check_acc(self, w, featureset, labelset):
        y = labelset.to(self.device)
        
        output = w@featureset.mT
        
        softmax_fun = torch.nn.Softmax(dim=1)
        s = softmax_fun(output.mT)
        
        prec, = accuracy(s, y, topk=(1,))
        acc = prec/len(featureset)
        return acc.item()

    def get_window_idx(self, window):
        idx_list = []
        
        for _, idx in self.labels.items():
            label_measure = self.measure[idx]
            label_measure += np.random.normal(0, 0.0001, len(label_measure))
            label_measure_rank = label_measure.argsort()

            thres_idx_s = round(len(label_measure)*(1-window))
            thres_idx_f = round(len(label_measure)*(1-window-self.pruning))
            
            if thres_idx_s != len(label_measure):
                window_list = (
                    (label_measure[label_measure_rank[thres_idx_f]] <= label_measure) &
                    (label_measure < label_measure[label_measure_rank[thres_idx_s]])
                    ).nonzero()[0]
            else:
                window_list = (
                    label_measure[label_measure_rank[thres_idx_f]] <= label_measure
                    ).nonzero()[0]
            
            idx_list += list(np.array(idx)[window_list])
        return idx_list
    
    def get_easy_idx(self):
        idx_list = []
        
        for _, idx in self.labels.items():
            label_measure = self.measure[idx]
            label_measure += np.random.normal(0, 0.0001, len(label_measure))
            label_measure_rank = label_measure.argsort()

            thres_idx_s = round(len(label_measure)*(0.5))

            window_list = (label_measure <= label_measure[label_measure_rank[thres_idx_s]]).nonzero()[0]

            idx_list += list(np.array(idx)[window_list])
        return idx_list

    def train(self):
        # epoch starts from 1!
        t_training = time.time()

        for epoch in range(1, 21):
            t_epoch = time.time()
            self.t_epoch = t_epoch

            # train accuracy
            train_acc_list = self._train_single_epoch(epoch)
            self._print_and_write(epoch, train_acc_list, t_epoch=t_epoch)

            if self.args.model_save:
                self.save_model(epoch)

            if epoch in self.check_epoch:
                with torch.no_grad():
                    self._save_regression_acc(epoch)

            # record time
            self.writer.add_scalar("Time", time.time()-t_epoch, epoch)
        
        print("Total Learning time: {:2f}s".format(time.time() - t_training))

    def _additional_parser(self, parser):
        parser.add_argument("--no_iter", action='store_true', default=False)
        parser.add_argument("--noise", action='store_true', default=False)
        parser.add_argument("--measure", type=str, default="forgetting")
        
        return parser

def _additional_parser(parser):  
    parser.add_argument("--no_iter", action='store_true', default=False)
    parser.add_argument("--noise", action='store_true', default=False)
    parser.add_argument("--measure", type=str, default="forgetting")
    parser.add_argument("--para_gpu", type=int, default=0)
    
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
    window_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9]

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    window_list = manager.list(window_list)

    def worker(GPU_num, window_list, lock, args=args):
        while True:
            window = get_window(window_list, lock)
            print(window, window_list)
            if window is None:
                break

            trainer = SaveRegressionacc(args, GPU_num, window, lock)
            trainer.train()
    
    procs = []
    for i in range(6):
        proc = multiprocessing.Process(target=worker, args=((i%3)+args.para_gpu, window_list, lock))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

if __name__ == '__main__':
    main()
