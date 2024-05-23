from collections import defaultdict
import datetime
import time

import numpy as np
import torch
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from base_trainer import BaseTrainer
import utils.args

class MakeAllMeasure(BaseTrainer):
    def __init__(self):
        super().__init__()

        self.results = defaultdict(lambda: np.zeros((self.args.epochs, len(self.train_dataset))))
        self.measures = defaultdict(lambda: np.zeros((self.args.epochs, len(self.train_dataset))))
        self.features = []
        self.all_features = []

        self.measure_loader = torch.utils.data.DataLoader(
            dataset=self.eval_dataset, batch_size=400, shuffle=False, num_workers=20)

    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.args = utils.args.get_args(self._additional_parser)
        print(self.args)

        # Set device and Fix seed
        self.device = 'cuda' if (torch.cuda.is_available() and not self.args.no_cuda) else 'cpu'

        utils.train.fix_seed(self.args.seed)

        self.max_test_acc = 0
        self.max_epoch = 0

    def _additional_parser(self, parser):
        parser.add_argument("--noise", action='store_true', default=False)
        parser.add_argument("--noise_rate", type=float, default=0)

        return parser

    def _load_data(self):
        super()._load_data()
        
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
            
            if os.path.isfile(f"data/noise_{self.args.dataset}_{self.args.noise_rate}.npy"):
                print("load_noisy_label")
                noise_label = np.load(f"data/noise_{self.args.dataset}_{self.args.noise_rate}.npy")
                self.train_dataset = NoisyDataset(
                    self.train_dataset, self.args.noise_rate, self.label_size, noise_label)
            else:
                self.train_dataset = NoisyDataset(
                    self.train_dataset, self.args.noise_rate, self.label_size)
                noise_label = self.train_dataset.get_noise_label()
                np.save(f"data/noise_{self.args.dataset}_{self.args.noise_rate}.npy", noise_label)

            self.eval_dataset = NoisyDataset(
                self.eval_dataset, self.args.noise_rate, self.label_size, noise_label)
            
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                        num_workers=16,
                                                        pin_memory=False)

    def reset(self):
        self._load_data()
        self._get_model()
        self._get_optimizer_and_scheduler()
        self._get_logger()

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

            with torch.no_grad():
                self.write_results(epoch)
            self.calc_measures(epoch)

            # record time
            self.writer.add_scalar("Time", time.time()-t_epoch, epoch)
        
        print("Total Learning time: {:2f}s".format(time.time() - t_training))
        self._train_summary()

    def save_model(self, epoch):
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.models_dir, "model_" + str(epoch) + ".pt")
        )

    def write_results(self, epoch):
        model = self.model
        model.eval()
        
        results = defaultdict(list)
        for (img, labels) in self.measure_loader:
            img = img.to(self.device)
            labels = labels.to(self.device)
            logit = model(img)

            output = F.softmax(logit, dim=1)
            correctness = output.argmax(1).eq(labels)
            
            results["correctness"].append(correctness)
            results["l2norm"].append(
                torch.norm(output - F.one_hot(labels, num_classes=self.label_size), dim=1))
        
        for key, value in results.items():
            self.results[key][epoch-1, :] = torch.cat(value, 0).cpu().numpy()

    def calc_measures(self, epoch):
        if epoch != 1:
            correct_diff = \
                self.results["correctness"][epoch-1, :] - self.results["correctness"][epoch-2, :]
            forgetting_idx = np.where(correct_diff == -1)[0]
            self.results["forgetting"][epoch-1, forgetting_idx] += 1
        
        self.measures["forgetting"][epoch-1, :] = self.results["forgetting"][:epoch, :].sum(0)
        
    def _train_summary(self):
        super()._train_summary()

        dir = f"data/measures_{self.args.dataset}_{self.args.model}"
        dir += f"_0_{self.args.noise_rate*100}" if self.args.noise else ""
        dir += 'simple.npz'

        np.savez(dir, EL2N=self.results['l2norm'][19], forgetting=self.measures["forgetting"][-1])
            
def main():
    trainer = MakeAllMeasure()
    trainer.train()

if __name__ == '__main__':
    main()
