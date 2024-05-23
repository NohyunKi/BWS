import os

from PIL import Image
import json

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset

def load_data(dataset, batch_size, is_timm=False):
    if dataset == "CIFAR10":
        CIFAR10_train_transf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224 if is_timm else 32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
            ])

        CIFAR10_test_transf = transforms.Compose([
            transforms.Resize(224 if is_timm else 32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
            ])

        train_dataset = torchvision.datasets.CIFAR10(root=f'./dataset/CIFAR10', 
                                                    train=True,
                                                    transform=CIFAR10_train_transf,
                                                    download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=f'./dataset/CIFAR10', 
                                                    train=False,
                                                    transform=CIFAR10_test_transf,
                                                    download=True)
        eval_dataset = torchvision.datasets.CIFAR10(root=f'./dataset/CIFAR10', 
                                                    train=True,
                                                    transform=CIFAR10_test_transf,
                                                    download=True)
        
        input_size = [32, 32, 3]
        label_size = 10  
                                                    
    elif dataset == "CIFAR100":
        # https://github.com/weiaicunzai/pytorch-cifar100/
        train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224 if is_timm else 32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
            ])

        transform_test = transforms.Compose([
            transforms.Resize(224 if is_timm else 32),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
            ])

        
        train_dataset = torchvision.datasets.CIFAR100(root=f'./dataset/CIFAR100', 
                                                    train=True,
                                                    transform=transform_train,
                                                    download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=f'./dataset/CIFAR100', 
                                                    train=False,
                                                    transform=transform_test,
                                                    download=True)  
        eval_dataset = torchvision.datasets.CIFAR100(root=f'./dataset/CIFAR100', 
                                                    train=True,
                                                    transform=transform_test,
                                                    download=True)  
        input_size = [32, 32, 3]
        label_size = 100

    elif dataset == "CIFAR2":
        class CIFAR2Dataset(torch.utils.data.Dataset):
            def __init__(self, train):
                self.label = {0: 0, 1: 1} # label book

                if train == "train":
                    train = True
                    transf = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                        ])
                elif train == "test":
                    train = False
                    transf = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                        ])
                elif train == "eval":
                    train = True
                    transf = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                        ])
                else:
                    raise NotImplementedError

                dataset = torchvision.datasets.CIFAR10(root=f'./dataset/CIFAR10', 
                                                        train=train,
                                                        transform=transf,
                                                        download=True)

                idxes = []
                for i in range(len(dataset)):
                    if dataset[i][1] in self.label:
                        idxes.append(i)
                self.dataset = torch.utils.data.Subset(dataset, idxes)


            def __getitem__(self, idx):
                data, label = self.dataset[idx]
                label = self.label[label]

                return data, label
                
            def __len__(self):
                return len(self.dataset)

            def idxes(self):
                _idxes = {0: [], 1: []}
                for i in range(len(self.dataset)):
                    for label in self.label:
                        if self.dataset[i][1] == label:
                            _idxes[self.label[label]].append(i)
                
                return _idxes
        
        
        train_dataset = CIFAR2Dataset(train="train")
        test_dataset = CIFAR2Dataset(train="test")
        eval_dataset = CIFAR2Dataset(train="eval")
        
        input_size = [32, 32, 3]
        label_size = 2  
    
    elif dataset == "FMNIST":
        FMNIST_train_transf = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])

        FMNIST_test_transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])

        train_dataset = torchvision.datasets.FashionMNIST(root=f'./dataset/FMNIST', 
                                                        train=True,
                                                        transform=FMNIST_train_transf,
                                                        download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=f'./dataset/FMNIST', 
                                                        train=False,
                                                        transform=FMNIST_test_transf,
                                                        download=True)
        eval_dataset = torchvision.datasets.FashionMNIST(root=f'./dataset/FMNIST', 
                                                        train=True,
                                                        transform=FMNIST_test_transf,
                                                        download=True)
        
        input_size = [28, 28, 1]
        label_size = 10

    elif dataset == "TinyImageNet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        def create_val_img_folder(img_dir):
            '''
            This method is responsible for separating validation images into separate sub folders
            '''
            val_dir = os.path.join(img_dir, 'val')
            img_dir = os.path.join(val_dir, 'images')

            fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
            data = fp.readlines()
            val_img_dict = {}
            for line in data:
                words = line.split('\t')
                val_img_dict[words[0]] = words[1]
            fp.close()

            # Create folder if not present and move images into proper folders
            for img, folder in val_img_dict.items():
                newpath = (os.path.join(img_dir, folder))
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                if os.path.exists(os.path.join(img_dir, img)):
                    os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

        img_dir = os.path.join("dataset/tiny-imagenet-200")
        create_val_img_folder(img_dir)

        train_dataset = torchvision.datasets.ImageFolder(
            f"{img_dir}/train", transform=train_transform)
        eval_dataset = torchvision.datasets.ImageFolder(
            f"{img_dir}/train", transform=test_transform)
        test_dataset = torchvision.datasets.ImageFolder(
            f"{img_dir}/val/images", transform=test_transform)

        input_size = [64, 64, 3]
        label_size = 200

    elif dataset == "ImageNet":
        class ImageNet(Dataset):
            def __init__(self, root, split, transform=None):
                self.samples = []
                self.targets = []
                self.transform = transform
                self.syn_to_class = {}
                self.name_to_idx = {}
                with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
                with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
                samples_dir = os.path.join(root, split)
                count = 0
                for entry in os.listdir(samples_dir):
                    if split == "train":
                        syn_id = entry
                        target = self.syn_to_class[syn_id]
                        syn_folder = os.path.join(samples_dir, syn_id)
                        for sample in os.listdir(syn_folder):
                            self.name_to_idx[sample] = count
                            count += 1
                            sample_path = os.path.join(syn_folder, sample)
                            self.samples.append(sample_path)
                            self.targets.append(target)
                    elif split == "val":
                        syn_id = self.val_to_syn[entry]
                        target = self.syn_to_class[syn_id]
                        sample_path = os.path.join(samples_dir, entry)
                        self.samples.append(sample_path)
                        self.targets.append(target)
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                x = Image.open(self.samples[idx]).convert("RGB")
                if self.transform:
                    x = self.transform(x)
                return x, self.targets[idx]

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
        test_transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )

        train_dataset = ImageNet("dataset/imagenet", "train", train_transform)
        eval_dataset = ImageNet("dataset/imagenet", "train", test_transform)
        test_dataset = ImageNet("dataset/imagenet", "val", test_transform)

        input_size = [224, 224, 3]
        label_size = 1000
    else:
        raise NotImplementedError
                                                    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True,
                                                    drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    pin_memory=True)

    return ((train_dataset, test_dataset, eval_dataset), 
            (train_loader, test_loader), 
            (input_size, label_size))