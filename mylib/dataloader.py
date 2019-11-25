import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torch.utils.data as Data


class DataLoader:
    def __init__(self, imgs_dir='data/images', labels_path='data/labels.txt', batch_sz=32):
        self.batch_sz = batch_sz
        self.imgs_dir = Path(imgs_dir)
        self.labels_path = Path(labels_path)
        self.labels = self.fname_labels(labels_path)
        self.n_classes = len(set(self.labels.values()))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])

    
    def fname_labels(self, path):
        with open(path, 'r') as infile:
            labels = {}
            for line in infile.readlines():
                fname, label = line.strip('\n').split(' ')
                labels[fname] = eval(label) - 1 
        return labels


    def train_loader(self, msg=True):
        X, y = [], []
        for fname in self.imgs_dir.glob('train*.jpg'):
            img = Image.open(str(fname)).convert('RGB')
            img = self.transform(img)
            X.append(img)
            fname = f'{fname.stem[:-8]}.jpg'
            y.append(self.labels[fname])
        
        tensor_X = torch.stack(X)
        tensor_y = torch.LongTensor(y)
        torch_dataset = Data.TensorDataset(tensor_X, tensor_y)
        data_loader = Data.DataLoader(dataset=torch_dataset,
                                    batch_size=self.batch_sz,
                                    shuffle=True,
                                    num_workers=0)
        if(msg): print(f'*Message: Loaded {len(tensor_X)} Training Images.')
        return data_loader

    
    def test_loader(self, msg=True):
        X, y = [], []
        for fname in self.imgs_dir.glob('test*.jpg'):
            img = Image.open(str(fname)).convert('RGB')
            img = self.transform(img)
            X.append(img)
            fname = f'{fname.stem[:-8]}.jpg'
            y.append(self.labels[fname])
            
        tensor_X = torch.stack(X)
        tensor_y = torch.LongTensor(y)
        torch_dataset = Data.TensorDataset(tensor_X, tensor_y)
        data_loader = Data.DataLoader(dataset=torch_dataset,
                                    batch_size=self.batch_sz,
                                    shuffle=True,
                                    num_workers=0)
        if(msg): print(f'*Message: Loaded {len(tensor_X)} Testing Images.')
        return data_loader


    def train_val_loader(self, val_ratio=.2, msg=True):
        X, y = [], []
        for fname in self.imgs_dir.glob('train*.jpg'):
            img = Image.open(str(fname)).convert('RGB')
            img = self.transform(img)
            X.append(img)
            fname = f'{fname.stem[:-8]}.jpg'
            y.append(self.labels[fname])
        
        tensor_X = torch.stack(X)
        tensor_y = torch.LongTensor(y)
        dataset = Data.TensorDataset(tensor_X, tensor_y)
        
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size])
        train_loader = Data.DataLoader(dataset=train_dataset,
                                    batch_size=self.batch_sz,
                                    shuffle=True,
                                    num_workers=0)
        val_loader = Data.DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_sz,
                                    shuffle=True,
                                    num_workers=0)
        if(msg): 
            print(f'*Message: Loaded {train_size} Training Images.')
            print(f'*Message: Loaded {val_size} Validation Images.')
        return train_loader, val_loader