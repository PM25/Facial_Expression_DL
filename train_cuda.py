from mylib.resnet import ResNet
from mylib.dataloader import DataLoader

import torch
import torchvision
from torch import nn
from argparse import ArgumentParser
import matplotlib.pyplot as plt


# Arguments
parser = ArgumentParser(description='Training ResNet Model')
parser.add_argument('--imgs', type=str, default='data/images', help='Path of Training Images')
parser.add_argument('--labels', type=str, default='data/labels.txt', help='Path of Labels File')
parser.add_argument('--save', type=str, default='models/model_cuda.pkl', help='Path to Save Models')
parser.add_argument('--val', type=float, default=.2, help='Validation Data Ratio')
parser.add_argument('--epoch', type=int, default=20

, help='Epoch')
parser.add_argument('--bs', type=int, default=32, help='Batch Size')
parser.add_argument('--lr', type=float, default=.005, help='Learning Rate')
parser.add_argument('--log', type=int, default=10, help='Log Interval')
args = parser.parse_args()


# Start from here!
if __name__ == '__main__':
    # Training & Validation Data
    data = DataLoader(imgs_dir=args.imgs, labels_path=args.labels, batch_sz=args.bs)
    n_classes = data.n_classes
    if(args.val > 0):
        train_loader, val_loader = data.train_val_loader(args.val)
    else:
        train_loader = data.train_loader()

    # Model & Loss Function & Optimizer
    model = ResNet(n_classes)
    model = model.cuda()
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training & Validation
    train_loss_his = []
    val_loss_his = []
    for epoch in range(args.epoch):
        # Training Model & Record Loss Value
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x.cuda())
            loss = loss_func(outputs, batch_y.cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if(step % args.log == 0):
                print(f'Epoch {epoch} | Step {step} | Loss {loss.item()}')
        
        train_loss /= len(train_loader)
        train_loss_his.append(train_loss)
        print(f'Training Loss {train_loss}')

        # Validation & Record Loss Value
        if(args.val > 0):
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:      
                    outputs = model(batch_X.cuda())
                    loss = loss_func(outputs, batch_y.cuda())
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_loss_his.append(val_loss)
            print(f'Validation Loss {val_loss}')

    # Plot Training & Validation Loss
    plt.plot(train_loss_his, label="Training Loss")
    plt.plot(val_loss_his, label="Validation Loss")
    plt.legend()
    plt.show()

    # Save Model
    torch.save(model, args.save)
    print(f"*Message: Model save to {args.save} successfully!")    