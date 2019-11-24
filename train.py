from mylib.dataloader import DataLoader

import torch
import torchvision
from torch import nn
from argparse import ArgumentParser


# Arguments
parser = ArgumentParser(description='Training ResNet Model')
parser.add_argument('--imgs', type=str, default='data/images', help='Path of Training Images')
parser.add_argument('--labels', type=str, default='data/labels.txt', help='Path of Labels File')
parser.add_argument('--save', type=str, default='models/model.pkl', help='Path to Save Models')
parser.add_argument('--bs', default=32, type=int, help='Batch Size')
parser.add_argument('--lr', default=.0001, type=float, help='Learning Rate')
parser.add_argument('--log', default=10, type=int, help='Log Interval')
args = parser.parse_args()


# Start from here!
if __name__ == '__main__':
    # Training Data
    data = DataLoader(imgs_dir=args.imgs, labels_path=args.labels, batch_sz=args.bs)
    train_loader = data.train_loader()
    n_classes = data.n_classes

    # Model & Loss Function & Optimizer
    model = torchvision.models.resnet18(pretrained=True)
    fc_in_size = model.fc.in_features
    model.fc = nn.Linear(fc_in_size, n_classes)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(9):
        for step, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_func(outputs, batch_y)
            loss.backward()
            optimizer.step()

            if(step % args.log == 0):
                print(f'Epoch {epoch} | Step {step} | Loss {loss.item()}')

    # Save Model
    torch.save(model, args.save)
    print(f"*Message: Model save to {args.save} successfully!")