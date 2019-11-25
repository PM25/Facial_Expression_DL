from mylib.dataloader import DataLoader

import torch
from argparse import ArgumentParser


# Arguments
parser = ArgumentParser(description='Testing Model')
parser.add_argument('--model', type=str, default='models/model.pkl', help='Path of Previous Trained Model')
parser.add_argument('--imgs', type=str, default='data/images', help='Path of Testing Images')
parser.add_argument('--labels', type=str, default='data/labels.txt', help='Path of Labels File')
parser.add_argument('--bs', default=32, type=int, help='Batch Size')
args = parser.parse_args()


# Start frome here!
if __name__ == '__main__':
    # Testing Data
    data = DataLoader(imgs_dir=args.imgs, labels_path=args.labels, batch_sz=args.bs)
    test_loader = data.test_loader()
    n_classes = data.n_classes
    classes = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness" ,"Anger", "Neutral"]

    # Load Model
    model = torch.load(args.model)

    # Evaluation
    class_correct = [ 0. for i in range(n_classes) ]
    class_total = [ 0. for i in range(n_classes) ]
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicts = torch.max(outputs, 1)
            correct = (predicts == batch_y).squeeze().tolist()
            for label, c in zip(batch_y, correct):
                class_correct[label] += c
                class_total[label] += 1
    
    print('-' * 10)
    for i in range(n_classes):
        print(f"Test Accuracy of {classes[i]}: {100*(class_correct[i]/class_total[i]):.2f}%")
    print('-' * 10)
    print(f'Overall Accuracy: {100*(sum(class_correct)/sum(class_total)):.2f}%')