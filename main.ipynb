import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
import matplotlib.pyplot as plt


# Arguments
parser = ArgumentParser(description='Predicting Facial Expression of an Image.')
parser.add_argument('--model', type=str, default='models/model.pkl', help='Path of Previous Trained Model')
parser.add_argument('--img', type=str, default='img.jpg', help='Path of Image')
args = parser.parse_args()


# PyTorch Transform function
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])

# Start frome here!
if __name__ == '__main__':
    classes = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness" ,"Anger", "Neutral"]
    n_classes = len(classes)

    # Load Model
    model = torch.load(args.model).eval()

    # Load Image & Predict 
    with torch.no_grad():
        img = Image.open(args.img).convert('RGB')
        img = transform(img)
        output = model(img.unsqueeze(0))
        output = F.softmax(output, dim=1)
        _, predict = torch.max(output, 1)
    
    plt.barh(range(n_classes), output.squeeze().tolist())
    plt.yticks(range(n_classes), classes)
    plt.title(f'Predict: {classes[predict]}')
    plt.show()