import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        z = x.view(b, c, -1).mean(dim=2)  
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
        s = s.view(b, c, 1, 1)
        return x * s.expand_as(x)

class SE_ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(SE_ASL_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(128)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.output(x)

class ValidationSequenceDataset(Dataset):
    def __init__(self, word_folder, transform=None):
        self.image_paths = sorted(
            [os.path.join(word_folder, f) for f in os.listdir(word_folder) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image

def validate_model_on_sequences(model, validation_root, class_labels, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tests_folder = os.path.join(validation_root, "Tests")
    results = {}

    for word in os.listdir(tests_folder):
        word_path = os.path.join(tests_folder, word)
        dataset = ValidationSequenceDataset(word_path, transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        predictions = []
        with torch.no_grad():
            for image in loader:
                image = image.to(device)
                output = model(image)
                pred = output.argmax(dim=1).item()
                predictions.append(class_labels[pred])

        results[word] = predictions
        print(f"[Validation] Word: {word} → Prediction Sequence: {predictions}")

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_labels = sorted(os.listdir("../Training_Data"))
    num_classes = len(class_labels)

    model = SE_ASL_CNN(num_classes=num_classes).to(device) # init
    validate_model_on_sequences(model, "../Validation_Data", class_labels, device)   # run validation

if __name__ == "__main__":
    main()
