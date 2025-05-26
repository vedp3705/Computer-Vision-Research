import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# added to use silicon gpu - faster traing
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu"))

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale

class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.attn = AttentionBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.attn(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

def train_and_save_model():
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    dataset = datasets.ImageFolder("../Training_Data", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ASLModel(num_classes=len(dataset.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    os.makedirs("./Models", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "class_to_idx": dataset.class_to_idx}, "./Models/asl_model.pt")

if __name__ == "__main__":
    train_and_save_model()

    checkpoint = torch.load("./Models/asl_model.pt", map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = ASLModel(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_dir = "../Validation_Data/Test"
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
    predicted_letters = []

    for fname in test_images:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(x)
            pred = output.argmax(1).item()
            predicted_letters.append(idx_to_class[pred])

    print("Predicted Word:", ''.join(predicted_letters))