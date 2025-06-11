import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

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
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
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
        x = x.view(b, t, -1)
        x = x.mean(dim=1)
        return self.fc2(x)

class WLASLDataset(Dataset):
    def __init__(self, json_path, videos_dir, split='val', num_frames=16):
        self.videos_dir = videos_dir
        self.data = []
        self.num_frames = num_frames

        with open(json_path, 'r') as f:
            content = json.load(f)

        for item in content:
            gloss = item['gloss']
            for inst in item['instances']:
                if inst['split'] != split:
                    continue
                video_id = inst['video_id']
                video_path = os.path.join(videos_dir, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    self.data.append((video_path, gloss))

        self.label_map = {label: i for i, label in enumerate(sorted(set([label for _, label in self.data])))}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self._load_video(video_path)
        return frames, self.label_map[label]

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if i in frame_idxs:
                if not ret:
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
        cap.release()
        frames = torch.stack(frames)
        return frames




def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WLASLDataset(json_path="../WLASL/start_kit/WLASL_v0.3.json", 
                           videos_dir="../WLASL/start_kit/videos", 
                           split='val')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    model = ASLModel(num_classes=len(dataset.label_map)).to(device)

    # load checkpoint
    checkpoint = torch.load("asl_video_model2.pt", map_location=device)
    model_state = checkpoint['model_state_dict']

    model_state.pop('fc2.weight', None)
    model_state.pop('fc2.bias', None)

    model.load_state_dict(model_state, strict=False)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(dataloader):
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                pred_label = dataset.inverse_label_map[p.item()]
                true_label = dataset.inverse_label_map[l.item()]
                print(f"Predicted: {pred_label}, Actual: {true_label}")

    print(f"\nValidation Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")




if __name__ == "__main__":
    evaluate()