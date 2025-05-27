import os
import torch
from torchvision import transforms
from PIL import Image
from asl_model import ASLModel
from glob import glob

device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

def evaluate_model():
    checkpoint = torch.load("./asl_model.pt", map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = ASLModel(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    test_path = "../../Validation_Data/Tests/"
    test_dirs = sorted([f for f in os.listdir(test_path)])

    for dir_name in test_dirs[1:]:
        dir_path = os.path.join(test_path, dir_name)
        test_imgs = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')])

        predicted_letters = []
        predicted_confidence = []

        for img_name in test_imgs:
            img_path = os.path.join(dir_path, img_name)
            image = Image.open(img_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                pred = probs.argmax(1).item()
                confidence = probs[0, pred].item()

                predicted_letters.append(idx_to_class[pred])
                predicted_confidence.append(round(confidence, 3))

        print("Predicted Word:", ''.join(predicted_letters))
        print("Confidence:", predicted_confidence)
if __name__ == "__main__":
    evaluate_model()
