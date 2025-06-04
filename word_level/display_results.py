import matplotlib.pyplot as plt
import re

# Read the log file
with open("./v2/results.txt", "r") as f:
    lines = f.readlines()

# Prepare containers
epochs = []
train_loss, val_loss = [], []
train_acc, val_acc = [], []

# Regular expression pattern to extract metrics
pattern = r"Epoch (\d+)/\d+: Train Loss: ([\d.]+), Train Acc: ([\d.]+) \| Val Loss: ([\d.]+), Val Acc: ([\d.]+)"

for line in lines:
    match = re.match(pattern, line)
    if match:
        epoch, t_loss, t_acc, v_loss, v_acc = match.groups()
        epochs.append(int(epoch))
        train_loss.append(float(t_loss))
        train_acc.append(float(t_acc))
        val_loss.append(float(v_loss))
        val_acc.append(float(v_acc))

# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc', marker='o')
plt.plot(epochs, val_acc, label='Val Acc', marker='o')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
