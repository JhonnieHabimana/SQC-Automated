import os
import numpy as np
import torch #ML library like numpy but for ML
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# ------------------- 1️⃣ SETUP DATA PATHS -------------------
data_dir = "/Users/admin/Desktop/dataset"
acceptable_dir = os.path.join(data_dir, "acceptable_spectrograms")
query_dir = os.path.join(data_dir, "query_spectrograms")

# ------------------- 2️⃣ IMAGE AUGMENTATION & PREPROCESSING -------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),  # Convert to tensor (a multi-dimensional array of data usually in the form of vectors and matrices)
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# ------------------- 3️⃣ LOAD DATASET -------------------
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Class distribution
class_counts = [len(os.listdir(acceptable_dir)), len(os.listdir(query_dir))]
print(f"✅ Found {class_counts[0]} 'acceptable' images and {class_counts[1]} 'query' images.")

# Assign weights to balance classes
weights = [1/class_counts[label] for _, label in dataset.samples] # weights are inversely proportional to the class frequencies
sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


# ------------------- 4️⃣ CREATE DATALOADERS -------------------
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------- 5️⃣ DEFINE CNN MODEL -------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #(in_channels, out_channels, kernel_size, padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #(Number of channels in the input image,Number of channels produced by the convolution,Size of the convolving kerne,Padding added to all four sides of the input)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Applies max pooling with a 2x2 filter and a stride of 2 (reduces spatial dimensions by half)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Fully connected layer: input size is 128 feature maps of size 16x16, output size is 256 neurons
        self.fc2 = nn.Linear(256, 1) # Fully connected layer: input size is 256 neurons, output size is 1 neuron (binary classification)
        self.relu = nn.ReLU() # ReLU activation function (introduces non-linearity)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation function (maps output to range [0,1] for binary classification)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # Apply first convolution, then ReLU, then 2x2 max pooling
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # Flatten the tensor starting from dimension 1
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# ------------------- 6️⃣ TRAIN THE MODEL -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam = Adaptive Moment Estimation is an advanced optimization algorithm used for training machine learning models

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

# ------------------- 7️⃣ EVALUATE THE MODEL -------------------
model.eval()
cnn_y_true = []
cnn_y_pred = []
cnn_y_scores = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        
        cnn_y_true.extend(labels.cpu().numpy())
        cnn_y_pred.extend(predicted.cpu().numpy())
        cnn_y_scores.extend(outputs.cpu().numpy())

# Convert to numpy arrays
cnn_y_true = np.array(cnn_y_true)
cnn_y_pred = np.array(cnn_y_pred)
cnn_y_scores = np.array(cnn_y_scores)

# ------------------- 8️⃣ CLASSIFICATION REPORT -------------------
print("\n🔹 CNN Classification Report on Test Set:")
print(classification_report(cnn_y_true, cnn_y_pred, target_names=["Acceptable", "Query"]))

# ------------------- 9️⃣ CONFUSION MATRIX -------------------
cnn_conf_matrix = confusion_matrix(cnn_y_true, cnn_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cnn_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Acceptable", "Query"], yticklabels=["Acceptable", "Query"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.show()

# ------------------- 🔟 ROC CURVE -------------------
cnn_fpr, cnn_tpr, _ = roc_curve(cnn_y_true, cnn_y_scores)
cnn_roc_auc = auc(cnn_fpr, cnn_tpr)

plt.figure(figsize=(7, 5))
plt.plot(cnn_fpr, cnn_tpr, color="blue", lw=2, label=f"CNN ROC Curve (AUC = {cnn_roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

cnn_accuracy = accuracy_score(cnn_y_true, cnn_y_pred)
print(f"CNN Test Accuracy: {cnn_accuracy * 100:.2f}%")

# Define the path to save the model
model_save_path = "/Users/admin/Desktop/cnn_model.pth"

# Save the model state dictionary
torch.save(model.state_dict(), model_save_path)

print(f"✅ Model saved successfully at: {model_save_path}")
