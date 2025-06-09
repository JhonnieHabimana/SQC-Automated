import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Use RF-labeled data to match other models
X_train_ffnn = X_train_rf
y_train_ffnn = y_train_rf
X_test_ffnn = X_test_rf
y_test_ffnn = y_test_rf

# Normalize features
scaler_ffnn = StandardScaler()
X_train_ffnn_scaled = scaler_ffnn.fit_transform(X_train_ffnn)
X_test_ffnn_scaled = scaler_ffnn.transform(X_test_ffnn)

# Convert to Torch tensors
X_train_ffnn_tensor = torch.tensor(X_train_ffnn_scaled, dtype=torch.float32)
y_train_ffnn_tensor = torch.tensor(y_train_ffnn, dtype=torch.float32).unsqueeze(1)
X_test_ffnn_tensor = torch.tensor(X_test_ffnn_scaled, dtype=torch.float32)
y_test_ffnn_tensor = torch.tensor(y_test_ffnn, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset_ffnn = TensorDataset(X_train_ffnn_tensor, y_train_ffnn_tensor)
train_loader_ffnn = DataLoader(train_dataset_ffnn, batch_size=32, shuffle=True)

class FFNNModel(nn.Module):
    def __init__(self, input_size):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model
input_size_ffnn = X_train_ffnn.shape[1]
model_ffnn = FFNNModel(input_size_ffnn)

# Loss and Optimizer
criterion_ffnn = nn.BCELoss()
optimizer_ffnn = optim.Adam(model_ffnn.parameters(), lr=0.001, weight_decay=1e-4)  # weight decay for regularization

# Training Loop
num_epochs_ffnn = 100
for epoch in range(num_epochs_ffnn):
    model_ffnn.train()
    for X_batch, y_batch in train_loader_ffnn:
        optimizer_ffnn.zero_grad()
        outputs = model_ffnn(X_batch)
        loss = criterion_ffnn(outputs, y_batch)
        loss.backward()
        optimizer_ffnn.step()
    
    # Log every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs_ffnn}], Loss: {loss.item():.4f}")

# Get predictions
model_ffnn.eval()
with torch.no_grad():
    y_test_ffnn_prob = model_ffnn(X_test_ffnn_tensor).numpy().flatten()
    y_test_ffnn_pred = (y_test_ffnn_prob > 0.5).astype(int)

# Map labels
label_map_ffnn = {0: 'acceptable', 1: 'query'}
y_test_ffnn_pred_labels = [label_map_ffnn[pred] for pred in y_test_ffnn_pred]
y_test_ffnn_true_labels = [label_map_ffnn[true] for true in y_test_ffnn.flatten()]

# Classification Report
print(f"FFNN Test AUC: {test_auc_ffnn:.4f}")
print("\nFFNN Classification Report on Test Set:")
print(classification_report(y_test_ffnn_true_labels, y_test_ffnn_pred_labels, target_names=['acceptable', 'query']))

# Confusion Matrix
conf_mat_ffnn = confusion_matrix(y_test_ffnn_true_labels, y_test_ffnn_pred_labels, labels=['acceptable', 'query'])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat_ffnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['acceptable', 'query'],
            yticklabels=['acceptable', 'query'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('FFNN Confusion Matrix (Test Set)')
plt.show()

# ROC Curve
fpr_ffnn, tpr_ffnn, _ = roc_curve(y_test_ffnn, y_test_ffnn_prob)
test_auc_ffnn = auc(fpr_ffnn, tpr_ffnn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_ffnn, tpr_ffnn, color='darkorange', lw=2,
         label=f'FFNN ROC Curve (AUC = {test_auc_ffnn:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('FFNN ROC Curve on Test Set')
plt.legend(loc='lower right')
plt.show()
