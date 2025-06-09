# ------------------- 1️⃣ CNN Model Definition -------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# ------------------- 2️⃣ Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "/Users/admin/Desktop/cnn_model.pth"
unseen_data_dir = "/Users/admin/Desktop/validation"

# ------------------- 3️⃣ Load Model -------------------
model = CNNModel().to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# ------------------- 4️⃣ Transformations -------------------
unseen_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 3 channels
])

# ------------------- 5️⃣ Load Unseen Data -------------------
image_paths = []
true_labels = []

for root, _, files in os.walk(unseen_data_dir):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)
            folder_name = os.path.basename(root).lower()
            if "acceptable_validation" in folder_name:
                true_labels.append(0)
            elif "query_validation" in folder_name:
                true_labels.append(1)

# ------------------- 6️⃣ Dataset & DataLoader -------------------
class UnlabeledDataset(Dataset):
    def __init__(self, image_paths, true_labels, transform=None):
        self.image_paths = image_paths
        self.true_labels = true_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = datasets.folder.default_loader(image_path)
        label = self.true_labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

unseen_dataset = UnlabeledDataset(image_paths, true_labels, transform=unseen_transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False)

# ------------------- 7️⃣ Predictions & Evaluation -------------------
y_true = []
y_pred = []
file_names = []

with torch.no_grad():
    for inputs, actual_label, img_path in unseen_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()

        y_true.append(actual_label.item())
        y_pred.append(predicted.item())
        file_names.append(os.path.basename(img_path[0]))

# ------------------- 8️⃣ Metrics -------------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n🔍 **Model Evaluation on Unseen Data**")
print(classification_report(y_true, y_pred, target_names=["Acceptable", "Query"]))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Acceptable", "Query"], yticklabels=["Acceptable", "Query"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Unseen Data")
plt.show()

accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ **Model Accuracy on Unseen Data: {accuracy * 100:.2f}%**")

train_accuracy = 0.98  # real training accuracy
test_accuracy = accuracy

print("\n🔎 **Overfitting Check:**")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if train_accuracy > test_accuracy + 10:
    print("⚠️ The model may have overfitted. Consider adding regularization or augmenting data.")
else:
    print("✅ The model generalizes well!")

# ------------------- 9️⃣ Save Predictions to CSV -------------------
results_df = pd.DataFrame({
    "filename": file_names,
    "actual_label": ["Acceptable" if label == 0 else "Query" for label in y_true],
    "predicted_label": ["Acceptable" if label == 0 else "Query" for label in y_pred]
})

csv_output_path = "/Users/admin/Desktop/cnn_predictions.csv"
results_df.to_csv(csv_output_path, index=False)
print(f"\n📄 Predictions saved to: {csv_output_path}")

# ------------------- 0️⃣ LOAD SAVED MODELS -------------------
model_dir = '/Users/admin/Desktop/ML/models'
svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
svm_selector = joblib.load(os.path.join(model_dir, 'svm_feature_selector.pkl'))
rf_selector = joblib.load(os.path.join(model_dir, 'rf_feature_selector.pkl'))

# ------------------- 1️⃣ LOAD VALIDATION FILES -------------------
val_base_path = '/Users/admin/Desktop/Validation2'
acceptable_val_path = os.path.join(val_base_path, 'acceptable_validation')
query_val_path = os.path.join(val_base_path, 'query_validation')

acceptable_val_files = [(os.path.join(acceptable_val_path, f), 'acceptable') for f in os.listdir(acceptable_val_path) if f.endswith('.csv')]
query_val_files = [(os.path.join(query_val_path, f), 'query') for f in os.listdir(query_val_path) if f.endswith('.csv')]
val_filelist = acceptable_val_files + query_val_files

# ------------------- 2️⃣ FEATURE EXTRACTION -------------------
def extract_patient_id(file_path):
    filename = os.path.basename(file_path)
    match = re.match(r'(\d+)_', filename)
    return match.group(1) if match else None

num_features = 26
X_val = np.full((len(val_filelist), num_features), np.nan)
val_labels = []
val_filenames = []
val_patient_ids = []

for i, (file, label) in enumerate(val_filelist):
    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()

    val_filenames.append(os.path.basename(file))
    val_labels.append(label)
    val_patient_ids.append(extract_patient_id(file))

    time = data['Time'].values
    acc_x = data['Acc_X'].values
    acc_y = data['Acc_Y'].values
    acc_z = data['Acc_Z'].values
    gyr_x = data['Gyr_X'].values
    gyr_y = data['Gyr_Y'].values
    gyr_z = data['Gyr_Z'].values
    acc_mag = data['Acc_Magnitude'].values
    gyr_mag = data['Gyr_Magnitude'].values

    number_strides = data['Number_Strides'].values[0]
    duration = data['Duration_s'].values[0]
    avg_stride_speed = data['Avg_Stride_Speed_mps'].values[0]
    avg_stride_length = data['AverageStrideLength'].values[0] if 'AverageStrideLength' in data.columns else np.nan
    avg_stride_duration = data['AverageStrideDuration'].values[0] if 'AverageStrideDuration' in data.columns else np.nan
    avg_stride_height_change = data['AverageStrideHeightChange'].values[0] if 'AverageStrideHeightChange' in data.columns else np.nan

    pkAmps, _ = find_peaks(acc_mag, distance=0.7 * 100, height=0.5 * np.max(acc_mag))
    X_val[i, 0] = len(pkAmps)
    f, Pxx = welch(acc_mag, fs=100, nperseg=256)
    X_val[i, 1] = f[np.argmax(Pxx)]
    X_val[i, 2] = np.mean(np.abs(np.diff(acc_mag)))
    X_val[i, 3] = entropy(np.abs(np.diff(acc_mag)))
    X_val[i, 4] = LinearRegression().fit(np.arange(len(acc_mag)).reshape(-1, 1), acc_mag).coef_[0]
    X_val[i, 5] = number_strides / (duration / 60) if duration > 0 else np.nan
    X_val[i, 6] = avg_stride_speed
    X_val[i, 7] = np.std(np.diff(time))
    X_val[i, 8] = duration
    X_val[i, 9] = number_strides
    X_val[i, 10] = avg_stride_length
    X_val[i, 11] = avg_stride_duration
    X_val[i, 12] = avg_stride_height_change
    X_val[i, 13] = np.std(data['Avg_Stride_Speed_mps']) if 'Avg_Stride_Speed_mps' in data.columns else np.nan
    X_val[i, 14] = len(pkAmps) / number_strides if number_strides > 0 else 0
    X_val[i, 15] = avg_stride_duration / avg_stride_length if avg_stride_length > 0 else np.nan
    X_val[i, 16] = np.std(acc_mag)
    X_val[i, 17] = np.std(gyr_mag)

    if 'Stride_Start' in data.columns:
        stride_starts = data['Stride_Start'].dropna().values
        stride_durations = np.diff(stride_starts)
        if len(stride_durations) >= 2:
            odd = stride_durations[1::2]
            even = stride_durations[0::2]
            min_len = min(len(odd), len(even))
            stride_symmetry = np.mean(np.abs(odd[:min_len] - even[:min_len]))
            X_val[i, 18] = stride_symmetry
    X_val[i, 19] = len(data.dropna(subset=['Turn_Start'])) / number_strides if 'Turn_Start' in data.columns and number_strides > 0 else 0
    X_val[i, 20] = np.sum(Pxx)
    X_val[i, 21] = entropy(Pxx)
    cumulative_power = np.cumsum(Pxx) / np.sum(Pxx)
    X_val[i, 22] = f[np.searchsorted(cumulative_power, 0.95)]
    X_val[i, 23] = np.max(Pxx) / np.sum(Pxx)
    X_val[i, 24] = f[np.argmax(Pxx)] - f[np.argmax(Pxx > 0.01 * np.max(Pxx))]
    f_gyr, Pxx_gyr = welch(gyr_mag, fs=100, nperseg=256)
    X_val[i, 25] = f_gyr[np.argmax(Pxx_gyr)]

# ------------------- 3️⃣ TRANSFORM FEATURES & PREDICT -------------------
X_val_rf_selected = rf_selector.transform(X_val)
X_val_svm_selected = svm_selector.transform(X_val_rf_selected)

y_val_pred = svm_model.predict(X_val_svm_selected)
label_map = {0: 'acceptable', 1: 'query'}
y_val_pred_labels = [label_map[y] for y in y_val_pred]
y_val_true_labels = val_labels

# ------------------- 4️⃣ EVALUATION -------------------
print("\n🔍 **Model Evaluation on Unseen Data**")
print(classification_report(y_val_true_labels, y_val_pred_labels, target_names=["acceptable", "query"]))

# Confusion matrix
conf_matrix = confusion_matrix(y_val_true_labels, y_val_pred_labels, labels=["acceptable", "query"])
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Acceptable", "Query"], 
            yticklabels=["Acceptable", "Query"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Unseen Data")
plt.tight_layout()
plt.show()

# Accuracy
accuracy = accuracy_score(y_val_true_labels, y_val_pred_labels)
print(f"\n**Model Accuracy on Unseen Data: {accuracy * 100:.2f}%**")

# Overfitting check
train_accuracy = 0.86  #training accuracy
test_accuracy = accuracy

print("\n🔎 **Overfitting Check:**")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if train_accuracy > test_accuracy + 10:
    print("The model may have overfitted. Consider regularization, feature tuning, or more training data.")
else:
    print("The model generalises well!")
