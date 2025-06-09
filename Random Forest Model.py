# Load data
feature_path = os.path.join(base_filepath, 'gait_features.csv')
features_df_rf = pd.read_csv(feature_path)

# Encode labels ('acceptable' → 0, 'query' → 1)
features_df_rf['Label'] = features_df_rf['Label'].map({'acceptable': 0, 'query': 1})

# Extract raw data
X_rf_raw = features_df_rf.drop(columns=['Patient_ID', 'Label']).values
y_rf = features_df_rf['Label'].values
patient_ids_rf = features_df_rf['Patient_ID'].values

# Feature Selection Using Random Forest
rf_selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector_model.fit(X_rf_raw, y_rf)

# Automatically select only important features
rf_feature_selector = SelectFromModel(rf_selector_model, prefit=True)
X_rf_selected = rf_feature_selector.transform(X_rf_raw)

# Get selected feature names
rf_selected_feature_names = features_df_rf.drop(columns=['Patient_ID', 'Label']).columns[rf_feature_selector.get_support()]
print("Automatically Selected Features (RF-Based):", rf_selected_feature_names)

# Final selected feature matrix
X_rf = X_rf_selected

#RF 2
# Get Unique Participants
unique_participants_rf = np.unique(patient_ids_rf)
print(f"Total unique participants: {len(unique_participants_rf)}")

# Randomly Split Participants (15 for Training, 5 for Testing)
np.random.seed(42)
np.random.shuffle(unique_participants_rf)

train_participants_rf = unique_participants_rf[:15]
test_participants_rf = unique_participants_rf[15:]

print(f"Train Participants: {train_participants_rf}")
print(f"Test Participants: {test_participants_rf}")

# Create Boolean Masks
train_mask_rf = np.isin(patient_ids_rf, train_participants_rf)
test_mask_rf = np.isin(patient_ids_rf, test_participants_rf)

# Split Data
X_train_rf, y_train_rf, train_ids_rf = X_rf[train_mask_rf], y_rf[train_mask_rf], patient_ids_rf[train_mask_rf]
X_test_rf, y_test_rf, test_ids_rf = X_rf[test_mask_rf], y_rf[test_mask_rf], patient_ids_rf[test_mask_rf]

print(f"Training Set: {X_train_rf.shape}, Test Set: {X_test_rf.shape}")

#RF 3
logo_rf = LeaveOneGroupOut()
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

tpr_list_rf, fpr_list_rf, auc_list_rf = [], [], []

print("\nStarting LOSO-CV for RF...")

for train_idx_rf, val_idx_rf in logo_rf.split(X_train_rf, y_train_rf, groups=train_ids_rf):
    X_tr_rf, X_val_rf = X_train_rf[train_idx_rf], X_train_rf[val_idx_rf]
    y_tr_rf, y_val_rf = y_train_rf[train_idx_rf], y_train_rf[val_idx_rf]

    if len(np.unique(y_val_rf)) < 2:
        print(f"Skipping Fold (Only One Class Present): {np.unique(y_val_rf)}")
        continue

    model_rf.fit(X_tr_rf, y_tr_rf)
    y_val_prob_rf = model_rf.predict_proba(X_val_rf)[:, 1]

    fpr_rf, tpr_rf, _ = roc_curve(y_val_rf, y_val_prob_rf)
    fpr_list_rf.append(fpr_rf)
    tpr_list_rf.append(tpr_rf)
    auc_list_rf.append(auc(fpr_rf, tpr_rf))

if len(auc_list_rf) == 0:
    print("⚠️ No valid folds had both classes. Cannot calculate AUC.")
    avg_auc_rf = np.nan
else:
    avg_auc_rf = np.mean(auc_list_rf)

print(f"Average AUC (RF LOSO-CV): {avg_auc_rf:.4f}")

#RF 4
model_rf.fit(X_train_rf, y_train_rf)
y_test_prob_rf = model_rf.predict_proba(X_test_rf)[:, 1]
y_test_pred_rf = model_rf.predict(X_test_rf)

label_map_rf = {0: 'acceptable', 1: 'query'}
y_test_pred_labels_rf = [label_map_rf[p] for p in y_test_pred_rf]
y_test_labels_rf = [label_map_rf[t] for t in y_test_rf]

#Classification Report
print("Classification Report (RF - Test Set):")
print(classification_report(y_test_labels_rf, y_test_pred_labels_rf, target_names=['acceptable', 'query']))

# Confusion Matrix
conf_mat_rf = confusion_matrix(y_test_labels_rf, y_test_pred_labels_rf, labels=['acceptable', 'query'])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['acceptable', 'query'], yticklabels=['acceptable', 'query'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (RF Test Set)')
plt.show()

# ROC + AUC
fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test_rf, y_test_prob_rf)
auc_rf_test = auc(fpr_rf_test, tpr_rf_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf_test, tpr_rf_test, color='darkorange', lw=2, label=f'RF Test ROC (AUC = {auc_rf_test:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (RF - Test Set)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Test AUC (RF): {auc_rf_test:.4f}")
