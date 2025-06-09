# Initialize and fit SVM on the RF-selected training features
svm_selector = SVC(kernel="linear", C=1.0)
svm_selector.fit(X_train_rf, y_train_rf)  # Using same training set as RF

# Apply model-based feature selection
svm_feature_selector = SelectFromModel(svm_selector, prefit=True)
X_train_svm = svm_feature_selector.transform(X_train_rf)
X_test_svm = svm_feature_selector.transform(X_test_rf)

# Get selected feature indices and names
selected_feature_indices_svm = svm_feature_selector.get_support(indices=True)
selected_features_svm = rf_selected_feature_names[selected_feature_indices_svm]

print("Selected Features (SVM-Based):", selected_features_svm)

#SVM 2
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_svm, y_train_rf)

y_test_prob_svm = svm_model.predict_proba(X_test_svm)[:, 1]
y_test_pred_svm = svm_model.predict(X_test_svm)

label_map_svm = {0: 'acceptable', 1: 'query'}
y_test_pred_labels_svm = [label_map_svm[p] for p in y_test_pred_svm]
y_test_labels_svm = [label_map_svm[t] for t in y_test_rf]

#Classification Report
print("Classification Report (SVM - Test Set):")
print(classification_report(y_test_labels_svm, y_test_pred_labels_svm, target_names=['acceptable', 'query']))

# Confusion Matrix
conf_mat_svm = confusion_matrix(y_test_labels_svm, y_test_pred_labels_svm, labels=['acceptable', 'query'])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['acceptable', 'query'], yticklabels=['acceptable', 'query'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (SVM Test Set)')
plt.show()

# ROC + AUC
fpr_svm_test, tpr_svm_test, _ = roc_curve(y_test_rf, y_test_prob_svm)
auc_svm_test = auc(fpr_svm_test, tpr_svm_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm_test, tpr_svm_test, color='green', lw=2, label=f'SVM Test ROC (AUC = {auc_svm_test:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (SVM - Test Set)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Test AUC (SVM): {auc_svm_test:.4f}")

