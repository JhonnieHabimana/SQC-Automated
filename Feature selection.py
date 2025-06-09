# Load data
feature_path = os.path.join(base_filepath, 'gait_features.csv')
features_df = pd.read_csv(feature_path)

# Encode labels ('acceptable' → 0, 'query' → 1)
features_df['Label'] = features_df['Label'].map({'acceptable': 0, 'query': 1})

X = features_df.drop(columns=['Patient_ID', 'Label']).values
y = features_df['Label'].values
patient_ids = features_df['Patient_ID'].values

# Feature Selection Using RF
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)

# Automatically select only the important features
selector = SelectFromModel(rf_selector, prefit=True)  
X_selected = selector.transform(X)

# Get the selected feature names
selected_features = features_df.drop(columns=['Patient_ID', 'Label']).columns[selector.get_support()]

print("Automatically Selected Features (RF-Based):", selected_features)

# Overwriting X with the selected features
X = X_selected
