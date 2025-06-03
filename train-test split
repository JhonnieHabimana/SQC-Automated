 Get Unique Participants
unique_participants = np.unique(patient_ids)
print(f"Total unique participants: {len(unique_participants)}")

# Randomly Split Participants (15 for Training, 5 for Testing)
np.random.seed(42) 
np.random.shuffle(unique_participants)

train_participants = unique_participants[:15]
test_participants = unique_participants[15:]

print(f"Train Participants: {train_participants}")
print(f"Test Participants: {test_participants}")

# Create Boolean Masks for Train/Test
train_mask = np.isin(patient_ids, train_participants)
test_mask = np.isin(patient_ids, test_participants)

# Split Data (Using Selected Features)
X_train, y_train, train_ids = X[train_mask], y[train_mask], patient_ids[train_mask]
X_test, y_test, test_ids = X[test_mask], y[test_mask], patient_ids[test_mask]

print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")
