import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Column names based on the KDD dataset description
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
    "logged_in", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files", 
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", 
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate", "label", "level"
]

# Load datasets (download the files if not already present)
train_df = pd.read_csv('KDDTrain+.txt', names=column_names)
test_df = pd.read_csv('KDDTest+.txt', names=column_names)

# Combine train and test for consistent encoding
combined_df = pd.concat([train_df, test_df], axis=0)

# Encode categorical features
for col in ['protocol_type', 'service', 'flag']:
    encoder = LabelEncoder()
    combined_df[col] = encoder.fit_transform(combined_df[col])

# Convert labels to binary classes (normal vs. attack)
combined_df['label'] = combined_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Split combined back to train and test
train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]

# Split features and target
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Get precision, recall, f1-score
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=clf.classes_)
accuracy = accuracy_score(y_test, y_pred)

# Print results
for i, label in enumerate(clf.classes_):
    print(f"Class: {label}")
    print(f"  Accuracy:  {accuracy: .4f}")
    print(f"  Precision: {precision[i]: .4f}")
    print(f"  Recall:    {recall[i]: .4f}")
    print(f"  F1-Score:  {f1[i]: .4f}")
    print()

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro', labels=clf.classes_)

print(f"Overall Accuracy:  {accuracy:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall:    {recall:.4f}")
print(f"Overall F1-Score:  {f1:.4f}")