import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.dummy import DummyClassifier

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

# Load training and test data
train_data = pd.read_csv('KDDTrain+.txt', names=column_names)
test_data = pd.read_csv('KDDTest+.txt', names=column_names)

# Combine train and test for consistent encoding
combined_df = pd.concat([train_data, test_data], axis=0)

# Encode categorical features
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    encoder = LabelEncoder()
    combined = pd.concat([train_data[col], test_data[col]])
    encoder.fit(combined)
    train_data[col] = encoder.transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])

# Map labels to binary classification
train_data['label'] = train_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
test_data['label'] = test_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Split features and target
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dummy classifier for baseline
dummy = DummyClassifier(strategy='most_frequent')

# Train Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [2, 4],
    'max_features': [None],
    'bootstrap': [True, False]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

# Fit the model to the data
dummy.fit(X_train, y_train)
random_search.fit(X_train, y_train)

# Find best parameters and evaluate with test data
y_dummy_pred = dummy.predict(X_test)
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Calculate dummy metrics
d_accuracy = accuracy_score(y_test, y_dummy_pred)
d_precision, d_recall, d_f1, _ = precision_recall_fscore_support(y_test, y_dummy_pred, average='macro', zero_division=0)

# Get accuracy, precision, recall, and f1-score
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

print(f"Dummy Accuracy:  {d_accuracy:.4f}")
print(f"Dummy Precision: {d_precision:.4f}")
print(f"Dummy Recall:    {d_recall:.4f}")
print(f"Dummy F1-Score:  {d_f1:.4f}")
print()

print(f"Overall Accuracy:  {accuracy:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall:    {recall:.4f}")
print(f"Overall F1-Score:  {f1:.4f}")