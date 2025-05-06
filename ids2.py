import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load datasets
train_data = pd.read_csv('Train_data.csv')
test_data = pd.read_csv('Test_data_with_labels.csv')  # Use updated test data with labels

# Preprocess Train Data
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

# Preprocess Test Data
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# One-hot encode categorical features (protocol_type, service, flag)
categorical_columns = ['protocol_type', 'service', 'flag']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit encoder on training data and transform both datasets
encoder.fit(X_train[categorical_columns])
encoded_train = encoder.transform(X_train[categorical_columns])
encoded_test = encoder.transform(X_test[categorical_columns])

# Convert encoded arrays to DataFrames with consistent column names
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

# Scale numerical features
numerical_columns = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                     'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                     'dst_host_srv_rerror_rate']
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train[numerical_columns])
scaled_test = scaler.transform(X_test[numerical_columns])

# Combine encoded categorical and scaled numerical features
X_train_preprocessed = pd.concat([encoded_train_df, pd.DataFrame(scaled_train)], axis=1)
X_test_preprocessed = pd.concat([encoded_test_df, pd.DataFrame(scaled_test)], axis=1)

# Balance the training dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed.values, y_train)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_balanced, y_train_balanced)

# Optimize Random Forest with RandomizedSearchCV
rf_model = RandomForestClassifier(n_jobs=-1)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [40, 80],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf,
                                      n_iter=10, cv=3, n_jobs=-1)
random_search_rf.fit(X_train_balanced, y_train_balanced)

# Evaluate models on test set
y_pred_dt = dt_model.predict(X_test_preprocessed.values)
y_pred_rf = random_search_rf.best_estimator_.predict(X_test_preprocessed.values)

print("\nDecision Tree Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_dt, average='weighted', zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred_dt, average='weighted', zero_division=0))

print("\nRandom Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred_rf, average='weighted', zero_division=0))
