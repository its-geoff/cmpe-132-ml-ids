import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("Train_data.csv")
df_test = pd.read_csv("Test_data.csv")

# Encode categorical features
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.transform(df_test[col])

# Binary labels (0 = normal, 1 = attack)
df['binary_label'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['binary_label'] = df_test['class'].apply(lambda x: 0 if x == 'normal' else 1)

# Split features and labels
X_train = df.drop(['class', 'level', 'binary_label'], axis=1)
y_train = df['binary_label']
X_test = df_test.drop(['class', 'level', 'binary_label'], axis=1)
y_test = df_test['binary_label']

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Grid search over KNN parameters
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski'],
    'p': [1, 2]  # Manhattan, Euclidean
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Evaluate best model
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

# Results
print("Best Hyperparameters:", grid_search.best_params_)
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

