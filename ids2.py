import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# Load data (using a small sample to avoid memory issues)
train_data = pd.read_csv('Train_data.csv').sample(frac=0.2, random_state=42)
test_data = pd.read_csv('Test_data.csv').sample(frac=0.2, random_state=42)

# Split into features and labels
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# Label encode categorical columns safely
categorical_cols = ['duration', 'protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))

    # Handle unseen values in test set
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)

# Scale numerical features
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Balance the training dataset using ROS
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# Baseline performance
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)

# Train Decision Tree model
# dt_model = DecisionTreeClassifier(random_state=42)
# param_grid_dt = {
#     'max_depth': [3, 5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['gini', 'entropy']  # Or 'log_loss' for sklearn >=1.1
# }
# grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy')
# grid_search_dt.fit(X_train_balanced, y_train_balanced)

# Optimize Random Forest with RandomizedSearchCV
rf_model = RandomForestClassifier(n_jobs=-1)
param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [60, 100],
    'min_samples_split': [4, 7],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf,
                              n_iter=5, cv=3, n_jobs=-1)
grid_search_rf.fit(X_train_balanced, y_train_balanced)

# Train a lightweight RandomForest
# clf = RandomForestClassifier(n_estimators=25, max_depth=8, random_state=42, n_jobs=-1)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# Evaluate models on test set
# best_dt = grid_search_dt.best_estimator_
# y_pred_dt = dt_search.predict(X_test_preprocessed.values)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test.values)

# Evaluate baseline performance
print("Baseline Accuracy:", accuracy_score(y_test, y_dummy_pred))
print("Baseline Precision:", precision_score(y_test, y_dummy_pred, average='weighted', zero_division=0))
print("Baseline Recall:", recall_score(y_test, y_dummy_pred, average='weighted', zero_division=0))
print("Baseline F1-Score:", f1_score(y_test, y_dummy_pred, average='weighted', zero_division=0))
print("")

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred_rf, average='weighted', zero_division=0))

