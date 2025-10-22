#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


#loading the data set
import pandas as pd
data = pd.read_csv('churn.csv')
data


# In[ ]:


## Descriptive Summary


# In[13]:


print("\nSummary statistics:")
data.describe()


# In[ ]:


## Data Information


# In[15]:


print("Shape of dataset:", data.shape)
print("\nColumn info:")
print(data.info())


# In[ ]:


##Checking for missing values


# In[11]:


print("\nMissing values per column:")
data.isnull().sum()


# In[ ]:


## Checking for outliers


# In[17]:


# Loop through all numeric columns and check for outliers using IQR
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(f"{column}: {len(outliers)} outliers")


# In[ ]:


## Replacing outliers with the mean


# In[19]:


import pandas as pd

# Loop through all numeric columns
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Median of the column
    median = data[column].median()
    
    # Replace outliers with median
    data[column] = data[column].apply(lambda x: median if x < lower or x > upper else x)
    
    # Recalculate outliers after replacement
    outliers_after = data[(data[column] < lower) | (data[column] > upper)]
    print(f"{column}: {len(outliers_after)} outliers remaining after replacement")

print("Outliers replaced with median successfully.")


# In[ ]:


## Target variable Distribution


# In[23]:


# Distribution of target variable
print("\nClass distribution:")
print(data['Churn'].value_counts())

# Plot class distribution
sns.countplot(x='Churn', data=data)
plt.title("Class Distribution")
plt.show()


# In[ ]:


##column data types


# In[25]:


for col in data.columns:
    print(f"Column '{col}' has data type: {data[col].dtype}")


# In[ ]:


## Label Encoding


# In[29]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Make a copy to avoid overwriting accidentally
df = data.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical object columns
categorical_cols = ['State', 'International plan', 'Voice mail plan']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Encode boolean column 'Churn' as 0/1
df['Churn'] = df['Churn'].astype(int)

print("Label encoding completed successfully.")
df


# In[ ]:


## Feature Engineering


# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Assuming df is your preprocessed dataset with encoding done
X = df.drop('Churn', axis=1)   # Features
y = df['Churn']                # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importances = model.feature_importances_

# Put into DataFrame for readability
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature importance scores:")
print(feature_importance_df)

# Optional: visualize
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance from Random Forest")
plt.show()


# In[ ]:


## Correlation Heatmap


# In[31]:


# Correlation matrix
corr = data.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


## train-Split-test


# In[35]:


from sklearn.model_selection import train_test_split

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Class distribution before balancing:\n", y_train.value_counts())


# In[ ]:


## Balancing dataset using SMOTE


# In[37]:


from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply only on training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:\n", y_train_resampled.value_counts())


# In[ ]:


## Visualization before and after class balance 


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Before SMOTE
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title("Before SMOTE")

# After SMOTE
sns.countplot(x=y_train_resampled, ax=axes[1])
axes[1].set_title("After SMOTE")

plt.tight_layout()
plt.show()


# In[ ]:


## Feature scaling 


# In[41]:


from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit on resampled training set and transform
X_train_scaled = scaler.fit_transform(X_train_resampled)

# Transform test set (use same scaler)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed. Train and test sets are now ready.")


# In[ ]:


## Random Forest


# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


##Logistic Regression


# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


## Support Vector Machine


# In[53]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


## KNeighborsClassifier(KNN)


# In[59]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


## Model Accuracy Comparison


# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy values from your results
accuracy_results = {
    "Random Forest": 0.8881,
    "Logistic Regression": 0.8657,
    "SVM": 0.8582,
    "KNN": 0.8955
}

plt.figure(figsize=(8,5))
sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="Blues_d")
plt.title("Model Comparison - Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()


# In[ ]:


## Evaluation metrics comparison


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Macro average metrics from your results
metrics_data = {
    "Random Forest": {"Precision": 0.82, "Recall": 0.65, "F1-score": 0.69},
    "Logistic Regression": {"Precision": 0.77, "Recall": 0.55, "F1-score": 0.55},
    "SVM": {"Precision": 0.43, "Recall": 0.50, "F1-score": 0.46},
    "KNN": {"Precision": 0.88, "Recall": 0.65, "F1-score": 0.70},
}

# Convert into DataFrame
metrics_df = []
for model, scores in metrics_data.items():
    for metric, value in scores.items():
        metrics_df.append([model, metric, value])

metrics_df = pd.DataFrame(metrics_df, columns=["Model", "Metric", "Score"])

# Plot grouped bar chart
plt.figure(figsize=(10,6))
sns.barplot(data=metrics_df, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Model Comparison - Precision, Recall, and F1-score")
plt.ylim(0,1)
plt.legend(title="Metric")
plt.show()


# In[ ]:




