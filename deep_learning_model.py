#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# In[57]:


# Load dataset
# Show top 5 rows
df = pd.read_csv('student_performance_dataset.csv')
print(df.head())
df.describe()


# In[58]:


# Scale Study_Hours_per_Week to same range as scores (50-100)
max_hours = df['Study_Hours_per_Week'].max()
df['Study_Hours_Scaled'] = df['Study_Hours_per_Week'] / max_hours * 100
# Internal Marks (weighted combination)
df['Internal_Marks'] = 0.6 * df['Past_Exam_Scores'] + 0.4 * df['Study_Hours_Scaled']


# In[59]:


df['Assignment_Submission_Rate'] = df['Study_Hours_Scaled']  # 0-100%


# In[60]:


df['Internal_Assessment_Marks'] = 0.5 * df['Internal_Marks'] + 0.5 * df['Past_Exam_Scores']


# In[61]:


# Encode Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Encode Parental Education Level
edu_mapping = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
df['Parental_Education_Level'] = df['Parental_Education_Level'].map(edu_mapping)

# Encode Internet Access and Extracurricular Activities
df['Internet_Access_at_Home'] = df['Internet_Access_at_Home'].map({'No': 0, 'Yes': 1})
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})

# Encode Pass/Fail if needed
df['Pass_Fail'] = df['Pass_Fail'].map({'Fail': 0, 'Pass': 1})


# In[62]:


scaler = MinMaxScaler()

numeric_cols = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
                'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# In[63]:


features = ['Gender', 'Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
            'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities',
            'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

X = df[features]
y = df['Final_Exam_Score']  # Regression target

print(X.head())
print(y.head())


# In[64]:


# Import Visualization libraries 


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


plt.figure(figsize=(8,5))
sns.histplot(df['Final_Exam_Score'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Final Exam Scores')
plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.show()


# In[67]:


numeric_cols = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
                'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

plt.figure(figsize=(15,8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, color='lightgreen')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[68]:


plt.figure(figsize=(12,8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[69]:


sns.pairplot(df[['Final_Exam_Score'] + numeric_cols], kind='scatter', diag_kind='kde')
plt.show()


# In[70]:


categorical_cols = ['Gender', 'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities']

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col], y=df['Final_Exam_Score'])
    plt.title(f'Final Exam Score by {col}')
    plt.show()


# In[71]:


# Seperate the input and output for the model 


# In[72]:


# Features (exclude Student_ID and Pass_Fail)
features = ['Gender', 'Study_Hours_per_Week', 'Attendance_Rate',
            'Past_Exam_Scores', 'Parental_Education_Level', 'Internet_Access_at_Home',
            'Extracurricular_Activities', 'Internal_Marks',
            'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

X = df[features]
y = df['Final_Exam_Score']


# In[73]:


# Seperate Train and Test dataset


# In[74]:


from sklearn.model_selection import train_test_split


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[76]:


# Import Libraries for neural network


# In[77]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# In[78]:


# Making 3 layer network 


# In[102]:


nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),          # Helps stabilize and speed up training
    Dropout(0.2),                  # Drop 20% neurons, smaller than before

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(1)  # Regression output
])



# In[103]:


# Compile the model 


# In[104]:


nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[105]:


# Train the model 


# In[106]:


nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[107]:


history = nn_model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=8)


# In[85]:


# Test the model 


# In[108]:


# Predict final exam scores for test data
y_pred_nn = nn_model.predict(X_test)

# Convert predictions from 2D array to 1D
y_pred_nn = y_pred_nn.flatten()
nn_model.save("student_score_model.h5")
nn_model.export("student_score_model")




# In[109]:


# Evluate the performance 


# In[110]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Compute RMSE manually
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
print(f"Neural Network RMSE: {rmse_nn:.2f}")


# In[111]:


from sklearn.metrics import r2_score

r2_nn = r2_score(y_test, y_pred_nn)
print(f"Neural Network R²: {r2_nn:.2f}")


# In[112]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[113]:


# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred_nn)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred_nn)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(y_test, y_pred_nn)

print(f"Neural Network Evaluation on Test Set:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")



# In[114]:


# Predection vs actual


# In[115]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_nn, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Neural Network: Actual vs Predicted Scores')
plt.show()


# In[116]:


# Training loss 


# In[117]:


plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Neural Network Training vs Validation Loss')
plt.legend()
plt.show()


# In[118]:


import pandas as pd
import numpy as np

# Original data
new_data = pd.DataFrame({
    'Gender': [0, 0, 1, 1, 1],
    'Study_Hours_per_Week': [0.924138, 0.206897, 0.379310, 0.586207, 0.931034],
    'Attendance_Rate': [0.964105, 0.563803, 0.750403, 0.841704, 0.973678],
    'Past_Exam_Scores': [0.99, 0.46, 0.48, 0.98, 0.26],
    'Parental_Education_Level': [3, 0, 3, 1, 2],
    'Internet_Access_at_Home': [1, 0, 1, 0, 0],
    'Extracurricular_Activities': [1, 0, 0, 0, 1],
    'Internal_Marks': [0.991354, 0.327043, 0.426936, 0.795833, 0.598015],
    'Assignment_Submission_Rate': [0.994138, 0.206897, 0.379310, 0.586207, 0.931034],
    'Internal_Assessment_Marks': [0.996737, 0.380448, 0.444837, 0.886533, 0.433451]
})

features = ['Gender', 'Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
            'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities',
            'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

X_new = new_data[features].copy()

# Set random seed for reproducibility
np.random.seed(42)

# Add noise to numeric columns
numeric_cols = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
                'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

for col in numeric_cols:
    noise = np.random.normal(0, 0.05, size=len(X_new))  # small noise ~5%
    X_new[col] = np.clip(X_new[col] + noise, 0, 1)  # keep values in [0,1] range

# Optionally, add randomness to categorical columns
X_new['Gender'] = X_new['Gender'].apply(lambda x: x if np.random.rand() > 0.1 else 1-x)
X_new['Internet_Access_at_Home'] = X_new['Internet_Access_at_Home'].apply(lambda x: x if np.random.rand() > 0.1 else 1-x)
X_new['Extracurricular_Activities'] = X_new['Extracurricular_Activities'].apply(lambda x: x if np.random.rand() > 0.1 else 1-x)
X_new['Parental_Education_Level'] = X_new['Parental_Education_Level'].apply(lambda x: np.clip(x + np.random.choice([-1,0,1]), 0, 3))

print(X_new)


# In[119]:


# Predict scores
y_pred_new = nn_model.predict(X_new)
y_pred_new = y_pred_new.flatten()  # Convert to 1D array

# Show predictions
new_data['Predicted_Final_Exam_Score'] = y_pred_new
print(new_data[['Predicted_Final_Exam_Score']].round())


# In[120]:


# Linear regression for same data 


# In[121]:


df.head()


# In[122]:


# Features you want to use
features = ['Gender', 'Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
            'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities',
            'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']

X = df[features].copy()
y = df['Final_Exam_Score']

# Scale numeric columns
scaler = MinMaxScaler()
numeric_cols = ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
                'Internal_Marks', 'Assignment_Submission_Rate', 'Internal_Assessment_Marks']
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[125]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluation
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Evaluation:")
print(f"MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")


# In[127]:


from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression Evaluation:")
print(f"MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")


# In[128]:


# Linear Regression
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_lr, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores (Linear Regression)")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# Random Forest
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_rf, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores (Random Forest)")
plt.title("Random Forest: Actual vs Predicted")
plt.show()


# In[129]:


# Linear Regression residuals
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(6,4))
sns.histplot(residuals_lr, bins=10, kde=True, color='blue')
plt.title("Linear Regression Residuals Distribution")
plt.xlabel("Error")
plt.show()

# Random Forest residuals
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(6,4))
sns.histplot(residuals_rf, bins=10, kde=True, color='green')
plt.title("Random Forest Residuals Distribution")
plt.xlabel("Error")
plt.show()


# In[130]:


importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title("Random Forest Feature Importance")
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred_lr, label='Predicted LR', marker='x')
plt.plot(y_pred_rf, label='Predicted RF', marker='s')
plt.title("Actual vs Predicted Scores")
plt.xlabel("Sample Index")
plt.ylabel("Final Exam Score")
plt.legend()
plt.show()


# In[ ]:


print("Hello Prejan")


# In[ ]:




