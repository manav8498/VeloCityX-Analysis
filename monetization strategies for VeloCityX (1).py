#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Load the CSV file into a DataFrame
file_path = 'downloads/2025-VeloCityX-Expanded-Fan-Engagement-Data.csv'
df = pd.read_csv(file_path)

# Step 1: Data Cleaning

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if any exist)
df = df.dropna()

# Verify no missing data remains
print(df.isnull().sum())

# Step 2: Data Analysis & Investigating Trends

# Correlation analysis to understand relationships
correlation_matrix = df.corr()

# View the correlation with 'Virtual Merchandise Purchases'
virtual_merch_corr = correlation_matrix["Virtual Merchandise Purchases"].sort_values(ascending=False)
print("Correlation with Virtual Merchandise Purchases:")
print(virtual_merch_corr)

# Correlation between race activities and purchases
activity_corr = df[['Fan Challenges Completed', 'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)', 'Virtual Merchandise Purchases']].corr()
print("Activity Correlation:")
print(activity_corr)

# Step 3: Clustering Analysis

# Prepare data for clustering (select relevant columns)
X_cluster = df[['Fan Challenges Completed', 'Predictive Accuracy (%)', 'Virtual Merchandise Purchases', 
        'Sponsorship Interactions (Ad Clicks)', 'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)']]

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_cluster)
df['Cluster'] = kmeans.labels_

# View the clustering results
print(df[['User ID', 'Cluster']].head())

# Step 4: Predictive Modeling (Logistic Regression)

# Features (X) and target (y)
X = df[['Fan Challenges Completed', 'Predictive Accuracy (%)', 'Sponsorship Interactions (Ad Clicks)', 
        'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)']]
y = df['Virtual Merchandise Purchases']

# Scale the data for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Address class imbalance by resampling the minority class
# Resampling the dataset to handle imbalanced classes (if needed)
df_majority = df[df['Virtual Merchandise Purchases'] == 0]
df_minority = df[df['Virtual Merchandise Purchases'] > 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # match number in majority class
                                 random_state=42)  # reproducible results

# Combine majority and upsampled minority classes
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Define the new features and labels
X = df_upsampled[['Fan Challenges Completed', 'Predictive Accuracy (%)', 'Sponsorship Interactions (Ad Clicks)', 
        'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)']]
y = df_upsampled['Virtual Merchandise Purchases']

# Re-scale the data
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model with increased iterations and balanced class weights
model = LogisticRegression(max_iter=500, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (using zero_division=1 to avoid undefined metrics)
print(classification_report(y_test, y_pred, zero_division=1))

# Step 5: Visualizations

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot for Clustering
plt.figure(figsize=(8, 6))
plt.scatter(df['Fan Challenges Completed'], df['Virtual Merchandise Purchases'], c=df['Cluster'], cmap='viridis')
plt.title('Clustering of Users Based on Engagement')
plt.xlabel('Fan Challenges Completed')
plt.ylabel('Virtual Merchandise Purchases')
plt.show()

# Step 6: New Fan Challenge Proposal

# Proposal based on analysis
print("""
New Fan Challenge Proposal: 
- Challenge Name: “Ultimate Live Engagement Challenge”
- Objective: Encourage users to spend more time on 'Live 360' and participate in chat activity.
- Predicted Outcome: Users who spend over 180 minutes on Live 360 and send over 100 messages are more likely to purchase virtual merchandise and interact with sponsors.
- Engagement and Monetization: Offer exclusive merchandise or sponsorship deals to drive more purchases and clicks.
""")


# In[ ]:




