# League of Legends Match Predictor

# Overview
This project implements a machine learning model to predict the outcome of League of Legends matches. The dataset includes various in-game statistics, and a logistic regression model is trained to classify whether a team wins or loses based on these features.

# Features
- Data preprocessing and feature engineering
- Model training using logistic regression with PyTorch
- Evaluation metrics including accuracy, confusion matrix, and ROC curve
- Model optimization with L2 regularization and hyperparameter tuning
- Feature importance analysis
- Model saving and loading for deployment

# Dataset
The dataset used in this project is `league_of_legends_data_large.csv`. It contains match statistics with the target variable `win`, indicating whether a team won (1) or lost (0).

# 1. Data Preprocessing
- Load the dataset using `pandas`
- Standardize the features using `StandardScaler`
- Convert data into PyTorch tensors

# 2. Model Training
- Train a logistic regression model using PyTorch
- Use stochastic gradient descent (SGD) optimizer
- Evaluate the model's accuracy

# 3. Model Optimization
- Apply L2 regularization to prevent overfitting
- Perform hyperparameter tuning to optimize the learning rate

# 4. Model Evaluation
- Generate a confusion matrix
- Plot the ROC curve and calculate AUC
- Display a classification report with precision, recall, and F1-score

# 5. Saving and Loading Model
- Save the trained model using `torch.save`
- Reload the model and ensure consistent performance

# 6. Feature Importance
- Extract model weights to assess feature importance
- Visualize feature impact using a bar plot

# Results
The model is evaluated based on accuracy, precision, recall, and F1-score. The optimal learning rate is determined through experimentation, and feature importance analysis highlights key factors influencing match outcomes.

