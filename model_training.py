import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

def preprocess_data(df):
    # Separate target variable and features
    target = df['Adopted'].map({'Yes': 1, 'No': 0})
    features = df.drop('Adopted', axis=1)

    # on-hot-encoding categorical variables is not required in this version of XGBoost library. A more efficient Optimal Partitioning technique is instead employed.
    features[features.select_dtypes(include=['object']).columns] = features.select_dtypes(include=['object']).astype("category")
    target = target.astype("category")

    return features, target

def train_xgboost_model(X_train, y_train, X_val, y_val):
    xgb_model = XGBClassifier(
        tree_method="hist", 
        enable_categorical=True, 
        max_cat_to_onehot=1,
        eval_metric='auc')

    # Train the model on the training data
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return xgb_model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)


def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved at:", model_path)


def main():
    # Load data (assuming you have a DataFrame named 'data')
    data = pd.read_csv(r'''./ai-platform-unified_datasets_tabular_petfinder-tabular-classification.csv''') 

    # Preprocess the data
    features, target = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train the XGBoost model with the best number of trees
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(xgb_model, X_test, y_test)

    # Save the final model
    model_dir = r'''./artifacts/model'''  # Choose a directory to save the model
    save_model(xgb_model, model_dir)

if __name__ == "__main__":
    main()