import pandas as pd
import numpy as np
import pickle

def preprocess_data(df):
    # Separate target variable and features
    target = df['Adopted'].map({'Yes': 1, 'No': 0})
    features = df.drop('Adopted', axis=1)

    # on-hot-encoding categorical variables is not required in this version of XGBoost library. A more efficient Optimal Partitioning technique is instead employed.
    features[features.select_dtypes(include=['object']).columns] = features.select_dtypes(include=['object']).astype("category")
    target = target.astype("category")

    return features, target

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, data):
    # Preprocess the new data
    features, target = preprocess_data(data)

    # Predict using the loaded model
    predictions = model.predict(features)

    # Create a DataFrame with features and predictions
    result_df = pd.DataFrame(features)
    result_df['Adopted'] = target
    result_df['Adopted'] = result_df['Adopted'].replace({1: 'Yes', 0: 'No'})
    result_df['Adopted_prediction'] = predictions
    result_df['Adopted_prediction'] = result_df['Adopted_prediction'].replace({1: 'Yes', 0: 'No'})
    return result_df

def main():
    # Load the saved XGBoost model
    result_path = r'''output/results.csv'''
    model_path = r'''artifacts/model/xgboost_model.pkl'''  # Path to the saved model
    xgb_model = load_model(model_path)

    # Load new data for scoring (assuming it's in a DataFrame named 'new_data')
    new_data = pd.read_csv('ai-platform-unified_datasets_tabular_petfinder-tabular-classification.csv') 

    # Make predictions on the new data
    result_df = predict(xgb_model, new_data)
    
    result_df.to_csv(result_path)

    # Print the results
    print(result_df)

if __name__ == "__main__":
    main()