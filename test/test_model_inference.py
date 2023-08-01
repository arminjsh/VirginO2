import pandas as pd
import pytest
import pickle
from model_inference import preprocess_data, predict

# Sample DataFrame for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Type':['Dog', 'Dog', 'Dog', 'Cat', 'Dog'],
        'Age': [5, 3, 2, 4, 6],
        'Breed1': ['Golden Retriever', 'Poodle', 'Siamese', 'Persian', 'Labrador'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Female'],
        'Color1': ['Golden', 'White', 'Brown', 'Orange', 'Black'],
        'Color2': ['No Color', 'Black', 'White', 'Brown', 'No Color'], 
        'MaturitySize': ['Large', 'Medium', 'Small', 'Small', 'Medium'],
        'FurLength': ['Long', 'Short', 'Short', 'Medium', 'Medium'], 
        'Vaccinated': ['Yes', 'Not Sure', 'No', 'Yes', 'No'],
        'Sterilized': ['Yes', 'No', 'Not Sure', 'Yes', 'Yes'],
        'Health': ['Healthy', 'Healthy', 'Minor Injury', 'Serious Injury', 'Healthy'],
        'Fee': [1000, 1500, 0, 150, 600], 
        'PhotoAmt': [1, 7, 3, 2, 8],
        'Adopted': ['Yes', 'No', 'No', 'Yes', 'Yes']
    })
    return data

# Test preprocess_data function
def test_preprocess_data(sample_data):
    features, target = preprocess_data(sample_data)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert len(features) == len(target)
    assert 'Adopted' not in features.columns
    assert 'Adopted' in target.name


# Test predict function
def test_predict(sample_data):
    # Load the XGBoost model in the script
    model_path = r'''artifacts/model/xgboost_model.pkl'''  # Path to the saved model
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)
    

    # Assuming we have a DataFrame named 'new_data' for testing predictions
    new_data = sample_data.copy()  # Use the same sample data for testing

    result_df = predict(xgb_model, new_data)

    assert isinstance(result_df, pd.DataFrame)
    assert 'Adopted' in result_df.columns
    assert 'Adopted_prediction' in result_df.columns
    assert len(result_df) == len(new_data)