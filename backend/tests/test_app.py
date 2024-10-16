#pip install flask numpy scikit-learn nltk pytest (before running pytest test_app.py)

import pytest
import json
import sys
import os
from unittest.mock import MagicMock, mock_open

# Adjust the PYTHONPATH to include the parent directory (backend)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def mock_pickle_load(file):
    # Return a mock object depending on which file is being loaded
    if 'kmeans_model.pkl' in file.name:
        # Return a mock KMeans model
        class MockKMeans:
            def predict(self, X):
                return [0]  # Return a fixed cluster label
        return MockKMeans()
    elif 'vectorizer.pkl' in file.name:
        # Return a mock vectorizer
        class MockVectorizer:
            def transform(self, X):
                return [[0.1, 0.2, 0.3]]  # Return a fixed vector
        return MockVectorizer()
    else:
        # Return None for other files
        return None

@pytest.fixture
def app(monkeypatch):
    # Mock 'open' and 'pickle.load' in 'utils.utils' before importing 'app'
    mock_file = mock_open()
    monkeypatch.setattr('builtins.open', mock_file)
    monkeypatch.setattr('pickle.load', mock_pickle_load)

    # Now import the app
    from app import app  # Import the Flask app instance from 'app.py'

    return app

@pytest.fixture
def client(app):
    # Create a test client for the Flask app
    with app.test_client() as client:
        yield client

def test_similarity_valid_input(client, mocker):
    # Mock 'get_similar_verses' in 'routes.verseSearch'
    mock_results = [
        {"verse": "Verse 1", "similarity": 0.9},
        {"verse": "Verse 2", "similarity": 0.8},
        {"verse": "Verse 3", "similarity": 0.7},
    ]

    # Mock the get_similar_verses function in 'utils.utils'
    mocker.patch('utils.utils.get_similar_verses', return_value=mock_results)

    # Create the POST request payload
    user_input = {"user_input": "In the beginning"}
    response = client.post('/api/similarity', data=json.dumps(user_input), content_type='application/json')

    # Check the status code and response data
    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data['user_input'] == "In the beginning"
    assert len(response_data['results']) == 3

    similarities = [result['similarity'] for result in response_data['results']]
    assert similarities == [0.9, 0.8, 0.7]

def test_similarity_missing_input(client):
    # Test the case when no input is provided in the request
    response = client.post('/api/similarity', data=json.dumps({}), content_type='application/json')

    # Check the status code and error message
    assert response.status_code == 400
    response_data = response.get_json()
    assert response_data['error'] == "No input provided"
