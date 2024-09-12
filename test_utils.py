import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    # Defining two vectors for testing cosine similarity
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([1, 1, 0])
    
    # Calculating cosine similarity using the function to be tested
    result = cosine_similarity(vector1, vector2)
    
    # Expected result is the cosine of the angle between the two vectors
    # For these vectors, cos(45 degrees) = sqrt(2)/2 ≈ 0.7071
    expected_result = np.sqrt(2) / 2
    
    # Checking if the result is close to the expected value
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    # Defining a set of points and a query point
    points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    query_point = np.array([6, 7])
    
    # Calculating nearest neighbor using the function to be tested
    result = nearest_neighbor(query_point, points)
    
    # Expected index of th
