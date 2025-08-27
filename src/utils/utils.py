import os
import pickle
import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.exception import CustomException

logger = get_logger(__name__)

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Error in save_object: {e}")
        raise CustomException(e)

def load_object(file_path):
    """
    Load a Python object from a file using pickle
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error in load_object: {e}")
        raise CustomException(e)

def save_numpy_array_data(file_path, array):
    """
    Save numpy array data to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
            
        logger.info(f"Numpy array saved successfully at: {file_path}")
    except Exception as e:
        logger.error(f"Error in save_numpy_array_data: {e}")
        raise CustomException(e)

def load_numpy_array_data(file_path):
    """
    Load numpy array data from file
    """
    try:
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)
            
        logger.info(f"Numpy array loaded successfully from: {file_path}")
        return array
    except Exception as e:
        logger.error(f"Error in load_numpy_array_data: {e}")
        raise CustomException(e)