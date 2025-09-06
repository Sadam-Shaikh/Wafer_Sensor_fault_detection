import os
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from urllib.parse import quote_plus

# Use relative imports when deployed
try:
    from src.utils.logger import get_logger
    from src.utils.exception import CustomException
except ImportError:
    # Fallback to direct imports if src module is not found
    from utils.logger import get_logger
    from utils.exception import CustomException

logger = get_logger(__name__)

class MongoDBConnection:
    """
    MongoDB Connection Utility for Wafer Fault Detection Project
    """
    def __init__(self, connection_string=None):
        """
        Initialize MongoDB connection with the provided connection string
        """
        try:
            # Use provided connection string or build one with properly escaped credentials
            if connection_string:
                self.connection_string = connection_string
            else:
                # Get credentials from environment or use defaults
                username = os.getenv('MONGODB_USERNAME', 'saddamshekh1934')
                password = os.getenv('MONGODB_PASSWORD', 'S@a_dd@a_m@7498')
                host = os.getenv('MONGODB_HOST', 'waffresdata.5xbt8f6.mongodb.net')
                
                # Properly escape username and password
                escaped_username = quote_plus(username)
                escaped_password = quote_plus(password)
                
                # Build connection string
                self.connection_string = f"mongodb+srv://{escaped_username}:{escaped_password}@{host}/?retryWrites=true&w=majority&appName=WaffresData"
            
            # Initialize client and connect immediately
            self.client = MongoClient(self.connection_string)
            self.db = None
            logger.info("MongoDB connection initialized")
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {e}")
            self.client = None
            self.db = None
            raise CustomException(f"MongoDB Initialization Error: {e}")
        
    def connect(self, db_name="wafer_fault_detection"):
        """
        Connect to MongoDB and return database instance
        """
        try:
            # Ensure client is initialized
            if self.client is None:
                self.client = MongoClient(self.connection_string)
            
            # Ping the server to verify connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
            # Get database
            self.db = self.client[db_name]
            return self.db
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise CustomException(f"MongoDB Connection Error: {e}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise CustomException(f"MongoDB Error: {e}")
    
    def test_connection(self):
        """
        Test MongoDB connection
        
        Returns:
            bool: True if connection is successful
        """
        try:
            if self.client is None:
                self.client = MongoClient(self.connection_string)
            
            # The ismaster command is cheap and does not require auth
            self.client.admin.command('ismaster')
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    def close(self):
        """
        Close the MongoDB connection
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def save_dataframe(self, collection_name, dataframe, metadata=None):
        """
        Save pandas DataFrame to MongoDB collection
        
        Args:
            collection_name (str): Name of the collection
            dataframe (pd.DataFrame): DataFrame to save
            metadata (dict): Additional metadata to store with the data
        """
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if self.db is None:
                self.connect()
                
            # Convert DataFrame to records
            records = dataframe.to_dict("records")
            
            # Add metadata if provided
            if metadata:
                for record in records:
                    record.update({"metadata": metadata})
            
            # Insert into collection
            collection = self.db[collection_name]
            result = collection.insert_many(records)
            
            logger.info(f"Successfully saved {len(result.inserted_ids)} records to {collection_name}")
            return result.inserted_ids
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to MongoDB: {e}")
            raise CustomException(f"MongoDB Save Error: {e}")
    
    def load_dataframe(self, collection_name, query=None, projection=None):
        """
        Load data from MongoDB collection into pandas DataFrame
        
        Args:
            collection_name (str): Name of the collection
            query (dict): MongoDB query to filter documents
            projection (dict): Fields to include/exclude in the result
        """
        try:
            if self.db is None:
                self.connect()
                
            collection = self.db[collection_name]
            
            # Find documents based on query
            cursor = collection.find(query or {}, projection or {})
            
            # Convert to DataFrame
            df = pd.DataFrame(list(cursor))
            
            # Drop MongoDB _id column if it exists
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            logger.info(f"Successfully loaded {len(df)} records from {collection_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from MongoDB: {e}")
            raise CustomException(f"MongoDB Load Error: {e}")
    
    def save_predictions(self, collection_name, data_id, predictions, metadata=None):
        """
        Save model predictions to MongoDB
        
        Args:
            collection_name (str): Name of the collection
            data_id (str): Identifier for the data being predicted
            predictions (list/array): Model predictions
            metadata (dict): Additional metadata about the prediction
        """
        try:
            if self.db is None:
                self.connect()
                
            # Prepare document
            document = {
                "data_id": data_id,
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                "timestamp": pd.Timestamp.now()
            }
            
            # Add metadata if provided
            if metadata:
                document.update({"metadata": metadata})
            
            # Insert into collection
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            
            logger.info(f"Successfully saved predictions to {collection_name}")
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Error saving predictions to MongoDB: {e}")
            raise CustomException(f"MongoDB Prediction Save Error: {e}")