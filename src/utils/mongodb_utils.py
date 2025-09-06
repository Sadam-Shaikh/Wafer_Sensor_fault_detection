import os
import pymongo
import pandas as pd
from dotenv import load_dotenv

# Try to import from src or directly
try:
    from src.utils.logger import get_logger
    from src.utils.exception import CustomException
except ImportError:
    from utils.logger import get_logger
    from utils.exception import CustomException

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class MongoDBClient:
    def __init__(self, connection_string=None):
        """
        Initialize MongoDB client with connection string
        
        Args:
            connection_string: MongoDB connection string. If None, uses environment variable or default
        """
        try:
            # Use provided connection string or get from environment or use default
            self.connection_string = connection_string or os.getenv('MONGODB_URI', 
                'mongodb+srv://saddamshekh1934:S%40a_dd%40a_m%407498@waffresdata.5xbt8f6.mongodb.net/?retryWrites=true&w=majority&appName=WaffresData')
            
            # Alternatively, use urllib.parse to properly escape username and password
            # from urllib.parse import quote_plus
            # username = "saddamshekh1934"
            # password = "S@a_dd@a_m@7498"
            # self.connection_string = f"mongodb+srv://{quote_plus(username)}:{quote_plus(password)}@waffresdata.5xbt8f6.mongodb.net/?retryWrites=true&w=majority&appName=WaffresData"
            
            # Create MongoDB client
            self.client = pymongo.MongoClient(self.connection_string)
            
            # Default database and collections
            self.db_name = os.getenv('MONGODB_DB', 'wafer_fault_detection')
            self.db = self.client[self.db_name]
            
            logger.info(f"MongoDB connection established to database: {self.db_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise CustomException(e)
    
    def test_connection(self):
        """
        Test MongoDB connection
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # The ismaster command is cheap and does not require auth
            self.client.admin.command('ismaster')
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    def save_dataframe(self, df, collection_name, metadata=None):
        """
        Save pandas DataFrame to MongoDB collection
        
        Args:
            df: pandas DataFrame to save
            collection_name: name of the collection
            metadata: additional metadata to store with the data
            
        Returns:
            str: ID of the inserted document with metadata
            list: List of IDs for the inserted data records
        """
        try:
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Get or create collection
            collection = self.db[collection_name]
            
            # Insert records
            result = collection.insert_many(records)
            record_ids = result.inserted_ids
            
            # Store metadata with reference to the data
            if metadata:
                metadata['record_count'] = len(records)
                metadata['record_ids'] = [str(id) for id in record_ids]
                metadata_collection = self.db[f"{collection_name}_metadata"]
                metadata_id = metadata_collection.insert_one(metadata).inserted_id
                
                logger.info(f"Saved {len(records)} records to {collection_name} with metadata ID: {metadata_id}")
                return str(metadata_id), [str(id) for id in record_ids]
            
            logger.info(f"Saved {len(records)} records to {collection_name}")
            return None, [str(id) for id in record_ids]
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to MongoDB: {e}")
            raise CustomException(e)
    
    def load_dataframe(self, collection_name, query=None, limit=None):
        """
        Load data from MongoDB collection into pandas DataFrame
        
        Args:
            collection_name: name of the collection
            query: MongoDB query to filter documents
            limit: maximum number of documents to return
            
        Returns:
            pandas DataFrame with the data
        """
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Find documents
            if query is None:
                query = {}
                
            cursor = collection.find(query)
            
            if limit:
                cursor = cursor.limit(limit)
                
            # Convert to DataFrame
            df = pd.DataFrame(list(cursor))
            
            # Drop MongoDB _id column if it exists
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            logger.info(f"Loaded {len(df)} records from {collection_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from MongoDB: {e}")
            raise CustomException(e)
    
    def save_predictions(self, input_data, predictions, collection_name="predictions"):
        """
        Save prediction results to MongoDB
        
        Args:
            input_data: input data used for prediction (DataFrame or dict)
            predictions: prediction results
            collection_name: name of the collection to store predictions
            
        Returns:
            str: ID of the inserted document
        """
        try:
            # Convert input data to dict if it's a DataFrame
            if isinstance(input_data, pd.DataFrame):
                input_data = input_data.to_dict('records')
                
            # Create document with input and predictions
            document = {
                "input_data": input_data,
                "predictions": predictions,
                "timestamp": pd.Timestamp.now()
            }
            
            # Get or create collection
            collection = self.db[collection_name]
            
            # Insert document
            result = collection.insert_one(document)
            
            logger.info(f"Saved prediction to {collection_name} with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving prediction to MongoDB: {e}")
            raise CustomException(e)
    
    def close(self):
        """
        Close MongoDB connection
        """
        try:
            self.client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")