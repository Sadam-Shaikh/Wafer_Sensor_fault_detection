import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.utils import save_object
from config.config import MODEL_TRAINER_CONFIG

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        """
        Initialize model trainer with configuration
        """
        self.model_trainer_config = MODEL_TRAINER_CONFIG
    
    def train_model(self, X_train, y_train):
        """
        Train different models and return the best one
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best model object
        """
        try:
            # Define models to train
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            
            # Train and evaluate models
            model_report = {}
            for model_name, model in models.items():
                logger.info(f"Training {model_name} model")
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_train)
                
                # Calculate metrics
                accuracy = accuracy_score(y_train, y_pred)
                precision = precision_score(y_train, y_pred, zero_division=0)
                recall = recall_score(y_train, y_pred, zero_division=0)
                f1 = f1_score(y_train, y_pred, zero_division=0)
                
                model_report[model_name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            # Find best model based on accuracy
            best_model_name = max(model_report, key=lambda x: model_report[x]["accuracy"])
            best_model = model_report[best_model_name]["model"]
            best_accuracy = model_report[best_model_name]["accuracy"]
            
            logger.info(f"Best model: {best_model_name} with accuracy: {best_accuracy}")
            
            # Check if model meets base accuracy requirement
            if best_accuracy < self.model_trainer_config["base_accuracy"]:
                logger.warning(f"No model achieved the base accuracy of {self.model_trainer_config['base_accuracy']}")
                raise Exception("No model achieved the required base accuracy")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(e)
    
    def initiate_model_training(self, train_array, test_array):
        """
        Initiate the model training process
        
        Args:
            train_array: Training data array
            test_array: Test data array
            
        Returns:
            Path to the trained model file
        """
        try:
            # Create model directory
            os.makedirs(os.path.dirname(self.model_trainer_config["model_file_path"]), exist_ok=True)
            
            logger.info("Splitting training and test data into features and target")
            
            # Split into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model on test data
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            
            logger.info(f"Test metrics - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")
            
            # Save model
            model_path = self.model_trainer_config["model_file_path"]
            save_object(model_path, model)
            
            logger.info(f"Model saved at: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(e)