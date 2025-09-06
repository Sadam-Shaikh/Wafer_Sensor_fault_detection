import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Use relative imports when deployed
try:
    from src.utils.logger import get_logger
    from src.utils.exception import CustomException
    from src.utils.utils import save_object
    from src.config.config import MODEL_TRAINER_CONFIG
except ImportError:
    # Fallback to direct imports if src module is not found
    from utils.logger import get_logger
    from utils.exception import CustomException
    from utils.utils import save_object
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
            # Define models to train with class_weight to handle imbalance
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            
            # Check if we have at least 2 classes in the target
            unique_classes = np.unique(y_train)
            logger.info(f"Unique classes in training data: {unique_classes}")
            
            if len(unique_classes) < 2:
                logger.warning("Only one class found in training data. Adding synthetic samples for the missing class.")
                # Determine which class is missing (0 or 1)
                existing_class = unique_classes[0]
                missing_class = 1 if existing_class == 0 else 0
                
                # Create synthetic samples with the missing class
                synthetic_count = max(int(len(y_train) * 0.4), 1)  # At least 1 sample
                
                # Create synthetic features by copying some existing samples
                indices_to_copy = np.random.choice(len(X_train), size=synthetic_count, replace=True)
                synthetic_X = X_train[indices_to_copy].copy()
                
                # Create synthetic target with the missing class
                synthetic_y = np.full(synthetic_count, missing_class)
                
                # Combine original and synthetic data
                X_train = np.vstack([X_train, synthetic_X])
                y_train = np.concatenate([y_train, synthetic_y])
                
                logger.info(f"Added {synthetic_count} synthetic samples with class {missing_class}")
                logger.info(f"Updated training data shape: X={X_train.shape}, y={y_train.shape}")
                logger.info(f"Updated unique classes: {np.unique(y_train)}")
                logger.warning("This is for demonstration purposes only. In production, ensure proper class distribution.")
            
            
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