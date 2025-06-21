# File: backend/data/models/model_loader.py
"""
Model loading and management utilities for TruthGuard
"""

import os
import json
import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Centralized model loading and management"""
    
    def __init__(self, models_dir: str = "/app/data/models/"):
        self.models_dir = Path(models_dir)
        self.models_cache = {}
        self.config_cache = {}
        self.load_config()
        
    def load_config(self):
        """Load model configuration"""
        try:
            config_path = self.models_dir / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config_cache = json.load(f)
                logger.info("Model configuration loaded successfully")
            else:
                logger.warning("Model configuration file not found")
        except Exception as e:
            logger.error(f"Error loading model configuration: {e}")
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a specific model by name"""
        try:
            # Check if model is already cached
            if model_name in self.models_cache:
                return self.models_cache[model_name]
            
            # Get model configuration
            if model_name not in self.config_cache:
                logger.error(f"Model {model_name} not found in configuration")
                return None
            
            model_config = self.config_cache[model_name]
            model_path = self.models_dir / model_config.get('file_path', '')
            
            # Load model based on type
            model = self._load_model_by_type(model_path, model_config)
            
            if model:
                self.models_cache[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_model_by_type(self, model_path: Path, config: Dict) -> Optional[Any]:
        """Load model based on its type"""
        try:
            model_type = config.get('model_type', '').lower()
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Scikit-learn models
            if any(ml_type in model_type for ml_type in ['random_forest', 'logistic_regression', 'svm']):
                return joblib.load(model_path)
            
            # Pickle files
            elif model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            
            # Sentence transformers
            elif 'sentence-transformers' in model_type:
                return SentenceTransformer(model_type)
            
            # TensorFlow/Keras models
            elif model_path.suffix in ['.h5', '.hdf5']:
                try:
                    import tensorflow as tf
                    return tf.keras.models.load_model(model_path)
                except ImportError:
                    logger.error("TensorFlow not available for loading .h5 models")
                    return None
            
            # PyTorch models
            elif model_path.suffix in ['.pt', '.pth', '.bin']:
                try:
                    import torch
                    return torch.load(model_path, map_location='cpu')
                except ImportError:
                    logger.error("PyTorch not available for loading .pt models")
                    return None
            
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        if model_name in self.config_cache:
            return self.config_cache[model_name]
        return {}
    
    def list_available_models(self) -> List[str]:
        """List all available models"""
        return list(self.config_cache.keys())
    
    def unload_model(self, model_name: str):
        """Unload a model from cache"""
        if model_name in self.models_cache:
            del self.models_cache[model_name]
            logger.info(f"Model {model_name} unloaded from cache")
    
    def reload_model(self, model_name: str):
        """Reload a model (useful after retraining)"""
        self.unload_model(model_name)
        return self.load_model(model_name)
    
    def get_model_performance(self, model_name: str) -> Dict:
        """Get performance metrics for a model"""
        config = self.get_model_info(model_name)
        
        performance_metrics = {}
        
        # Extract common metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'mae', 'rmse']:
            if metric in config:
                performance_metrics[metric] = config[metric]
        
        # Add training date and version
        performance_metrics.update({
            'version': config.get('version', 'unknown'),
            'training_date': config.get('training_date', 'unknown'),
            'model_type': config.get('model_type', 'unknown')
        })
        
        return performance_metrics

class ModelValidator:
    """Validate model performance and detect drift"""
    
    def __init__(self, loader: ModelLoader):
        self.loader = loader
    
    def validate_model(self, model_name: str, test_data: pd.DataFrame, 
                      target_column: str) -> Dict:
        """Validate model performance on test data"""
        try:
            model = self.loader.load_model(model_name)
            if not model:
                return {'error': 'Model not found'}
            
            # Prepare features
            feature_columns = self._get_feature_columns(model_name)
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions, model_name)
            
            return {
                'model_name': model_name,
                'validation_date': datetime.now().isoformat(),
                'metrics': metrics,
                'sample_size': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return {'error': str(e)}
    
    def _get_feature_columns(self, model_name: str) -> List[str]:
        """Get feature columns for a specific model"""
        config = self.loader.get_model_info(model_name)
        return config.get('features', [])
    
    def _calculate_metrics(self, y_true, y_pred, model_name: str) -> Dict:
        """Calculate performance metrics based on model type"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            config = self.loader.get_model_info(model_name)
            model_type = config.get('model_type', '').lower()
            
            metrics = {}
            
            # Classification metrics
            if any(clf_type in model_type for clf_type in ['classifier', 'logistic', 'forest']):
                metrics.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1_score': f1_score(y_true, y_pred, average='weighted')
                })
            
            # Regression metrics
            elif any(reg_type in model_type for reg_type in ['regression', 'lstm']):
                metrics.update({
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2_score': r2_score(y_true, y_pred)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def detect_drift(self, model_name: str, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame) -> Dict:
        """Detect data drift between reference and current data"""
        try:
            from scipy.stats import ks_2samp
            
            feature_columns = self._get_feature_columns(model_name)
            drift_results = {}
            
            for feature in feature_columns:
                if feature in reference_data.columns and feature in current_data.columns:
                    # Kolmogorov-Smirnov test for distribution comparison
                    statistic, p_value = ks_2samp(
                        reference_data[feature].dropna(),
                        current_data[feature].dropna()
                    )
                    
                    drift_results[feature] = {
                        'ks_statistic': statistic,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05  # 5% significance level
                    }
            
            # Overall drift assessment
            drift_detected_count = sum(1 for result in drift_results.values() 
                                     if result['drift_detected'])
            
            return {
                'model_name': model_name,
                'drift_analysis_date': datetime.now().isoformat(),
                'features_analyzed': len(feature_columns),
                'features_with_drift': drift_detected_count,
                'drift_percentage': drift_detected_count / len(feature_columns) if feature_columns else 0,
                'feature_drift_details': drift_results,
                'overall_drift_detected': drift_detected_count > len(feature_columns) * 0.3
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift for model {model_name}: {e}")
            return {'error': str(e)}

class EnsemblePredictor:
    """Ensemble prediction using multiple models"""
    
    def __init__(self, loader: ModelLoader):
        self.loader = loader
        self.ensemble_config = loader.config_cache.get('ensemble_config', {})
    
    def predict(self, features: Dict) -> Dict:
        """Make ensemble prediction"""
        try:
            if not self.ensemble_config.get('use_ensemble', False):
                return {'error': 'Ensemble not configured'}
            
            models = self.ensemble_config.get('models', [])
            voting_strategy = self.ensemble_config.get('voting_strategy', 'simple')
            
            predictions = {}
            weights = {}
            
            # Get predictions from each model
            for model_config in models:
                model_name = model_config['name']
                model_weight = model_config.get('weight', 1.0)
                
                model = self.loader.load_model(model_name)
                if model:
                    # Prepare features for this specific model
                    model_features = self._prepare_features(model_name, features)
                    
                    if model_features is not None:
                        prediction = model.predict([model_features])[0]
                        predictions[model_name] = prediction
                        weights[model_name] = model_weight
            
            if not predictions:
                return {'error': 'No valid predictions obtained'}
            
            # Combine predictions based on voting strategy
            if voting_strategy == 'weighted':
                final_prediction = self._weighted_vote(predictions, weights)
            else:
                final_prediction = self._simple_vote(predictions)
            
            confidence = self._calculate_confidence(predictions, weights)
            
            return {
                'ensemble_prediction': final_prediction,
                'confidence': confidence,
                'individual_predictions': predictions,
                'models_used': list(predictions.keys()),
                'voting_strategy': voting_strategy
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    def _prepare_features(self, model_name: str, features: Dict) -> Optional[List]:
        """Prepare features for a specific model"""
        try:
            config = self.loader.get_model_info(model_name)
            required_features = config.get('features', [])
            
            model_features = []
            for feature in required_features:
                if feature in features:
                    model_features.append(features[feature])
                else:
                    logger.warning(f"Feature {feature} not found for model {model_name}")
                    return None
            
            return model_features
            
        except Exception as e:
            logger.error(f"Error preparing features for {model_name}: {e}")
            return None
    
    def _weighted_vote(self, predictions: Dict, weights: Dict) -> float:
        """Calculate weighted average of predictions"""
        weighted_sum = sum(predictions[model] * weights[model] 
                          for model in predictions.keys())
        total_weight = sum(weights[model] for model in predictions.keys())
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _simple_vote(self, predictions: Dict) -> float:
        """Calculate simple average of predictions"""
        return sum(predictions.values()) / len(predictions)
    
    def _calculate_confidence(self, predictions: Dict, weights: Dict) -> float:
        """Calculate confidence score for ensemble prediction"""
        if len(predictions) < 2:
            return 0.5
        
        values = list(predictions.values())
        variance = np.var(values)
        
        # Lower variance indicates higher confidence
        confidence = 1.0 / (1.0 + variance)
        
        return min(max(confidence, 0.0), 1.0)

# Usage example and initialization
if __name__ == "__main__":
    # Initialize model loader
    loader = ModelLoader()
    
    # List available models
    print("Available models:", loader.list_available_models())
    
    # Load a specific model
    classifier = loader.load_model('claim_classifier')
    if classifier:
        print("Claim classifier loaded successfully")
        print("Model info:", loader.get_model_info('claim_classifier'))
    
    # Initialize ensemble predictor
    ensemble = EnsemblePredictor(loader)
    
    # Example features for prediction
    sample_features = {
        'text_length': 150,
        'sentiment_score': -0.2,
        'source_credibility': 0.7,
        'author_credibility': 0.5,
        'citation_count': 3,
        'expert_consensus': 0.8,
        'temporal_features': 0.6,
        'social_engagement': 0.3
    }
    
    # Make ensemble prediction
    result = ensemble.predict(sample_features)
    print("Ensemble prediction result:", result)