import mlflow
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml,save_json,create_directories
from mlflow.models.signature import infer_signature
import numpy as np
import dagshub
dagshub.init(repo_owner='shreyojitdas95', repo_name='mlops-end-to-end', mlflow=True)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _create_valid_generator(self):
        """Set up the validation data generator."""
        datagenerator_args = {
            "rescale": 1. / 255,
            "validation_split": 0.30
        }

        dataflow_args = {
            "target_size": self.config.params_image_size[:-1],
            "batch_size": self.config.params_batch_size,
            "interpolation": "bilinear"
        }

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_args)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_args
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Load a pre-trained model from the specified path."""
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """Evaluate the model and save the score."""
        self.model = self.load_model(self.config.path_of_model)
        self._create_valid_generator()
        
        # Evaluate the model and save score
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """Save the evaluation score to a JSON file."""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """Log the model and metrics to MLflow."""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log metrics
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # Generate an input example using a dummy image
            input_example = np.random.rand(1, *self.config.params_image_size)  # Shape (1, 224, 224, 3)
            
            # Auto-generate model signature
            signature = infer_signature(input_example, self.model.predict(input_example))

            if tracking_url_type != "file":
                mlflow.tensorflow.log_model(self.model, "model", 
                                            registered_model_name="VGG16Model",
                                            signature=signature, 
                                            input_example=input_example)
            else:
                mlflow.tensorflow.log_model(self.model, "model", 
                                            signature=signature, 
                                            input_example=input_example)