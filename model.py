import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments
)

class SentimentModel:
    """
    A class to handle sentiment analysis model training using DistilBERT.
    """
    def __init__(self, dataset_path: str, model_checkpoint: str = 'distilbert-base-cased'):
        """
        Initializes the SentimentModel with dataset path and model checkpoint.
        :param dataset_path: Path to the preprocessed dataset CSV file.
        :param model_checkpoint: Pretrained model checkpoint.
        """
        self.dataset_path = dataset_path
        self.checkpoint = model_checkpoint
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.config = self._configure_model()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint, config=self.config
            )
            self.training_args = self._set_training_arguments()
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {e}")
    
    def _configure_model(self) -> AutoConfig:
        """
        Configures model labels mapping.
        :return: Model configuration object.
        """
        try:
            config = AutoConfig.from_pretrained(self.checkpoint)
            target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
            config.id2label = {v: k for k, v in target_map.items()}
            config.label2id = target_map
            return config
        except Exception as e:
            raise RuntimeError(f"Error configuring model: {e}")
    
    def _tokenize(self, batch):
        """
        Tokenizes a batch of sentences.
        :param batch: Dictionary containing 'sentence' key.
        :return: Tokenized batch.
        """
        try:
            return self.tokenizer(batch['sentence'], truncation=True)
        except Exception as e:
            raise RuntimeError(f"Error during tokenization: {e}")
    
    def load_and_preprocess_data(self):
        """
        Loads and tokenizes the dataset.
        :return: Train-test split datasets.
        """
        try:
            raw_dataset = load_dataset('csv', data_files=self.dataset_path)
            split_dataset = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
            return split_dataset.map(self._tokenize, batched=True)
        except Exception as e:
            raise RuntimeError(f"Error loading or preprocessing data: {e}")
    
    def _set_training_arguments(self) -> TrainingArguments:
        """
        Sets up training arguments.
        :return: TrainingArguments object.
        """
        try:
            return TrainingArguments(
                output_dir='./model',
                evaluation_strategy='epoch',
                save_strategy='epoch',
                num_train_epochs=5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
            )
        except Exception as e:
            raise RuntimeError(f"Error setting training arguments: {e}")
    
    def compute_metrics(self, logits_and_labels):
        """
        Computes accuracy and F1 score.
        :param logits_and_labels: Tuple containing model logits and true labels.
        :return: Dictionary with accuracy and F1 score.
        """
        try:
            logits, labels = logits_and_labels
            predictions = np.argmax(logits, axis=-1)
            acc = np.mean(predictions == labels)
            f1 = f1_score(labels, predictions, average='macro')
            return {'accuracy': acc, 'f1': f1}
        except Exception as e:
            raise RuntimeError(f"Error computing metrics: {e}")
    
    def train_model(self):
        """
        Trains the model using the Trainer API.
        """
        try:
            tokenized_datasets = self.load_and_preprocess_data()
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['test'],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )
            trainer.train()
        except Exception as e:
            raise RuntimeError(f"Error during model training: {e}")

# train the model
if __name__ == "__main__":
    sentiment_model = SentimentModel('./data/preprocessed_data.csv')
    sentiment_model.train_model()
    
