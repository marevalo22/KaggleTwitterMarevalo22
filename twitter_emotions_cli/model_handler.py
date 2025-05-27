import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .preprocessing import basic_clean # Relative import from preprocessing.py within the same package

# Determine the absolute path to the directory containing this script
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the path to the trained model files relative to this script's location
DEFAULT_MODEL_DIR = os.path.join(_CURRENT_DIR, "trained_model_files")
# MAX_LENGTH as used during training
MODEL_MAX_LENGTH = 128

class EmotionModel:
    def __init__(self, model_dir_path: str = None):
        """
        Initializes the EmotionModel.

        Args:
            model_dir_path (str, optional): Path to the directory containing
                                            the saved model, tokenizer, and label classes.
                                            Defaults to a 'trained_model_files' subdirectory
                                            next to this script.
        """
        if model_dir_path is None:
            model_dir_path = DEFAULT_MODEL_DIR

        if not os.path.isdir(model_dir_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir_path}. "
                "Ensure the model files are in the 'trained_model_files' subdirectory "
                "or provide the correct path."
            )

        print(f"Loading model from: {model_dir_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir_path)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode

            label_classes_path = os.path.join(model_dir_path, "label_encoder_classes.npy")
            self.label_classes = np.load(label_classes_path, allow_pickle=True)
            print("Model, tokenizer, and label classes loaded successfully.")
            print(f"Available emotion labels: {self.label_classes}")

        except Exception as e:
            print(f"Error loading model components from {model_dir_path}: {e}")
            raise

    def predict(self, text_input: str or list[str]) -> str or list[str]:
        """
        Predicts emotion(s) for the given text input(s).

        Args:
            text_input (str or list[str]): A single text string or a list of text strings.

        Returns:
            str or list[str]: The predicted emotion label string or a list of them,
                              corresponding to the input type.
        """
        if not text_input:
            return [] if isinstance(text_input, list) else ""

        is_single_input = isinstance(text_input, str)
        texts_to_process = [text_input] if is_single_input else text_input

        # Clean texts
        cleaned_texts = [basic_clean(text) for text in texts_to_process]

        # Tokenize
        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODEL_MAX_LENGTH # Use the same max_length as training
        )

        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions_indices = torch.argmax(logits, dim=-1).cpu().numpy()

        # Map indices to labels
        predicted_labels = [self.label_classes[idx] for idx in predictions_indices]

        return predicted_labels[0] if is_single_input else predicted_labels

if __name__ == '__main__':
    # This section is for testing the EmotionModel class directly.
    # It assumes 'trained_model_files' is in the same directory as this script,
    # or adjacent to it if this script is run from its parent directory.
    print("\n--- Testing EmotionModel ---")
    try:
        # When testing locally, the DEFAULT_MODEL_DIR will be relative to this file's location.
        # If your 'trained_model_files' folder is in:
        # twitter_emotions_package/twitter_emotions_cli/trained_model_files
        # and this script (model_handler.py) is in:
        # twitter_emotions_package/twitter_emotions_cli/model_handler.py
        # then DEFAULT_MODEL_DIR should work.
        
        classifier = EmotionModel() # Uses DEFAULT_MODEL_DIR

        sample_texts = [
            "I am so happy and excited about this news!",
            "This is really sad and disappointing.",
            "I feel so angry right now I could scream.",
            "Feeling a lot of love for my friends today.",
            "That movie was surprisingly scary!"
        ]
        
        print("\nPredicting on sample texts:")
        for text in sample_texts:
            prediction = classifier.predict(text)
            print(f"Text: \"{text}\" -> Predicted Emotion: {prediction}")

        # Test batch prediction
        batch_predictions = classifier.predict(sample_texts)
        print("\nBatch predictions:")
        for text, pred in zip(sample_texts, batch_predictions):
            print(f"Text: \"{text}\" -> Predicted Emotion: {pred}")

    except FileNotFoundError as e:
        print(f"Testing Error: {e}")
        print("Make sure the 'trained_model_files' directory is correctly placed relative to 'model_handler.py' for direct testing.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        