import argparse
import os
# Assuming model_handler.py is in the same directory (twitter_emotions_cli)
from .model_handler import EmotionModel
# You can import __version__ if you defined it in your __init__.py
# from . import __version__

def run():
    """
    Main function to parse arguments and run emotion prediction.
    This will be the entry point for the CLI tool.
    """
    parser = argparse.ArgumentParser(
        description="Emotion Classification CLI Tool.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    # parser.add_argument(
    #     '-v', '--version',
    #     action='version',
    #     # version=f'%(prog)s {__version__}' # Requires __version__ to be imported
    #     version=f'%(prog)s 0.1.0' # Placeholder if not importing __version__
    # )
    parser.add_argument(
        "text",
        type=str,
        help="The text string to classify for emotion."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None, # EmotionModel will use its internal default path if this is None
        help=(
            "Optional: Path to the directory containing the saved model, "
            "tokenizer, and label classes.\nDefaults to the 'trained_model_files' "
            "subdirectory within the installed package."
        )
    )

    args = parser.parse_args()

    try:
        # Instantiate the EmotionModel.
        # If args.model_dir is None, EmotionModel uses its default path,
        # which is 'trained_model_files' relative to model_handler.py.
        classifier = EmotionModel(model_dir_path=args.model_dir)
        
        predicted_emotion = classifier.predict(args.text)
        
        print(f"Input Text: \"{args.text}\"")
        print(f"Predicted Emotion: {predicted_emotion}")

    except FileNotFoundError as e:
        print(f"ERROR: Model files not found. {e}")
        print("Please ensure that the 'trained_model_files' directory (containing model.safetensors, "
              "config.json, tokenizer files, and label_encoder_classes.npy) "
              "is correctly placed within the 'twitter_emotions_cli' package directory, "
              "or provide the correct path using the --model_dir argument.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # This allows you to run the script directly for testing the CLI logic,
    # e.g., python main.py "some text"
    # However, relative imports (like '.model_handler') might cause issues
    # if you run it as `python twitter_emotions_cli/main.py` from the parent directory.
    # It's best tested after installation or using `python -m twitter_emotions_cli.main "some text"`
    run()