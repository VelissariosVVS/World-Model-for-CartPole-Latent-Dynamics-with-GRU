"""
Main entry point for the world model project.

Provides a menu interface to:
    1. Train the autoencoder world model
    2. Evaluate the trained model
    3. Exit
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python's path so modules can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # =========================================================
    # IMPORT TRAINING AND EVALUATION SCRIPTS
    # =========================================================
    from train_autoencoder import train_autoencoder
    from evaluation_world_model import evaluate_world_model

    print("\n" + "=" * 50)
    print("   CARTPOLE WORLD MODEL")
    print("=" * 50)
    print("This project implements a latent dynamics world model")
    print("for the CartPole environment using an encoder-decoder")
    print("architecture with GRU-based latent transition.\n")

    # Define file paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_PATH = PROJECT_ROOT / "processed_cartpole_dynamics.pt"
    MODEL_PATH = PROJECT_ROOT / "world_model.pt"

    while True:
        print("\n--- Menu ---")
        print("1. Train world model")
        print("2. Evaluate world model")
        print("3. Exit")
        option = input("Select option (1/2/3): ").strip()

        if option == '1':
            # Check if data exists
            if not DATA_PATH.exists():
                print(f"\n❌ Error: Data file not found at {DATA_PATH}")
                print("   Please run data_preprocessing.ipynb first.")
                print("   The notebook should save the file to the project root.")
                continue

            try:
                train_autoencoder()
            except FileNotFoundError as e:
                print(f"\n❌ File error: {e}")
            except Exception as e:
                print(f"\n❌ Unexpected error during training: {e}")

        elif option == '2':
            # Check if model exists
            if not MODEL_PATH.exists():
                print(f"\n❌ Error: Model file not found at {MODEL_PATH}")
                print("   Please train the model first (option 1).")
                continue

            try:
                evaluate_world_model()
            except FileNotFoundError as e:
                print(f"\n❌ File error: {e}")
            except Exception as e:
                print(f"\n❌ Unexpected error during evaluation: {e}")

        elif option == '3':
            print("\nExiting. Goodbye!")
            break

        else:
            print("\nInvalid option. Please enter 1, 2, or 3.")