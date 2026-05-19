import torch
from text_features import TextFeatureExtractor
from text_encoder import TextEncoder
import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"


def main():
    # 1. Initialize the frozen feature extractor (used in Caching phase)
    print("Initializing Extractor...")
    extractor = TextFeatureExtractor(whisper_model_size="base")

    # Simulate a path to a 5-second audio extracted from your video
    # Note: Replace this with a real audio file path to test properly
    sample_audio = "C:\\mbaj\\data\\datasets\\RAVDESS\\Actor_01\\03-01-01-01-01-01-01.wav"

    try:
        print("\n--- Phase 1: Feature Extraction (Caching) ---")
        vector_768, transcribed_text = extractor.process_window(sample_audio)
        print(f"Transcribed Text: '{transcribed_text}'")
        print(f"Raw BERT Feature Shape: {vector_768.shape}")  # Should be (768,)

        # In your actual pipeline, you will save `vector_768` using np.save() here

        print("\n--- Phase 2: Custom Encoder (Training/Inference) ---")
        # 2. Initialize your trainable encoder
        text_encoder = TextEncoder()

        # Convert the cached numpy array back to a PyTorch tensor and add batch dimension
        # Shape becomes (1, 768)
        input_tensor = torch.tensor(vector_768).unsqueeze(0).float()

        # Pass through the trainable encoder
        text_vector_512 = text_encoder(input_tensor)

        print(f"Final Text Vector Shape: {text_vector_512.shape}")  # Should be (1, 512)
        print("Pipeline works perfectly according to MBAJ specs! ✅")

    except FileNotFoundError:
        print("Please provide a valid audio file path to test the Whisper transcription.")


if __name__ == "__main__":
    main()