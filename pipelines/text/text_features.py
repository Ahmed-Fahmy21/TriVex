import torch
import whisper
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings("ignore")


class TextFeatureExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", whisper_model_size="base"):
        self.device = device

        # 1. Load Whisper Model (Speech-to-Text) - Frozen
        print(f"Loading Whisper '{whisper_model_size}' model...")
        self.whisper_model = whisper.load_model(whisper_model_size).to(self.device)

        # 2. Load BERT Model (Text-to-Features) - Frozen
        print("Loading BERT model...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()  # Ensure it's in evaluation mode (Frozen)

    def extract_text_from_audio(self, audio_path_or_array):
        """
        Uses Whisper to transcribe a 5-second audio chunk.
        """
        # Whisper automatically handles 16kHz resampling internally if you pass a file path
        result = self.whisper_model.transcribe(audio_path_or_array, fp16=torch.cuda.is_available())
        text = result["text"].strip()
        return text if text else "[SILENCE]"

    def get_bert_cls_embedding(self, text):
        """
        Passes the text through BERT and returns the [CLS] token (768-dim).
        """
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():  # No training for BERT
            outputs = self.bert_model(**inputs)

        # Extract [CLS] token, which is the first token of the last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
        return cls_embedding.squeeze().cpu().numpy()  # Return as numpy array for caching

    def process_window(self, audio_path_or_array):
        """
        Full pipeline for a single 5-second window.
        Returns a 768-dim numpy array.
        """
        # Step 1: Audio to Text
        text = self.extract_text_from_audio(audio_path_or_array)

        # Step 2: Text to 768-dim Vector
        vector_768 = self.get_bert_cls_embedding(text)

        return vector_768, text  # Returning text as well might be useful for debugging