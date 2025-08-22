
import numpy as np
from src.components.text_preprocessing import TextProcessor
from src.core.base import BaseSummarizer
from src.utils.logging_setup import logger
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class BERTExtractiveSummarizer(BaseSummarizer):
    """
    Extractive summarizer using BERT embeddings.
    """
    def __init__(self, text_processor: TextProcessor, model_name: str = "bert-base-uncased"):
        super().__init__(text_processor)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model()
        logger.info(f"BERTExtractiveSummarizer initialized with model '{model_name}'.")

    def _load_model(self):
        try:
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            logger.info(f"BERT model '{self.model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Could not load BERT model '{self.model_name}': {e}")
            self.model = None

    def summarize(self, text: str, num_sentences: int = 3) -> list[str]:
        logger.info(f"Starting BERT extractive summarization for {num_sentences} sentences.")
        if self.model is None:
            return ["BERT summarizer not available due to missing dependencies or loading error."]

        original_sentences, _ = self.text_processor.preprocess_text(text)

        if not original_sentences or len(original_sentences) <= num_sentences:
            return original_sentences

        try:
            encoded_input = self.tokenizer(original_sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            centroid = np.mean(sentence_embeddings, axis=0)
            similarity_to_centroid = cosine_similarity(sentence_embeddings, centroid.reshape(1, -1))

            ranked_sentence_indices = np.argsort(similarity_to_centroid.flatten())[::-1]
            top_sentence_indices = ranked_sentence_indices[:num_sentences]

            final_summary = [original_sentences[i] for i in sorted(top_sentence_indices)]
            logger.info(f"BERT extractive summarization complete. Extracted {len(final_summary)} sentences.")
            return final_summary

        except Exception as e:
            logger.error(f"Error during BERT summarization: {e}")
            return [f"Error during BERT summarization: {e}"]
