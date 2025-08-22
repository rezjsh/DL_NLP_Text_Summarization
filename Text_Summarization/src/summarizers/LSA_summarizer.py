import numpy as np
from sklearn.decomposition import TruncatedSVD
from src.modules.text_preprocessing import TextProcessor
from src.core.base import BaseSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logging_setup import logger

class LSASummarizer(BaseSummarizer):
    """
    Extractive summarizer using Latent Semantic Analysis (LSA).
    """
    def __init__(self, text_processor: TextProcessor):
        super().__init__(text_processor)  # Corrected line: added parentheses
        self.vectorizer = TfidfVectorizer()
        logger.info("LSASummarizer initialized.")

    def summarize(self, text: str, num_sentences: int = 3) -> list[str]:
        logger.info(f"Starting LSA summarization for {num_sentences} sentences.")
        original_sentences, _ = self.text_processor.preprocess_text(text)

        if not original_sentences:
            logger.warning("No sentences found in the input text. Returning an empty summary.")
            return []

        if len(original_sentences) <= num_sentences:
            logger.info("Number of sentences requested is greater than or equal to the total sentences. Returning all.")
            return original_sentences

        if len(original_sentences) < 2:
            logger.warning("Not enough sentences for LSA. Returning all available sentences.")
            return original_sentences

        tfidf_matrix = self.vectorizer.fit_transform(original_sentences)
        n_components = min(num_sentences, tfidf_matrix.shape[0] - 1)

        if n_components <= 0:
            logger.warning("Not enough sentences for SVD. Returning first sentences.")
            return original_sentences[:num_sentences]

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        transformed_sentences = svd.fit_transform(tfidf_matrix)

        sentence_scores = np.abs(transformed_sentences[:, 0])
        ranked_sentence_indices = np.argsort(sentence_scores)[::-1]
        top_sentence_indices = ranked_sentence_indices[:num_sentences]

        final_summary = [original_sentences[i] for i in sorted(top_sentence_indices)]
        logger.info(f"LSA summarization complete. Extracted {len(final_summary)} sentences.")
        return final_summary
