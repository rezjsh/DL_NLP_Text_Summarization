
from src.core.base import BaseSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from src.modules.text_preprocessing import TextProcessor
from src.utils.logging_setup import logger

class TextRankSummarizer(BaseSummarizer):
    """
    Extractive summarizer using the TextRank algorithm.
    """
    def __init__(self, text_processor: TextProcessor):
        super().__init__(text_processor)
        self.vectorizer = TfidfVectorizer()
        logger.info("TextRankSummarizer initialized.")

    def _build_similarity_matrix(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.array([])

        sentence_vectors = self.vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        np.fill_diagonal(similarity_matrix, 0)
        logger.debug(f"Built similarity matrix of shape: {similarity_matrix.shape}")
        return similarity_matrix

    def summarize(self, text: str, num_sentences: int = 3) -> list[str]:
        logger.info(f"Starting TextRank summarization for {num_sentences} sentences.")
        original_sentences, _ = self.text_processor.preprocess_text(text)

        if not original_sentences:
            logger.warning("No sentences found in the input text. Returning an empty summary.")
            return []

        if len(original_sentences) <= num_sentences:
            logger.info("Number of sentences requested is greater than or equal to the total sentences. Returning all.")
            return original_sentences

        similarity_matrix = self._build_similarity_matrix(original_sentences)
        if similarity_matrix.size == 0:
            logger.warning("Similarity matrix is empty. Not enough sentences to build a graph.")
            return []

        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
        top_sentences_set = set(sentence for _, sentence in ranked_sentences[:num_sentences])
        
        final_summary = [s for s in original_sentences if s in top_sentences_set]
        
        logger.info(f"TextRank summarization complete. Extracted {len(final_summary)} sentences.")
        return final_summary[:num_sentences]
