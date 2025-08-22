from collections import defaultdict
from src.components.text_preprocessing import TextProcessor
from src.core.base import BaseSummarizer
from src.utils.logging_setup import logger

class TFIDFSummarizer(BaseSummarizer):
    """
    Extractive summarizer using a custom TF-IDF approach.
    """
    def __init__(self, text_processor: TextProcessor):
        super().__init__(text_processor)
        logger.info("TFIDFSummarizer initialized.")

    def _calculate_word_frequencies(self, sentences_words: list[list[str]]) -> dict[str, int]:
        logger.debug("Calculating word frequencies...")
        word_frequencies = defaultdict(int)
        for sentence_words in sentences_words:
            for word in sentence_words:
                word_frequencies[word] += 1
        logger.debug(f"Found {len(word_frequencies)} unique words.")
        return word_frequencies

    def _calculate_sentence_scores(self, original_sentences: list[str],
                                   sentences_words: list[list[str]],
                                   word_frequencies: dict) -> dict[str, float]:
        logger.debug("Calculating sentence scores...")
        sentence_scores = defaultdict(float)
        max_freq = max(word_frequencies.values()) if word_frequencies else 1

        for i, sentence_word_list in enumerate(sentences_words):
            for word in sentence_word_list:
                sentence_scores[original_sentences[i]] += word_frequencies[word] / max_freq

        logger.debug(f"Calculated scores for {len(sentence_scores)} sentences.")
        return sentence_scores

    def summarize(self, text: str, num_sentences: int = 3) -> list[str]:
        logger.info(f"Starting TF-IDF summarization for {num_sentences} sentences.")
        original_sentences, processed_sentences_words = self.text_processor.preprocess_text(text)

        if not original_sentences:
            logger.warning("No sentences found in the input text. Returning an empty summary.")
            return []

        if len(original_sentences) <= num_sentences:
            logger.info("Number of sentences requested is greater than or equal to the total sentences. Returning all.")
            return original_sentences

        word_frequencies = self._calculate_word_frequencies(processed_sentences_words)
        sentence_scores = self._calculate_sentence_scores(original_sentences, processed_sentences_words, word_frequencies)

        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences_set = set(sentence for sentence, _ in ranked_sentences[:num_sentences])

        final_summary = [s for s in original_sentences if s in top_sentences_set]

        logger.info(f"TF-IDF summarization complete. Extracted {len(final_summary)} sentences.")
        return final_summary[:num_sentences]
