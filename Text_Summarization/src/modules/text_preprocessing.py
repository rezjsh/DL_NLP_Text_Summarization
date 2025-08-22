from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from src.entity.config_entity import TextProcessingConfig
from src.utils.logging_setup import logger
import string


class TextProcessor:
    """
    Initializes the TextProcessor with specified language for stopwords.
    """
    def __init__(self, language='english'):
        """
        Initializes the TextProcessor with specified language for stopwords.
        Args:
            language (str): The language for stopwords (e.g., 'english').
        """
        logger.info(f"Initializing TextProcessor with language: '{language}'.")
        self.stopwords = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)

    def tokenize_sentences(self, text: str) -> list[str]:
        """
        Tokenizes the input text into individual sentences.
        """
        text = text.replace('\n', ' ').replace('  ', ' ')
        sentences = sent_tokenize(text)
        logger.debug(f"Tokenized text into {len(sentences)} sentences.")
        return sentences

    def tokenize_words(self, sentence: str) -> list[str]:
        """
        Tokenizes a sentence into words, converts them to lowercase,
        and removes stopwords and punctuation.
        """
        words = word_tokenize(sentence)
        filtered_words = [
            word.lower() for word in words
            if word.lower() not in self.stopwords and word not in self.punctuation
        ]
        logger.debug(f"Filtered {len(words)} words down to {len(filtered_words)}.")
        return filtered_words

    def preprocess_text(self, text: str) -> tuple[list[str], list[list[str]]]:
        """
        Applies the full preprocessing pipeline to the input text.
        """
        logger.info("Starting text preprocessing...")
        sentences = self.tokenize_sentences(text)
        processed_sentences_words = [self.tokenize_words(s) for s in sentences]
        logger.info("Text preprocessing complete.")
        return sentences, processed_sentences_words
