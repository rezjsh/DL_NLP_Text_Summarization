# src/utils/nltk_resources.py
# This module handles NLTK data downloads and dependency imports.
import nltk
from src.utils.logging_setup import logger

# Global variables to store imported libraries
transformers_pipeline = None
rouge_scorer = None
sentence_bleu = None

def download_nltk_resources():
    """
    Downloads necessary NLTK corpora if they are not already present.
    """
    try:
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK stopwords are already downloaded.")
    except LookupError:
        logger.warning("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
        logger.info("NLTK stopwords downloaded successfully.")

    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer is already downloaded.")
    except LookupError:
        logger.warning("NLTK punkt tokenizer not found. Downloading...")
        nltk.download('punkt')
        logger.info("NLTK punkt tokenizer downloaded successfully.")

    # Explicitly download punkt_tab to resolve persistent LookupError in some environments
    try:
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK punkt_tab tokenizer is already downloaded.")
    except LookupError:
        logger.warning("NLTK punkt_tab tokenizer not found. Downloading...")
        nltk.download('punkt_tab')
        logger.info("NLTK punkt_tab tokenizer downloaded successfully.")


def import_optional_libraries():
    """
    Imports and stores optional deep learning and evaluation libraries.
    """
    global transformers_pipeline, rouge_scorer, sentence_bleu

    # Try to import transformers library
    if transformers_pipeline is None:
        try:
            from transformers import pipeline
            transformers_pipeline = pipeline
            logger.info("The 'transformers' library was imported successfully.")
        except ImportError:
            logger.warning("The 'transformers' library is not installed. T5/BERT summarizers will be disabled.")
    
    # Try to import evaluation libraries
    if rouge_scorer is None or sentence_bleu is None:
        try:
            from rouge_score import rouge_scorer as rs
            from nltk.translate.bleu_score import sentence_bleu as sb
            rouge_scorer = rs
            sentence_bleu = sb
            logger.info("The 'rouge_score' and 'nltk.bleu_score' libraries were imported successfully.")
        except ImportError:
            logger.warning("The 'rouge_score' or 'nltk' library is not installed. The evaluation feature will be disabled.")
