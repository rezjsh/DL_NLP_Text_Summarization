
from src.modules.text_preprocessing import TextProcessor
from src.utils.logging_setup import logger

class BaseSummarizer:
    """
    Abstract base class for all summarizers.
    """
    def __init__(self, text_processor: TextProcessor):
        logger.info(f"Initializing BaseSummarizer with processor: {text_processor.__class__.__name__}.")
        self.text_processor = text_processor

    def summarize(self, text: str, **kwargs) -> list[str]:
        raise NotImplementedError("Summarization method not implemented.")