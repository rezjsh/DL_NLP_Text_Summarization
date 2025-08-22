
from src.modules.text_preprocessing import TextProcessor
from src.factory.summarizer_factory import SummarizerFactory
from src.utils.logging_setup import logger

class Summarizer:
    """
    The main orchestrator class for text summarization.
    """
    def __init__(self, language='english'):
        logger.info(f"Initializing Summarizer with language: '{language}'.")
        self.text_processor = TextProcessor(language)
        self.summarizer_factory = SummarizerFactory(self.text_processor)

    def summarize_text(self, text: str, method: str, **kwargs) -> list[str]:
        """
        Summarizes the given text using the specified method.
        """
        logger.info(f"Requesting summary using method: '{method}'.")
        try:
            summarizer = self.summarizer_factory.get_summarizer(method)
            summary = summarizer.summarize(text, **kwargs)
            logger.info(f"Summary generated successfully using '{method}'.")
            return summary
        except ValueError as e:
            logger.error(f"Failed to summarize text: {e}")
            return [f"Error: {e}"]
