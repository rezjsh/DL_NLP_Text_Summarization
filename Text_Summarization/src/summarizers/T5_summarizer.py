from src.core.base import BaseSummarizer
from src.components.text_preprocessing import TextProcessor
from src.utils.logging_setup import logger
from transformers import pipeline


class T5Summarizer(BaseSummarizer):
    """
    Abstractive summarizer using a pre-trained T5 model from Hugging Face Transformers.
    """
    def __init__(self, text_processor: TextProcessor, model_name: str = "t5-small"):
        super().__init__(text_processor)
        self.model_name = model_name
        self.summarization_pipeline = None
        self._load_model()
        logger.info(f"T5Summarizer initialized with model '{model_name}'.")

    def _load_model(self):
        try:
            self.summarization_pipeline = pipeline("summarization", model=self.model_name)
            logger.info(f"T5 abstractive model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load abstractive T5 model '{self.model_name}': {e}")
            self.summarization_pipeline = None

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> list[str]:
        logger.info(f"Starting T5 abstractive summarization (max_length={max_length}, min_length={min_length}).")
        if self.summarization_pipeline is None:
            return ["T5 summarizer not available due to missing dependencies or loading error."]

        try:
            summary = self.summarization_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            summary_text = summary[0]['summary_text']
            logger.info("T5 abstractive summarization complete.")
            return [summary_text]
        except Exception as e:
            logger.error(f"Error during T5 summarization: {e}")
            return [f"Error during T5 summarization: {e}"]
