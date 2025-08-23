from typing import Type, Dict
from dataclasses import dataclass
from src.summarizers.Bert_summarizer import BERTExtractiveSummarizer
from src.modules.text_preprocessing import TextProcessor
from src.summarizers.LSA_summarizer import LSASummarizer
from src.summarizers.T5_summarizer import T5Summarizer
from src.summarizers.tfidf_summarizer import TFIDFSummarizer
from src.summarizers.textrank_summarizer import TextRankSummarizer
from src.core.base import BaseSummarizer
from src.utils.logging_setup import logger


@dataclass
class SummarizerFactory:
    text_processor: TextProcessor
    _summarizer_instances: Dict[str, BaseSummarizer] = None

    _summarizer_map: Dict[str, Type[BaseSummarizer]] = None

    def __post_init__(self):
        if self._summarizer_instances is None:
            self._summarizer_instances = {}

        if self._summarizer_map is None:
            self._summarizer_map = {
                'tfidf': TFIDFSummarizer,
                'textrank': TextRankSummarizer,
                'lsa': LSASummarizer,
                'bert_extractive': BERTExtractiveSummarizer,
                't5': T5Summarizer,
            }
        logger.info("SummarizerFactory initialized.")

    def get_summarizer(self, method: str) -> BaseSummarizer:
        method_lower = method.lower()

        if method_lower in self._summarizer_instances:
            logger.info(f"Returning cached instance of summarizer: '{method_lower}'.")
            return self._summarizer_instances[method_lower]

        summarizer_cls = self._summarizer_map.get(method_lower)
        if not summarizer_cls:
            logger.error(f"Unknown summarization method requested: '{method_lower}'.")
            raise ValueError(f"Unknown summarization method: {method_lower}")

        logger.info(f"Creating a new instance of summarizer: '{method_lower}'.")
        instance = summarizer_cls(self.text_processor)
        self._summarizer_instances[method_lower] = instance
        return instance
