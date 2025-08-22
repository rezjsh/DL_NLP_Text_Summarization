
import re
from src.utils.logging_setup import logger
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class SummarizationEvaluator:
    """
    A class to evaluate the quality of a generated summary against a reference summary.
    """
    def __init__(self):
        try:
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.bleu_smoothing = SmoothingFunction().method4
            self.available = True
            logger.info("SummarizationEvaluator initialized successfully.")
        except ImportError:
            logger.error("Evaluation libraries (rouge_score or nltk) not found. Evaluation will be disabled.")
            self.available = False

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def evaluate_summary(self, generated_summary: str, reference_summary: str) -> dict:
        """
        Computes ROUGE and BLEU scores for a generated summary.
        """
        if not self.available:
            logger.warning("Evaluation libraries are not available. Skipping evaluation.")
            return {"evaluation_status": "skipped", "message": "Evaluation libraries not installed."}

        logger.info("Starting evaluation of generated summary.")
        logger.debug(f"Generated: '{generated_summary}'")
        logger.debug(f"Reference: '{reference_summary}'")

        try:
            rouge_results = self.scorer.score(reference_summary, generated_summary)
            metrics = {
                "rouge1": rouge_results["rouge1"].fmeasure,
                "rouge2": rouge_results["rouge2"].fmeasure,
                "rougeL": rouge_results["rougeL"].fmeasure
            }

            reference_tokens = self._normalize_text(reference_summary).split()
            candidate_tokens = self._normalize_text(generated_summary).split()

            # Handle cases where candidate or reference tokens are empty
            if not candidate_tokens or not reference_tokens:
                 bleu_score = 0.0
                 logger.warning("Empty candidate or reference tokens for BLEU calculation. BLEU score set to 0.0.")
            else:
                 bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.bleu_smoothing)

            metrics["bleu"] = bleu_score
            metrics["evaluation_status"] = "success"
            logger.info("Evaluation complete. Metrics calculated.")
            return metrics
        except Exception as e:
            logger.error(f"Error during summary evaluation: {e}", exc_info=True)
            return {"evaluation_status": "error", "error_message": str(e)}

