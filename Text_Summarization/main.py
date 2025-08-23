# This script demonstrates and evaluates different summarization methods.
import json
import sys
import os
from src.components.evaluate import SummarizationEvaluator
from src.utils.nltk_resources import download_nltk_resources
from src.components.summarizer import Summarizer
from src.utils.logging_setup import logger

# Ensure the project root is in the Python path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


download_nltk_resources()
def evaluate_and_log_summary(summarizer_instance: Summarizer,
                             evaluator_instance: SummarizationEvaluator,
                             text: str,
                             method: str,
                             reference: str,
                             **kwargs) -> tuple[str, dict]:
    """
    Generates a summary, evaluates it, and logs the results.
    Returns the generated summary and a dictionary of the evaluation metrics.
    """
    logger.info("="*60)
    logger.info(f"--- Demonstrating {method.upper()} Summarization ---")
    logger.info("="*60)

    summary = ""
    metrics = {"method": method}

    try:
        summary_list = summarizer_instance.summarize_text(text, method=method, **kwargs)
        summary = " ".join(summary_list) if isinstance(summary_list, list) else summary_list
        logger.info(f"Generated Summary ({method.upper()}): {summary}")

        if not summary or "Error:" in summary:
            logger.warning("Could not evaluate summary due to a model error or empty summary.")
            metrics["evaluation_status"] = "failed"
            metrics["error_message"] = summary
        elif not evaluator_instance.available:
             logger.warning("Evaluation skipped because libraries are not available.")
             metrics["evaluation_status"] = "skipped"
             metrics["message"] = "Evaluation libraries not installed."
        else:
            logger.info("\n--- Evaluation Results ---")
            evaluation_scores = evaluator_instance.evaluate_summary(summary, reference)
            for metric_name, score in evaluation_scores.items():
              if isinstance(score, (int, float)):
                logger.info(f"    - {metric_name.upper()}: {score:.4f}")
              else:
                logger.info(f"    - {metric_name.upper()}: {score}")


            metrics.update(evaluation_scores)
            metrics["evaluation_status"] = "success"

    except Exception as e:
        logger.error(f"An error occurred during {method} summarization: {e}", exc_info=True)
        metrics["evaluation_status"] = "failed"
        metrics["error_message"] = str(e)

    logger.info("\n")
    return summary, metrics


def save_results_to_json(results_data: dict, file_path: str):
    """
    Saves a dictionary of evaluation results to a JSON file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logger.info(f"Evaluation results saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save results to JSON file: {e}", exc_info=True)


if __name__ == "__main__":
    sample_text = """
    A solar eclipse is a celestial event where the Moon passes between the Sun and
    Earth, and the Moon fully or partially blocks the Sun. This can only happen at
    new moon when the Sun, Moon and Earth are in alignment or nearly so. In a total
    eclipse, the Moon completely covers the Sun's disk, and the sky darkens as if it
    were twilight. During the total phase, observers can see the Sun's corona, which
    is normally obscured by the bright light of the Sun itself.

    There are four types of solar eclipses: total, partial, annular, and hybrid.
    A partial eclipse occurs when the Moon only partially obscures the Sun. An annular
    eclipse happens when the Moon's size is not large enough to completely cover the Sun,
    leaving a "ring of fire" visible around the Moon. A hybrid eclipse is a rare type
    that shifts between a total and annular eclipse depending on the observer's location.

    Observing a solar eclipse requires special eye protection, as looking directly at
    the Sun can cause permanent eye damage. Safe viewing methods include using eclipse
    glasses or creating a pinhole projector.
    """

    reference_summary = "A solar eclipse happens when the Moon blocks the Sun from Earth. There are four types of solar eclipses: total, partial, annular, and hybrid. Observing an eclipse requires special eye protection to avoid permanent damage."



    # Instantiate the summarizer and evaluator
    summarizer = Summarizer(language='english')
    evaluator = SummarizationEvaluator()

    # Define the summarization methods and their specific arguments
    evaluation_methods = {
        't5': {'max_length': 60, 'min_length': 20},
        'textrank': {'num_sentences': 3},
        'lsa': {'num_sentences': 3},
        'bert_extractive': {'num_sentences': 3}
    }

    all_results = {}
    summaries = {}

    for method, kwargs in evaluation_methods.items():
        summary, metrics = evaluate_and_log_summary(summarizer, evaluator, sample_text, method, reference_summary, **kwargs)
        all_results[method] = metrics
        summaries[method] = summary

    # Save all the collected results to a JSON file
    results_file = "evaluation_results.json"
    save_results_to_json(all_results, results_file)

    # Print the final, formatted results to the console.
    print("\n" + "="*60)
    print("--- Final Evaluation Results Summary ---")
    print("="*60)
    print(json.dumps(all_results, indent=4))
    print("="*60)


    # Print the generated summaries
    print("\n" + "="*60)
    print("--- Generated Summaries ---")
    print("="*60)
    for method, summary_text in summaries.items():
        print(f"--- {method.upper()} Summary ---")
        print(summary_text)
        print("-" * (len(method) + 10)) # Separator based on method name length
    print("="*60)