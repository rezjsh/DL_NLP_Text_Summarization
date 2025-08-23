# --- Flask Application Setup ---
from flask import Flask, render_template, request, flash, redirect, url_for
import sys
import pathlib

from Text_Summarization.src.components.summarizer import Summarizer
# from src.components.summarizer import Summarizer


app_dir = pathlib.Path(__file__).parent.resolve()
templates_path = app_dir / "Text_Summarization" / "app" / "templates"
static_path = app_dir / "Text_Summarization" / "app" / "static"


app = Flask(__name__,
    template_folder=str(templates_path), 
    static_folder=str(static_path)
    )
app.secret_key = 'super_secret_key'
summarizer = Summarizer(language='english')

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main page for the summarizer."""
    original_text = ""
    summary = ""
    selected_method = "tfidf"
    num_sentences = 3

    if request.method == 'POST':
        original_text = request.form.get('text_input', '').strip()
        selected_method = request.form.get('summarizer_method', 'tfidf')
        num_sentences = int(request.form.get('num_sentences', 3))

        if not original_text:
            flash("Please provide some text to summarize.", 'error')
            return render_template('index.html', original_text=original_text, summary=summary, method=selected_method, num_sentences=num_sentences)
        kwargs = {
            "text": original_text,
            "method": selected_method
        }

        try:
            if selected_method in ['tfidf', 'lsa', 'bert', 'textrank']:
                summary_list = summarizer.summarize_text(
                    num_sentences=num_sentences,
                    **kwargs
            )
            elif selected_method in ['t5']:
                summary_list = summarizer.summarize_text(
                    max_length=num_sentences * 30,
                    min_length=num_sentences * 10,
                    **kwargs)
                
            else:
                # Fallback if method not recognized
                summary_list = summarizer.summarize_text(
                    text=original_text,
                    method=selected_method
                )
            summary = " ".join(summary_list)
            
            if "Error:" in summary:
                flash(summary, 'error')
                summary = ""
            else:
                flash(f"Summarization successful using {selected_method.upper()}.", 'success')

        except Exception as e:
            flash(f"An unexpected error occurred: {e}", 'error')
            print(f"ERROR during summarization: {e}", file=sys.stderr)
    
    return render_template('index.html', original_text=original_text, summary=summary, method=selected_method, num_sentences=num_sentences)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
