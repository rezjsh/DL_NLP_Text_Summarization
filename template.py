import pathlib
import logging
import traceback
from typing import List, Dict, Optional


# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def create_directories(dirs: List[str]) -> None:
    """
    Create a list of directories, including parent directories.
    Logs success or failure for each directory.
    """
    for dir_path_str in dirs:
        path = pathlib.Path(dir_path_str)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory (or verified exists): {path}")
        except OSError as e:
            logging.error(f"Failed to create directory {path}: {e}")
            logging.debug(traceback.format_exc())


def create_files(files: List[str], initial_contents: Optional[Dict[str, str]] = None) -> None:
    """
    Create files, ensuring parent directories exist.
    Optionally initialize files with specified content if empty.
    Logs success or failure for each file.
    """
    initial_contents = initial_contents or {}
    for file_path_str in files:
        file_path = pathlib.Path(file_path_str)
        try:
            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # If file does not exist, create it and write initial content if provided
            if not file_path.exists():
                file_path.touch()
                if file_path.name in initial_contents:
                    file_path.write_text(initial_contents[file_path.name], encoding='utf-8')
                    logging.info(f"Created file {file_path} with initial content.")
                else:
                    logging.info(f"Created empty file: {file_path}")
            else:
                # If file exists and is empty, write initial content if provided
                if file_path.stat().st_size == 0 and file_path.name in initial_contents:
                    file_path.write_text(initial_contents[file_path.name], encoding='utf-8')
                    logging.info(f"Added initial content to existing empty file: {file_path}")
                else:
                    logging.info(f"File already exists: {file_path}")

        except OSError as e:
            logging.error(f"Failed to create or write to file {file_path}: {e}")
            logging.debug(traceback.format_exc())
        except Exception as e:
            logging.error(f"Unexpected error for file {file_path}: {e}")
            logging.debug(traceback.format_exc())


def update_gitignore(gitignore_path: pathlib.Path, lines_to_add: List[str]) -> None:
    """
    Create or update a .gitignore file to include specified lines.
    Adds only missing lines without duplicating.
    Logs actions taken.
    """
    try:
        if not gitignore_path.exists():
            gitignore_path.write_text("\n".join(lines_to_add) + "\n", encoding='utf-8')
            logging.info(f"Created {gitignore_path} with initial content.")
            return

        existing_content = gitignore_path.read_text(encoding='utf-8')
        existing_lines = set(line.strip() for line in existing_content.splitlines() if line.strip())

        missing_lines = [line for line in lines_to_add if line and line not in existing_lines]

        if missing_lines:
            with gitignore_path.open("a", encoding='utf-8') as f:
                for line in missing_lines:
                    f.write(f"\n{line}")
                    logging.info(f"Added '{line}' to {gitignore_path}.")
            logging.info(f"Updated {gitignore_path} with missing lines.")
        else:
            logging.info(f"No updates needed for {gitignore_path}.")

    except OSError as e:
        logging.error(f"Failed to create or update {gitignore_path}: {e}")
        logging.debug(traceback.format_exc())
    except Exception as e:
        logging.error(f"Unexpected error updating {gitignore_path}: {e}")
        logging.debug(traceback.format_exc())


def main():
    PROJECT_NAME = "Text_Summarization"

    dirs_to_create = [
        PROJECT_NAME,
        f"{PROJECT_NAME}/config",
        f"{PROJECT_NAME}/data/01_raw",
        f"{PROJECT_NAME}/data/02_interim",
        f"{PROJECT_NAME}/data/03_processed",
        f"{PROJECT_NAME}/data/04_external",
        f"{PROJECT_NAME}/docs",
        f"{PROJECT_NAME}/logs",
        f"{PROJECT_NAME}/models/evaluation",
        f"{PROJECT_NAME}/notebooks",
        f"{PROJECT_NAME}/reports/figures",
        f"{PROJECT_NAME}/tests",
        f"{PROJECT_NAME}/src/data",
        f"{PROJECT_NAME}/src/features",
        f"{PROJECT_NAME}/src/models",
        f"{PROJECT_NAME}/src/modules",
        f"{PROJECT_NAME}/src/evaluation",
        f"{PROJECT_NAME}/src/utils",
        f"{PROJECT_NAME}/src/components",
        f"{PROJECT_NAME}/src/pipeline",
        f"{PROJECT_NAME}/src/constants",
        f"{PROJECT_NAME}/src/config",
        f"{PROJECT_NAME}/src/entity",
    ]

    files_to_create = [
        # Config
        f"{PROJECT_NAME}/config/config.yaml",
        f"{PROJECT_NAME}/config/logging_config.yaml",
        # Src package init files and modules
        f"{PROJECT_NAME}/src/__init__.py",
        f"{PROJECT_NAME}/src/data/__init__.py",
        f"{PROJECT_NAME}/src/data/make_dataset.py",
        f"{PROJECT_NAME}/src/features/__init__.py",
        f"{PROJECT_NAME}/src/features/build_features.py",
        f"{PROJECT_NAME}/src/models/__init__.py",
        f"{PROJECT_NAME}/src/models/train_model.py",
        f"{PROJECT_NAME}/src/models/predict_model.py",
        f"{PROJECT_NAME}/src/evaluation/__init__.py",
        f"{PROJECT_NAME}/src/evaluation/evaluate.py",
        f"{PROJECT_NAME}/src/utils/__init__.py",
        f"{PROJECT_NAME}/src/utils/logging_setup.py",
        f"{PROJECT_NAME}/src/utils/helpers.py",
        f"{PROJECT_NAME}/src/utils/file_io.py",
        f"{PROJECT_NAME}/src/utils/exceptions.py",
        # Test files
        f"{PROJECT_NAME}/tests/__init__.py",
        f"{PROJECT_NAME}/tests/test_data_processing.py",
        f"{PROJECT_NAME}/tests/test_feature_engineering.py",
        f"{PROJECT_NAME}/tests/test_model_training.py",
        f"{PROJECT_NAME}/tests/test_utils.py",
        # Root files
        f"{PROJECT_NAME}/.env.example",
        f"{PROJECT_NAME}/Dockerfile",
        f"{PROJECT_NAME}/environment.yml",
        f"{PROJECT_NAME}/main.py",
        "requirements.txt",
        "setup.py",
        # Components
        f"{PROJECT_NAME}/src/components/__init__.py",
        f"{PROJECT_NAME}/src/components/data_ingestion.py",
        f"{PROJECT_NAME}/src/components/data_validation.py",
        f"{PROJECT_NAME}/src/components/data_transformation.py",
        f"{PROJECT_NAME}/src/components/model_trainer.py",
        f"{PROJECT_NAME}/src/components/model_evaluation.py",
        # Pipeline
        f"{PROJECT_NAME}/src/pipeline/__init__.py",
        f"{PROJECT_NAME}/src/pipeline/stage_01_data_ingestion.py",
        f"{PROJECT_NAME}/src/pipeline/stage_02_data_validation.py",
        f"{PROJECT_NAME}/src/pipeline/stage_03_data_transformation.py",
        f"{PROJECT_NAME}/src/pipeline/stage_04_model_trainer.py",
        f"{PROJECT_NAME}/src/pipeline/stage_05_model_evaluation.py",
        # Constants
        f"{PROJECT_NAME}/src/constants/__init__.py",
        f"{PROJECT_NAME}/src/constants/constants.py",
        # Config
        f"{PROJECT_NAME}/src/config/__init__.py",
        f"{PROJECT_NAME}/src/config/configuration.py",
        # params.yaml
        f"{PROJECT_NAME}/params.yaml",
        # Entity
        f"{PROJECT_NAME}/src/entity/__init__.py",
        f"{PROJECT_NAME}/src/entity/config_entity.py",
    ]

    # Initial content for specific files that should have a header
    initial_file_contents = {
        "README.md": f"# {PROJECT_NAME}\n",
        "config.yaml": "# Add your project configuration here\n",
        "logging_config.yaml": "# Logging configuration\n",
        "params.yaml": "# Parameters configuration\n",
    }

    gitignore_content_lines = [
        "# Standard Python ignores...",
        "__pycache__/",
        "*.py[cod]",
        "*.so",
        "",
        "# Environment stuff...",
        ".env",
        ".venv",
        "env/",
        "venv/",
        "",
        "# Data (usually managed outside git or with LFS/DVC)",
        "# data/",
        "",
        "# Logs",
        "logs/",
        "*.log",
        "",
        "# Models (usually large)",
        "models/*.pkl",
        "models/*.h5",
        "models/*.onnx",
        "",
        "# Notebook checkpoints",
        ".ipynb_checkpoints",
        "",
        "# IDE folders",
        ".vscode/",
        ".idea/",
        "",
        "# OS files",
        ".DS_Store",
        "Thumbs.db",
    ]

    logging.info(f"Starting project structure creation for: {PROJECT_NAME}")

    # Create directories
    create_directories(dirs_to_create)

    # Create files
    create_files(files_to_create)

    # Handle .gitignore in the root directory
    update_gitignore(pathlib.Path(".gitignore"), gitignore_content_lines)

    # Ensure README.md in root with project title exists or initialized
    readme_path = pathlib.Path("README.md")
    try:
        if not readme_path.exists() or readme_path.stat().st_size == 0:
            readme_path.write_text(initial_file_contents.get("README.md", f"# {PROJECT_NAME}\n"), encoding='utf-8')
            logging.info(f"Created or updated README.md with project title.")
        else:
            logging.info(f"README.md already exists and is not empty.")
    except Exception as e:
        logging.error(f"Failed to create or write to README.md: {e}")
        logging.debug(traceback.format_exc())

    logging.info("Project structure creation process finished.")


if __name__ == "__main__":
    main()
