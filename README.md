# net-fingerprinting
## Group project for COSC 760
**`config/`** (Hydra Configurations)
Stores declarative `.yaml` files to manage project parameters dynamically via Hydra.

* **`models/`**: Hyperparameters, architecture definitions, and tuning ranges for machine learning / DNN models.
* **`data_loading/`**: Paths, batch sizes, download urls.
* **`data_spliting/`**: Strategies and parameters for train/validation/test splits, k-fold cross-validation, or stratification.
* **`evaluation/`**: Definitions for which metrics to calculate, logging preferences.
* **`visualization/`**: Plotting parameters, styling preferences, and export formats for graphs.
* **`default.yaml`**: The primary Hydra entry point that ties all the sub-configs together into a single global configuration object.

**`data/`** (Data Storage)
Organized progressively so raw data is never mutated.

* **`raw/`**: The original, immutable data dump. This should be treated as read-only.
* **`interim/`**: Intermediate data that has been partially cleaned or transformed, useful for debugging the pipeline.
* **`processed/`**: The final, canonical datasets used directly for model training.

**`src/`** (Source Code)
The actual Python modules and scripts that execute the project logic.

* **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA), prototyping, and interactive experimentation.
* **`models/`**: Code defining the model classes, training loops, and inference/prediction functions.
* **`preprocessing/`**: Scripts that clean, encode, and feature-engineer data from the `raw` to the `processed` state.
* **`postprocessing/`**: Logic applied after the model makes predictions, such as formatting outputs, thresholding probabilities, or ensembling.
* **`evaluation/`**: Functions that score model performance against ground truth and generate reports.
* **`visualization/`**: Code to generate charts, feature importance plots, or prediction overlays.
* **`utils/`**: Shared helper functions, custom loggers, decorators, or small tools used across multiple modules.

**Root Files**

* **`main.py`**: The execution entry point for the pipeline (typically wrapped with `@hydra.main`), which orchestrates the config loading and runs the data/training pipeline.
* **`requirements.txt`**: A list of Python packages and versions needed to run the project.
* **`.gitignore`**: Tells Git which files/folders to ignore (e.g., keeping large datasets in `data/` or compiled `__pycache__` out of version control).
* **`README.md`**: The front page of the project explaining what it does, how to install dependencies, and how to run `main.py`.