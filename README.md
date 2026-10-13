# Forensic Glass Type Classification

A machine learning project that classifies glass samples based on their refractive index and chemical composition. The project explores data analysis, model comparison, classification performance, and prediction through a Python application.

## Project Overview

Glass fragments recovered from crime scenes may provide useful forensic evidence. This project uses machine learning techniques to explore the classification of glass samples based on their measured properties.

The project includes exploratory data analysis, visualization, model evaluation, and a saved classification model.

## Features

- Exploratory data analysis of the glass dataset
- Glass class distribution visualization
- Correlation heatmap and feature analysis
- Comparison of machine learning models
- Confusion matrix and per-class F1-score visualization
- Feature importance analysis
- Saved model for reuse
- Interactive application using Streamlit (if applicable to the current implementation)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit (for the interactive application)

## Dataset

This project uses the Glass Identification dataset from the UCI Machine Learning Repository.

The dataset contains 214 glass samples and measurements such as refractive index and the chemical composition of the glass. The target variable represents the glass category.

**Original dataset:** https://doi.org/10.24432/C5WW2P

Dataset citation:

German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P.

The dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Please refer to the original dataset page for its licence and attribution requirements.

## Project Structure

```text
forensic-glass-classifier/
├── app.py
├── glass_classifier.py
├── glass.csv
├── glass_model.joblib
├── requirements.txt
├── results.txt
├── ML_ASSIGNMENT_2.pdf
└── figures/
    ├── 01_class_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_boxplots.png
    ├── 04_model_comparison.png
    ├── 05_confusion_matrix.png
    ├── 06_feature_importance.png
    └── 07_per_class_f1.png
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Aswani-T-A/forensic-glass-classifier.git
   ```

2. Move into the project directory:

   ```bash
   cd forensic-glass-classifier
   ```

3. Create and activate a Python virtual environment if desired.

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

Run the analysis script:

```bash
python glass_classifier.py
```

If `app.py` is the Streamlit application, launch it with:

```bash
streamlit run app.py
```

Use the generated figures and `results.txt` to review the analysis and model evaluation results.

## Model Evaluation

The project generates visualizations to examine class distribution, feature relationships, model comparison, confusion matrix, feature importance, and per-class F1-scores.

Refer to `results.txt` for the actual results produced by the current implementation. Performance depends on the selected model, preprocessing, and evaluation procedure.

## References

1. German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P.
2. Scikit-learn developers. *Scikit-learn Documentation: Model Selection and Evaluation*. https://scikit-learn.org/stable/model_selection.html
3. Scikit-learn developers. *Scikit-learn Documentation: Metrics*. https://scikit-learn.org/stable/api/sklearn.metrics.html
4. Pandas documentation: https://pandas.pydata.org/docs/
5. Streamlit documentation: https://docs.streamlit.io/

## Disclaimer

This project is an educational machine learning assignment. Its predictions should not be treated as definitive forensic evidence or a substitute for professional forensic laboratory analysis.

## Author

Developed as a machine learning assignment.

GitHub repository: https://github.com/Aswani-T-A/forensic-glass-classifier