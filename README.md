# Forensic Glass Type Classification

**A machine learning application for classifying glass samples based on refractive index and chemical composition.**

🚀 **[Try the Live Application](https://forensic-glass-classifier.streamlit.app/)** | **[View Source Code on GitHub](https://github.com/Aswani-T-A/forensic-glass-classifier)**

## Project Overview

Glass fragments recovered from crime scenes can provide useful forensic information. This project explores machine learning methods for classifying glass samples using their measured properties.

The project includes exploratory data analysis, data visualization, model comparison, evaluation metrics, and an interactive web application for predictions.

## Live Application
![Glass prediction interface](<glass-prediction 1.png>)
![Glass prediction result](<glass-prediction 2.png>)

Use the deployed application to enter glass measurements and obtain a model prediction directly in your browser. No local Python installation is required.

**Application:** https://forensic-glass-classifier.streamlit.app/

## Features

- Exploratory data analysis of the glass dataset
- Glass class distribution visualization
- Correlation heatmap and feature analysis
- Comparison of machine learning models
- Confusion matrix and per-class F1-score visualization
- Feature importance analysis
- Saved model for prediction
- Interactive Streamlit prediction interface

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Dataset

This project uses the Glass Identification dataset from the UCI Machine Learning Repository. It contains 214 glass samples with refractive-index and chemical-composition measurements.

**Original dataset:** [UCI Machine Learning Repository — Glass Identification](https://doi.org/10.24432/C5WW2P)

**Citation:**

German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P

Refer to the original dataset page for the current licence and attribution terms.

## Project Structure

```text
forensic-glass-classifier/
├── app.py
├── glass_classifier.py
├── glass.csv
├── glass_model.joblib
├── requirements.txt
├── results.txt
└── figures/
    ├── 01_class_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_boxplots.png
    ├── 04_model_comparison.png
    ├── 05_confusion_matrix.png
    ├── 06_feature_importance.png
    └── 07_per_class_f1.png
```

## Run Locally

To run the project on your own computer:

1. Clone the repository:

   ```bash
   git clone https://github.com/Aswani-T-A/forensic-glass-classifier.git
   ```

2. Enter the project directory:

   ```bash
   cd forensic-glass-classifier
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Model Evaluation

The project generates visualizations for class distribution, feature relationships, model comparison, confusion matrix, feature importance, and per-class F1-scores.

See `results.txt` for the results produced by the current implementation. Actual performance depends on the model, preprocessing, and evaluation procedure.

## References

1. German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Pandas documentation: https://pandas.pydata.org/docs/
4. Streamlit documentation: https://docs.streamlit.io/

## Disclaimer

This is an educational machine learning project. Predictions should not be treated as definitive forensic evidence or a substitute for professional forensic laboratory analysis.

## Author

Developed as a machine learning assignment.

- **Live application:** https://forensic-glass-classifier.streamlit.app/
- **GitHub repository:** https://github.com/Aswani-T-A/forensic-glass-classifier
