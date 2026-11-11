# Reasoning Trajectory Detection

An AI/ML application that predicts whether a generated answer is correct or incorrect using Natural Language Processing and Machine Learning.

## Project Overview

This project analyzes generated answers using question and reasoning text to predict final-answer correctness.

It uses the ThoughtSource dataset and a Logistic Regression classifier with TF-IDF feature extraction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib
- Seaborn
- Streamlit

## Dataset

Source: ThoughtSource

The project uses four datasets:

- CommonsenseQA
- OpenBookQA
- StrategyQA
- WorldTree

Total cleaned trajectories: 2,393

Unique question groups: 399

## Project Workflow

1. Dataset Collection
2. Exploratory Data Analysis
3. Data Cleaning
4. Text Preprocessing
5. Feature Extraction using TF-IDF
6. Model Training
7. Model Evaluation
8. Error Analysis
9. Data Visualization
10. Streamlit Application Development

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 62.37% |
| Precision | 72.56% |
| Recall | 71.65% |
| F1 Score | 72.10% |

## Application Features

- Dataset overview dashboard
- Interactive answer prediction
- Correct and incorrect prediction probabilities
- Model performance metrics
- Confusion matrix and other visualizations
- Project information

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/malinibathala-debug/Reasoning-Trajectory-Detection.git
```

### 2. Navigate to the project folder

```bash
cd Reasoning_Trajectory_Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

## Important Limitation

The model predicts final-answer correctness based on text patterns. It does not independently verify the logical validity of every reasoning step.

## Future Improvements

- Use larger reasoning datasets.
- Explore advanced NLP and transformer-based models.
- Improve classification performance.
- Develop methods for evaluating individual reasoning steps.

## Author

Developed as an AI/ML project using Python, Scikit-learn, TF-IDF, and Streamlit.