import streamlit as st
import joblib
import re
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "reasoning_model.pkl"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Page configuration
st.set_page_config(
    page_title="Reasoning Trajectory Detection",
    page_icon="🧠",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Header
st.title("🧠 Reasoning Trajectory Detection")

st.write(
    "An AI/ML application that predicts whether a generated "
    "answer is correct or incorrect using Machine Learning."
)

st.divider()

# Sidebar
st.sidebar.title("Project Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Dashboard", "Prediction", "Model Performance", "About Project"]
)

# Dashboard
if page == "Dashboard":

    st.header("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Trajectories", "2,393")
    col2.metric("Unique Questions", "399")
    col3.metric("Model Accuracy", "62.37%")

    st.subheader("Project Overview")

    st.write(
        """
        This project uses Natural Language Processing and Machine Learning
        techniques to analyze generated answers and predict their correctness.

        **Technologies Used:**
        - Python
        - Pandas and NumPy
        - TF-IDF Vectorization
        - Logistic Regression
        - Scikit-learn
        - Streamlit
        """
    )

# Prediction
elif page == "Prediction":

    st.header("Answer Prediction")

    question = st.text_area(
        "Enter Your Question",
        placeholder="Type your question here..."
    )

    reasoning = st.text_area(
        "Enter Generated Reasoning / Answer",
        placeholder="Type the generated reasoning or answer here..."
    )

    if st.button("Predict", use_container_width=True):

        if not question.strip() or not reasoning.strip():

            st.warning("Please enter both question and reasoning.")

        else:

            model_text = (
                "question: " + question +
                " reasoning: " + reasoning
            )

            model_text = model_text.lower()
            model_text = re.sub(r"\s+", " ", model_text).strip()

            prediction = model.predict([model_text])[0]
            probabilities = model.predict_proba([model_text])[0]

            classes = list(model.classes_)

            correct_probability = probabilities[classes.index(1)] * 100
            incorrect_probability = probabilities[classes.index(0)] * 100

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("Prediction: Correct Answer ✅")
            else:
                st.error("Prediction: Incorrect Answer ❌")

            col1, col2 = st.columns(2)

            col1.metric(
                "Correct Answer Probability",
                f"{correct_probability:.2f}%"
            )

            col2.metric(
                "Incorrect Answer Probability",
                f"{incorrect_probability:.2f}%"
            )

            st.info(
                "Note: This is a machine learning prediction based on "
                "text patterns. It does not verify the logical validity "
                "of each reasoning step."
            )

# Model Performance
elif page == "Model Performance":

    st.header("Model Performance")

    st.write("Logistic Regression evaluation results:")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "62.37%")
    col2.metric("Precision", "72.56%")
    col3.metric("Recall", "71.65%")
    col4.metric("F1 Score", "72.10%")

    st.subheader("Project Visualizations")

    graph_files = {
        "Label Distribution": "label_distribution.png",
        "Model Comparison": "model_comparison.png",
        "Confusion Matrix": "confusion_matrix.png",
        "Prediction Results": "prediction_results.png"
    }

    for title, filename in graph_files.items():

        graph_path = OUTPUTS_DIR / filename

        if graph_path.exists():
            st.subheader(title)
            st.image(str(graph_path), use_container_width=True)
        else:
            st.warning(f"{filename} was not found.")

# About Project
elif page == "About Project":

    st.header("About the Project")

    st.write(
        """
        **Project Title:** Reasoning Trajectory Detection

        **Objective:**
        To develop a Machine Learning application that predicts
        the correctness of generated answers using question and
        reasoning text.

        **Dataset:**
        ThoughtSource

        **Machine Learning Model:**
        - Logistic Regression
        - TF-IDF Vectorization

        **Project Workflow:**
        1. Dataset Collection
        2. Exploratory Data Analysis
        3. Data Cleaning
        4. Text Preprocessing
        5. Feature Extraction
        6. Model Training
        7. Model Evaluation
        8. Streamlit Deployment
        """
    )

    st.subheader("Important Limitation")

    st.write(
        "The model predicts final-answer correctness. "
        "It does not independently verify whether every reasoning "
        "step is logically valid."
    )

# Footer
st.divider()

st.caption(
    "Developed using Python | Scikit-learn | TF-IDF | "
    "Logistic Regression | Streamlit"
)