# Text Categorization

A machine learning project to categorize medical text using a pre-trained BERT-based model. The project demonstrates how to train a model for text classification using the HuggingFace Transformers library, with an emphasis on fine-tuning BERT and DistilBERT for text categorization tasks.

## Project Description

The goal of this project is to classify medical text into various medical specialties. The model is based on the BERT and DistilBERT architectures, leveraging their transformer-based language understanding for high accuracy in text classification.

Key features of the project:
- Text classification using BERT and DistilBERT.
- Hyperparameter tuning using Optuna for optimizing model performance.
- Evaluation of the model using accuracy, confusion matrix, and classification report.

## Dataset

The dataset used for this project is medical text data, specifically focusing on text categorization for medical specialties. You can place your CSV dataset in the data/raw/ directory. The data should have at least three columns:

description: The medical text.
transcription: The transcribed medical data.
medical_specialty: The category label (e.g., cardiology, oncology, etc.).
You can modify the dataset path in the main.py or hyperparameter_tuning.py scripts as needed.


## Installation and setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Janani0608/text_categorization.git
   cd text_categorization

2. **Setup a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

4. **Training the model:**
   ```bash
   python -m scripts.main

5. **Hyperparameter tuning:**
   ```bash
   python hyperparameter_tuning.py

6. **Evaluating the model:**  

   After training, the script will generate a confusion matrix and a classification report, which will be saved as confusion_matrix.png and printed to the terminal.