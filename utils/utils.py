import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import seaborn as sns



# Ensure required NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Converting to lowercase
    - Adding spaces between numbers and letters (e.g., '5mg' -> '5 mg')
    - Removing numbers
    - Expanding contractions (e.g., "can't" -> "can not")
    - Removing punctuation and special characters
    - Removing stopwords

    Args:
    text (str): Input string to preprocess.

    Returns:
    str: Preprocessed text.
    """

    # Convert to lower case
    text = text.lower()

    # Add space between numbers and letters (e.g., '5mg' -> '5 mg', '17yo' -> '17 yo')
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation and special characters (keep alphanumeric and spaces only)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens into a single string
    text = ' '.join(tokens)

    return text



def get_sentence_embedding(sent, model):
    '''
    create embeddings by calculating mean of vectors of words in each review (preprocessed_text)
    '''
    list_vectors = []
    flag=False
    
    if hasattr(model, 'get_word_vector'):
        flag=True

    if(flag):
        for word in sent:
            vector = model.get_word_vector(word)
            list_vectors.append(vector)
    else:
        for word in sent:
            vector = model.wv[word]
            list_vectors.append(vector)

    mean_vector = np.array(list_vectors).mean(axis=0)
    return mean_vector


# Creating "embeddings" column
def get_embedding_df(df, embedding_dim, model):
    '''
    returns df with embedded columns. flag indicates if its fasttext
    '''
    #df['embeddings'] = df['preprocessed_text'].apply(lambda x: get_sentence_embedding(x.split(), model))
    embeddings = df['content_preprocessed'].apply(lambda x: get_sentence_embedding(x.split(), model))
    

    #creating a column for each vector in embedding - 100 columns
    cols = [f'e_{i}' for i in range(1, embedding_dim + 1)]
    
    # Convert 'embeddings' column to a DataFrame and concatenate
    embeddings_df = pd.DataFrame(embeddings.tolist(), index=df.index, columns=cols)
    df = pd.concat([df, embeddings_df], axis=1)

    return (df.copy(), cols)


def evaluate_model(estimator, X_train, y_train, X_test, y_test, display_metrics=False):
    """
    Evaluate a classification model by calculating metrics and plotting confusion matrices.

    Parameters:
    - estimator: Trained classification model.
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Testing features.
    - y_test: Testing labels.

    Returns:
    - metrics_dict: Dictionary containing accuracy, precision, recall, and F1 score for both train and test data.
    """
    model_name = str(estimator).split('(')[0]
    
    # Predictions
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    
    # Metrics Calculation
    metrics_dict = [
        {
            'Estimator': model_name,
            'Dataset': 'Train', 
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred, average='macro'),
            'Recall': recall_score(y_train, y_train_pred, average='macro'),
            'F1 Score': f1_score(y_train, y_train_pred, average='macro')
        },
        {
            'Estimator': model_name,
            'Dataset': 'Test',
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred, average='macro'),
            'Recall': recall_score(y_test, y_test_pred, average='macro'),
            'F1 Score': f1_score(y_test, y_test_pred, average='macro')
        }
    ]

    if(display_metrics):
        
        # Print Metrics
        print("Classification Report (Train):\n", classification_report(y_train, y_train_pred))
        print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))
    
        # Confusion Matrices
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
    
        # Plot Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title("Train Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
    
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
        axes[1].set_title("Test Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
    
        plt.tight_layout()
        plt.show()

    return metrics_dict
