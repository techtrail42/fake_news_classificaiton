import os
import pickle
import numpy as np
import pandas as pd
import re
import spacy
import requests
import lime
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import string
import nltk

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TensorFlow after setting environment variables
import tensorflow as tf

# Suppress additional warnings
tf.get_logger().setLevel('ERROR')


import tensorflow as tf

class AttentionSum(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionSum, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

def load_model_and_tokenizer():
    """Load the pre-trained LSTM model, tokenizer, and calibrator"""
    try:
        # Load model with custom objects
        model = tf.keras.models.load_model(
            'complete_fake_news_model.h5',
            custom_objects={'AttentionSum': AttentionSum}
        )
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        with open('isotonic_calibrator.pkl', 'rb') as f:
            calibrator = pickle.load(f)
        
        # Return all three objects
        return model, tokenizer, calibrator
    except Exception as e:
        raise Exception(f"Error loading model, tokenizer, or calibrator: {str(e)}")

def load_spacy_model():
    try:
        # Use the large model for a balance of speed and accuracy
        nlp = spacy.load("en_core_web_lg")
        return nlp
    except OSError:
        # If the model isn't installed, download it automatically
        print("Downloading spaCy 'en_core_web_lg' model... This may take a moment.")
        from spacy.cli import download
        download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
        return nlp
    except Exception as e:
        raise Exception(f"Error loading spaCy model: {str(e)}")

def load_free_summarizer():
    """Load free Hugging Face summarizer"""
    try:
        from transformers import pipeline
        
        # free, lightweight model that works well for summarization
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",  
            device=-1,  # Use CPU to avoid GPU requirements
            framework="pt"
        )
        return summarizer
    except ImportError:
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "transformers", "torch", "torchvision", "torchaudio"])
            from transformers import pipeline
            summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1,
                framework="pt"
            )
            return summarizer
        except Exception as e:
            raise Exception(f"Could not install transformers: {str(e)}")
    except Exception as e:
        try:
            from transformers import pipeline
            # Fallback to a smaller, faster model
            summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                device=-1,
                framework="pt"
            )
            return summarizer
        except Exception as e2:
            raise Exception(f"Could not load any summarizer: {str(e2)}")

def expand_contractions(text):
    """Expand contractions in text"""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def clean_text(text):
    if pd.isna(text) or text == '':
        return ''

    # Ensure text is a string and convert to lowercase
    text = str(text).lower()

    # Expand contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are",
        "'ve": " have", "'ll": " will", "'d": " would", "'m": " am",
        "let's": "let us", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "isn't": "is not", "wasn't": "was not",
        "weren't": "were not", "haven't": "have not", "hasn't": "has not",
        "hadn't": "had not", "wouldn't": "would not",
        "shouldn't": "should not", "couldn't": "could not",
        "mustn't": "must not"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove URLs, email addresses, and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)

    # Remove source artifacts like "21st Century Wire" or "via Reuters"
    text = re.sub(r'(\d+)(st|nd|rd|th)\s*century\s*wire', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*via\s+[\w\s.-]+', '', text)
    text = re.sub(r'pic.twitter.com/\w+', '', text)

    # Tokenize the text, which will separate words from punctuation
    tokens = word_tokenize(text)

    # Initialize lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Add specific words to the stop word list that might introduce bias
    additional_stopwords = {
        'reuters', 'breitbart', 'cnn', 'fox', 'bbc', 'nbc', 'abc', 'cbs',
        'associated', 'press', 'ap', 'bloomberg', 'wall', 'street', 'journal',
        'new', 'york', 'times', 'washington', 'post', 'guardian',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december',
        'said', 'according', 'reported', 'also', 'u', 'image', 'images', 'video', 'videos',
        'file', 'files', 'tag', 'tags', 'loading', 'download', 'nyp', 'rasmussen', 'vice',
        'pic', 'via', 'twitter', 'news'
    }
    stop_words.update(additional_stopwords)

    cleaned_tokens = []
    for token in tokens:
        # Keep only alphabetic tokens
        if token.isalpha():
            lemma = lemmatizer.lemmatize(token)
            # Check length and stopwords AFTER lemmatizing
            if len(lemma) > 1 and lemma not in stop_words:
                cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)

def extract_content_features(text):
    """Extract content-based features from original text"""
    if not text:
        return [0, 0, 0, 0, 0, 0]
    words = text.split()
    if len(words) == 0:
        return [0, 0, 0, 0, 0, 0]
    exclamation_ratio = text.count('!') / len(text)
    question_ratio = text.count('?') / len(text)
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
    avg_word_length = sum(len(word) for word in words) / len(words)
    sentence_count = len(re.split(r'[.!?]+', text))
    avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0
    return [exclamation_ratio, question_ratio, caps_ratio,
            avg_word_length, sentence_count, avg_sentence_length]


def preprocess_article(title, text, tokenizer):
    """
    Prepares a single article for prediction by cleaning, combining,
    tokenizing, and padding it exactly as done in the training pipeline.
    """
    # Extract content features from the ORIGINAL, uncleaned text
    original_combined_text = title + ' ' + text
    content_features = np.array([extract_content_features(original_combined_text)]) # Shape: (1, 6)

    # Clean title and text separately using the new clean_text function
    clean_title = clean_text(title)
    clean_body = clean_text(text)  

    # Combine title and text with a 2x weight for the title, mimicking training
    weighted_title = ' '.join([clean_title] * 2)
    combined_cleaned_text = (weighted_title + ' ' + clean_body).strip()

    # Tokenize the final combined text
    sequences = tokenizer.texts_to_sequences([combined_cleaned_text])

    # Pad the sequences to the model's required length (500)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=500, padding='post', truncating='post'
    )

    # Return all four necessary components
    # The variable `clean_body` is now included in the return statement.
    return padded_sequences, combined_cleaned_text, clean_body, content_features

def predict_article(model, calibrator, inputs): 
    """Make prediction using the LSTM model and calibrate the probability"""
    try:
        # Make a prediction with both text and feature inputs
        raw_prediction = model.predict(inputs, verbose=0) # 'inputs' is now a list
        raw_confidence = float(raw_prediction[0][0])

        calibrated_confidence = calibrator.predict([raw_confidence])[0]
        
        # Determine the final class and confidence percentage
        is_real = calibrated_confidence > 0.5
        confidence_percentage = calibrated_confidence * 100 if is_real else (1 - calibrated_confidence) * 100
        
        # Return both calibrated confidence and raw confidence for potential logging or display
        return is_real, confidence_percentage, calibrated_confidence, raw_confidence
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")


def extract_entities(text, nlp):
    """Extract named entities from text using spaCy with improved logic"""
    if not nlp:
        return []
    
    doc = nlp(text)
    entities = []
    
    # list of relevant entity types for news articles
    allowed_labels = ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'LAW', 'WORK_OF_ART', 'FAC', 'LOC']
    
    # Stopwords for filtering out generic entities
    stop_words = set(stopwords.words('english'))

    for ent in doc.ents:
        # Check if entity type is relevant
        if ent.label_ in allowed_labels:
            # Clean the entity text
            clean_ent = ent.text.strip()
            
            # Filter out very short, long, or single-word entities that are common words
            if clean_ent and 1 < len(clean_ent) < 50 and clean_ent.lower() not in stop_words:
                entities.append(clean_ent)
    
    # Remove duplicates while preserving order and return the top 5
    unique_entities = list(dict.fromkeys(entities))
    return unique_entities[:5] # Return the top 5 most prominent entities

def search_news_api(query, api_key):
    """Search for related articles using NewsAPI"""
    if not api_key:
        return []
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'sortBy': 'relevancy',
            'language': 'en',
            'pageSize': 3,
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        else:
            return []
    except Exception as e:
        return []

def summarize_with_free_model(text, summarizer):
    """Summarize text using free Hugging Face model"""
    if not summarizer:
        # Fallback to simple extraction-based summarization
        return extract_key_sentences(text, num_sentences=4)
    
    try:
        # Prepare text for summarization
        # Truncate text to fit model limits (BART can handle up to 1024 tokens)
        text = text[:1000]  
        
        # Count approximate tokens (rough estimate: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        
        # Ensure minimum length for summarization
        if len(text.split()) < 15:
            return text
        
        # Dynamic length adjustment based on input size
        if estimated_tokens < 50:
            # Very short text - just return as is or minimal summary
            return text
        elif estimated_tokens < 100:
            # Short text - conservative summary
            max_len = min(estimated_tokens - 5, 80)
            min_len = min(30, max_len - 10)
        else:
            # Longer text - can afford longer summaries
            max_len = min(estimated_tokens // 2, 180)  # At most half the input length
            min_len = min(60, max_len - 20)
        
        # Ensure min_len is not greater than max_len
        min_len = max(20, min(min_len, max_len - 10))
        
        # Generate summary with dynamic length
        summary = summarizer(
            text, 
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        
        return summary[0]['summary_text']
    except Exception as e:
        # Fallback to simple extraction with more sentences
        return extract_key_sentences(text, num_sentences=4)

def extract_key_sentences(text, num_sentences=4):
    """Simple extractive summarization as fallback"""
    sentences = text.split('.')
    # Clean up sentences and remove empty ones
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= num_sentences:
        return text
    
    # Return first few sentences as a simple summary
    return '. '.join(sentences[:num_sentences]) + '.'

def create_lime_explanation(model, tokenizer, combined_cleaned_text, content_features, class_names=['Fake', 'Real']):
    """Create LIME explanation for the prediction."""
    try:
        # Define the prediction function that LIME will use
        def predict_fn(texts):
            # This ensures they are processed just like the training data
            cleaned_texts_for_lime = [clean_text(text) for text in texts]

            # Tokenize and pad the cleaned texts
            sequences = tokenizer.texts_to_sequences(cleaned_texts_for_lime)
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=500, padding='post', truncating='post'
            )

            # The content features remain constant for all of LIME's text perturbations
            num_samples = len(texts)
            content_features_repeated = np.repeat(content_features, num_samples, axis=0)

            # Make predictions
            predictions = model.predict([padded, content_features_repeated], verbose=0)
            
            # Return probabilities for both classes [prob_fake, prob_real]
            return np.column_stack([1 - predictions, predictions])

        explainer = LimeTextExplainer(class_names=class_names, random_state=42)

        # Generate the explanation on the already cleaned and combined text
        explanation = explainer.explain_instance(
            combined_cleaned_text,
            predict_fn,
            num_features=15,  # Matches the Colab notebook
            num_samples=1000  # Matches the Colab notebook
        )
        return explanation
    except Exception as e:
        print(f"LIME explanation error: {str(e)}")
        return None

def create_lime_visualizations(explanation, prediction_class):
    """Create LIME visualizations with custom styling for a dark theme"""
    if not explanation:
        return None, None

 
    features = explanation.as_list()
    words = [f[0] for f in features]
    importance = [f[1] for f in features]

    # Use theme-appropriate colors for the bars
    bar_colors = ['#ef4444' if imp < 0 else '#10b981' for imp in importance]

    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=words,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{imp:.3f}" for imp in importance],
            textposition='auto',
            textfont_color='white'
        )
    ])

    fig.update_layout(
        title=f"Top Words Influencing Prediction: '{prediction_class}'",
        xaxis_title="Importance Score",
        yaxis_title="Words",
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',      # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',     # Transparent background
        font_color='#e2e8f0',              # Light text for title and labels
        xaxis=dict(gridcolor='#475569'),
        yaxis=dict(autorange="reversed", gridcolor='#475569')
    )


    final_html = generate_custom_lime_html(explanation)

    return fig, final_html


def generate_custom_lime_html(explanation):
    """
    Generates a custom HTML for highlighted text from a LIME explanation.
    This bypasses the built-in as_html method to avoid versioning issues.
    """
    # Get the processed text and the feature weights from the explanation
    text = explanation.domain_mapper.indexed_string.raw
    exp = explanation.as_list()
    
    # Create a dictionary mapping a feature (word) to its weight
    word_weights = {word: weight for word, weight in exp}

    # Use regex to split the text by spaces and punctuation but keep them
    parts = re.split(r'(\W+)', text)
    
    html_parts = []
    for part in parts:
        weight = word_weights.get(part.lower())
        if weight:
            # Apply a background color based on the weight
            # Redish for "fake" contributors (negative weight)
            # Greenish for "real" contributors (positive weight)
            if weight < 0:
                color = "rgb(255, 170, 153)" # Light red
            else:
                color = "rgb(153, 255, 153)" # Light green
            
            # Wrap the word in a span with the style
            html_parts.append(f'<span style="background:{color}; color: white; padding: 2px 4px; border-radius: 3px;">{part}</span>')
        else:
            # If the part is not in our explanation, append it as is
            html_parts.append(part)
            
    return f'<div style="color: #e2e8f0; line-height: 1.6;">{"".join(html_parts)}</div>'