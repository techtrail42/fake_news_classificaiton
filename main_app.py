import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from langdetect import detect, LangDetectException
import re

import nltk
nltk.download('punkt')

BIASED_WORDS = {
    'reuters', 'breitbart', 'cnn', 'fox', 'bbc', 'nbc', 'abc', 'cbs',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'washington', 'berlin', 'york', 'london', 'paris', 'moscow', 'beijing',
    'said', 'times', 'news', 'reported', 'according'
}

# Define the BiasAwareTokenizer class
class BiasAwareTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.biased_words = BIASED_WORDS

    def fit_on_texts(self, texts):
        # First, do normal fitting
        super().fit_on_texts(texts)
        # Then remove biased words from word_index
        words_to_remove = [word for word in self.word_index if word.lower() in self.biased_words]
        for word in words_to_remove:
            del self.word_index[word]
            if word in self.word_counts:
                del self.word_counts[word]
        # Rebuild word_index with consecutive indices
        sorted_words = sorted(self.word_index.items(), key=lambda x: x[1])
        self.word_index = {word: i+1 for i, (word, _) in enumerate(sorted_words)}
        print(f"Removed {len(words_to_remove)} biased words from vocabulary")

# Existing imports from utils
from utils import (
    load_model_and_tokenizer,
    load_spacy_model,
    load_free_summarizer,
    preprocess_article,
    predict_article,
    extract_entities,
    search_news_api,
    summarize_with_free_model,
    create_lime_explanation,
    create_lime_visualizations
)

# Page configuration
st.set_page_config(
    page_title="NewsGuard AI - Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid #475569;
    }
    
    /* Header Brand */
    .brand-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .brand-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .brand-header p {
        color: #e2e8f0;
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Premium Cards */
    .premium-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Input Section */
    .input-section {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .input-section h3 {
        color: #f1f5f9;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction Results */
    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4);
    }

    .lime-positive {
        background-color: #60a5fa !important; /* Brighter blue */
        color: #ffffff !important; /* White text */
        padding: 2px 4px;
        border-radius: 3px;
    }

    .lime-positive[style*="background: rgb(255, 102, 0)"] {
        background-color: #ffaa33 !important; /* Lighter orange */
        color: #000000 !important; /* Black text */
        font-weight: 600;
    }

    .lime-text {
        font-size: 1.1rem !important; /* Larger font */
        line-height: 1.8 !important; /* More spacing */
        color: #f1f5f9 !important; /* Light gray-white text */
    }

    .lime-text-container {
        background: rgba(30, 41, 59, 0.95) !important; /* Darker background for better contrast */
        padding: 1rem;
        border-radius: 8px;
        color: #f1f5f9 !important; /* Light gray-white text */
    }

    /* Add specific styling for LIME highlighted text */
    .lime-text-container span {
        color: #f1f5f9 !important; /* Ensure all text is light */
    }

    /* Style for negative importance words (contributing to fake) */
    .lime-text-container span[style*="background-color: rgb(255"] {
        color: #1f2937 !important; /* Dark text on light background */
        font-weight: 600;
    }

    /* Style for positive importance words (contributing to real) */
    .lime-text-container span[style*="background-color: rgb(0"] {
        color: #ffffff !important; /* White text on dark background */
        font-weight: 600;
    }

    .lime-text-container::-webkit-scrollbar {
        width: 12px;
    }

    .lime-text-container::-webkit-scrollbar-thumb {
        background: #93c5fd !important; /* Brighter scrollbar */
    }

    .real-news {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .fake-news {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .uncertain-news {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Warning Box */
    .warning-card {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 500;
    }
    
    /* Status Cards */
    .status-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Summary Cards */
    .summary-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .summary-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .summary-card h4 {
        color: #f1f5f9;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .summary-card p {
        color: #cbd5e1;
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    .summary-card a {
        color: #60a5fa;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .summary-card a:hover {
        color: #93c5fd;
    }
    
    /* Progress Bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Section Headers */
    .section-header {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 1.5rem;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 16px 16px 0 0;
        text-align: center;
        color: #94a3b8;
    }
    
    .footer-brand {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .footer-tech {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 1rem;
    }
    
    .footer-copyright {
        font-size: 0.8rem;
        opacity: 0.7;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    /* Sidebar Content */
    .sidebar-content {
        padding: 1rem;
        color: #e2e8f0;
    }
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-info {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-info h4 {
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .sidebar-info p {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    /* Loading Spinner */
    .spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Fade-in Animation */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    /* Custom styles for Streamlit tabs */
    button[data-baseweb="tab"] {
        font-size: 1.1rem !important; /* Larger font size */
        padding: 0.75rem 1.5rem !important; /* More padding */
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.1) !important;
        color: #93c5fd !important; /* Brighter color for selected tab */
        border-bottom: 3px solid #667eea !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #1e293b;
        color: #e2e8f0;
        text-align: center;
        border-radius: 8px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Character counter */
    .char-counter {
        text-align: right;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    .char-counter.warning {
        color: #f59e0b;
    }

    .char-counter.error {
        color: #ef4444;
    }
    /* Hide Streamlit elements */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        model, tokenizer, calibrator = load_model_and_tokenizer()
        return model, tokenizer, calibrator
    except Exception as e:
        st.error(f"Error loading model, tokenizer, or calibrator: {str(e)}")
        st.error("Please ensure 'complete_fake_news_model.h5', 'tokenizer.pkl', and 'isotonic_calibrator.pkl' are in the same directory.")
        return None, None, None

@st.cache_resource
def load_nlp_model():
    """Load spaCy model for entity extraction"""
    try:
        nlp = load_spacy_model()
        return nlp
    except Exception as e:
        st.error(f"Error loading spaCy model: {str(e)}")
        st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")
        return None

@st.cache_resource
def load_summarizer():
    """Load free Hugging Face summarizer"""
    try:
        summarizer = load_free_summarizer()
        return summarizer
    except Exception as e:
        st.warning(f"Could not load BART summarizer, trying lighter model: {str(e)}")
        return None

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-header">
                <i class="fas fa-shield-alt"></i>
                NewsGuard AI
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # About section
        st.markdown("""
        <div class="sidebar-info">
            <h4><i class="fas fa-info-circle"></i> About</h4>
            <p>Advanced AI-powered fake news detection system using deep learning and natural language processing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section (collapsible)
        with st.expander("⚙️ Features", expanded=True):
            st.markdown("""
            <div class="sidebar-info" style="background:transparent; border:none; box-shadow:none; margin:0; padding:0.5rem;">
                <div class="tooltip">
                    <p>• <strong>Real-time news analysis</strong></p>
                    <span class="tooltiptext">Instantly analyze articles using advanced LSTM neural networks for fake news detection</span>
                </div>
                <div class="tooltip">
                    <p>• <strong>AI model explanations</strong></p>
                    <span class="tooltiptext">LIME explanations show which words influenced the AI's decision, making predictions transparent</span>
                </div>
                <div class="tooltip">
                    <p>• <strong>Source verification</strong></p>
                    <span class="tooltiptext">Cross-reference articles with NewsAPI to find related sources for fact-checking</span>
                </div>
                <div class="tooltip">
                    <p>• <strong>Entity extraction</strong></p>
                    <span class="tooltiptext">Extract people, organizations, and locations from articles using spaCy NLP</span>
                </div>
                <div class="tooltip">
                    <p>• <strong>Automated summarization</strong></p>
                    <span class="tooltiptext">Generate concise summaries of articles using free Hugging Face transformer models</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Accuracy section
        st.markdown("""
        <div class="sidebar-info">
            <h4><i class="fas fa-chart-line"></i> Accuracy</h4>
            <p>Our LSTM model achieves around 93% accuracy on news classification tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown("""
    <div class="brand-header fade-in">
        <h1><i class="fas fa-shield-alt"></i> NewsGuard AI</h1>
        <p>Advanced Fake News Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, tokenizer, calibrator = load_models()
    nlp = load_nlp_model()
    summarizer = load_summarizer()
    
    if model is None or tokenizer is None or calibrator is None:
        st.error("Failed to load required models. Please check your model files.")
        return
    
    # Input Section
    st.markdown("""
    <div class="input-section fade-in">
        <h3><i class="fas fa-edit"></i> Article Input</h3>
        <p style="font-size: 0.95rem; color: #cbd5e1; margin-top: 1rem;">
            For the most accurate analysis, please paste the full, unedited text of the news article below. The system is optimized for English-language content and was trained on articles that are at least 10 words long.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        title = st.text_input("📰 Article Title", placeholder="Enter the article title...")
    
    with col2:
        text = st.text_area(
            "📄 Article Text", 
            height=200, 
            placeholder="Paste the full article text here..."
        )
    
    if st.button("🔍 Analyze Article", type="primary"):
        if not title or not text:
            st.error("Please provide both title and article text.")
            return
        try:
            # Combine title and text for validation
            full_text = f"{title} {text}"
            
            # Check for non-English content
            if detect(full_text) != 'en':
                st.error("Error: Only English language content is supported. Please provide an English article.")
                return

            # Check for corrupted or unreadable text
            # Calculates the percentage of non-alphanumeric characters (excluding spaces)
            alphanumeric_chars = re.sub(r'\s', '', full_text) # Remove spaces before counting
            if len(alphanumeric_chars) > 0:
                non_alphanumeric_ratio = len(re.findall(r'[^a-zA-Z0-9]', alphanumeric_chars)) / len(alphanumeric_chars)
                if non_alphanumeric_ratio > 0.5: # Threshold for "corrupted" text
                    st.error("Error: The input text appears to be corrupted or unreadable. Please check the content and try again.")
                    return
                    
        except LangDetectException:
            st.error("Error: Could not determine the language of the text. Please ensure it is valid English text.")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Preprocess
        status_text.markdown("""
        <div class="status-card">
            <span class="spinner"></span>
            Preprocessing article...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(10)
        
        padded_sequences, combined_text, clean_text_content, content_features = preprocess_article(title, text, tokenizer)
        
        # Check article length
        word_count = len(clean_text_content.split())
        if word_count < 10:
            st.markdown("""
            <div class="warning-card fade-in">
                <i class="fas fa-exclamation-triangle"></i>
                Warning: Article is very short (less than 10 words after cleaning). 
                This may affect prediction accuracy.
            </div>
            """, unsafe_allow_html=True)
        elif word_count > 1000:
            st.markdown("""
            <div class="warning-card fade-in">
                <i class="fas fa-exclamation-triangle"></i>
                Warning: Article is very long (over 1000 words). 
                Only the first 500 tokens will be used for prediction.
            </div>
            """, unsafe_allow_html=True)
        
        # Make prediction
        status_text.markdown("""
        <div class="status-card">
            <span class="spinner"></span>
            Making prediction...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(30)
        
        try:
            # FR-002: Confidence Scoring & Uncertainty Threshold
            is_real, confidence_percentage, _, _ = predict_article(model, calibrator, [padded_sequences, content_features])

            # Determine plain language confidence level
            if confidence_percentage >= 80:
                confidence_level = "High"
            elif confidence_percentage >= 70:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

            # Check for "Uncertain" classification
            if confidence_percentage < 70:
                prediction_class = "UNCERTAIN"
                box_class = "uncertain-news"
                icon = "fas fa-question-circle"
                recommendation_message = "<p style='font-size: 1rem; opacity: 0.9;'>Recommendation: The model's confidence is low. It is highly recommended to consult additional trusted sources to verify the information.</p>"
                
                st.markdown(f"""
                <div class="prediction-card {box_class} fade-in">
                    <i class="{icon}"></i>
                    Article Classification: {prediction_class}
                    <br>
                    <span style="font-size: 1rem; opacity: 0.9;">Confidence: {confidence_percentage:.1f}% ({confidence_level})</span>
                    {recommendation_message}
                </div>
                """, unsafe_allow_html=True)
                
                # Stop further analysis for uncertain results
                st.warning("Further AI analysis is paused due to low confidence.")
                status_text.empty()
                progress_bar.progress(100)
                return

            else:
                prediction_class = "REAL" if is_real else "FAKE"
                box_class = "real-news" if is_real else "fake-news"
                icon = "fas fa-check-circle" if is_real else "fas fa-times-circle"
                
                st.markdown(f"""
                <div class="prediction-card {box_class} fade-in">
                    <i class="{icon}"></i>
                    Article Classification: {prediction_class}
                    <br>
                    <span style="font-size: 1rem; opacity: 0.9;">Confidence: {confidence_percentage:.1f}% ({confidence_level})</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return
    
        # Create the hybrid tabbed layout
        tab_ai, tab_verify = st.tabs(["🤖 AI Explanation", "📰 Verification Sources"])

        # AI Explanation Tab
        with tab_ai:
            st.markdown("""
            <div style="font-size: 1rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 1.5rem;">
                Using LIME (Local Interpretable Model-agnostic Explanations), this section reveals the inner workings of the AI. The charts below highlight the specific words and phrases that most heavily influenced the final classification, providing transparency into the model's decision-making process.
                <ul style="margin-top: 0.5rem; padding-left: 20px;">
                    <li><span style="color: #10b981; font-weight: bold;">Green</span> indicates words contributing to a 'REAL' classification.</li>
                    <li><span style="color: #ef4444; font-weight: bold;">Red</span> indicates words contributing to a 'FAKE' classification.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            status_text.markdown("""
            <div class="status-card">
                <span class="spinner"></span>
                Generating AI explanation...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(50)
            
            explanation = create_lime_explanation(model, tokenizer, combined_text, content_features)
            
            if explanation:
                fig, html_explanation = create_lime_visualizations(explanation, prediction_class)
                
                if fig:
                    st.markdown('<h3 class="section-header fade-in"><i class="fas fa-chart-bar"></i> Feature Importance</h3>', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                if html_explanation:
                    st.markdown('<h3 class="section-header"><i class="fas fa-highlighter"></i> Highlighted Text</h3>', unsafe_allow_html=True)
                    st.components.v1.html(html_explanation, height=400, scrolling=True)
            else:
                st.error("Could not generate the AI model explanation.")

        # verification Sources Tab
        with tab_verify:
            status_text.markdown("""
            <div class="status-card">
                <span class="spinner"></span>
                Searching for verification sources...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(70)
            
            newsapi_key = st.secrets.get("NEWSAPI_KEY")
            
            if is_real and newsapi_key:
                st.markdown("""
                <div style="font-size: 1rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 1.5rem;">
                    To help you independently verify the story, this section uses Natural Language Processing (NLP) to extract key topics (people, organizations, events) from the article. It then searches for these topics across a wide range of reputable news outlets via the NewsAPI, presenting related articles for cross-referencing.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<h3 class="section-header fade-in"><i class="fas fa-check-double"></i> Related Articles</h3>', unsafe_allow_html=True)
                entities = extract_entities(clean_text_content, nlp)
                
                if entities:
                    search_query = " ".join(entities[:3])
                    articles = search_news_api(search_query, newsapi_key)
                    
                    if articles:
                        st.write(f"Found {len(articles)} related articles for verification:")
                        for i, article in enumerate(articles[:3]):
                            status_text.markdown(f"""
                            <div class="status-card">
                                <span class="spinner"></span>
                                Summarizing article {i+1} with free AI...
                            </div>
                            """, unsafe_allow_html=True)
                            progress_bar.progress(70 + (i+1) * 10)
                            
                            article_text = article.get('description', '') or article.get('content', '')
                            summary = summarize_with_free_model(article_text, summarizer)
                            if not summary:
                                summary = article.get('description', 'No summary available')
                            
                            published_date = article.get('publishedAt', '').split('T')[0] if article.get('publishedAt') else 'Unknown'
                            
                            st.markdown(f"""
                            <div class="summary-card fade-in">
                                <h4><i class="fas fa-newspaper"></i> {article.get('title', 'No title')}</h4>
                                <p><strong><i class="fas fa-building"></i> Source:</strong> {article.get('source', {}).get('name', 'Unknown')}</p>
                                <p><strong><i class="fas fa-calendar"></i> Published:</strong> {published_date}</p>
                                <p><strong><i class="fas fa-robot"></i> AI Summary:</strong> {summary}</p>
                                <p><a href="{article.get('url', '#')}" target="_blank"><i class="fas fa-external-link-alt"></i> Read full article</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-card fade-in">
                            <i class="fas fa-search"></i>
                            <strong>No Automatic Matches Found</strong>
                            <p style="margin-top: 0.5rem; font-weight: normal;">The automated search did not find any related articles. This can happen if a story is very new or the topic is niche.</p>
                            <p style="font-weight: normal;"><strong>Recommendation:</strong> For a more thorough check, we recommend manually searching the article's title or key topics on a trusted news aggregator or search engine.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Could not extract entities for verification search.")
            
            elif not is_real:
                st.markdown('<h3 class="section-header fade-in"><i class="fas fa-user-check"></i> Human Verification Recommended</h3>', unsafe_allow_html=True)
                st.markdown("""
                <div class="premium-card fade-in">
                    <p>The AI has classified this article as likely <strong>FAKE</strong>. Consequently, an automated search for related articles from genuine sources was not performed.</p>
                    <p><strong>Next Step:</strong> We strongly encourage you to investigate why the model reached this conclusion by reviewing the LIME analysis in the <strong>"🤖 AI Explanation"</strong> tab. This will show you the specific words that influenced the decision and can guide your own fact-checking.</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class="warning-card fade-in">
                    <i class="fas fa-key"></i>
                    Verification requires a NewsAPI key. Please configure the key in Streamlit secrets to enable this feature.
                </div>
                """, unsafe_allow_html=True)
        
        # Complete
        status_text.markdown("""
        <div class="status-card">
            <i class="fas fa-check-circle"></i>
            Analysis complete!
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(100)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">
            <i class="fas fa-shield-alt"></i> NewsGuard AI
        </div>
        <div class="footer-tech">
            Built with Streamlit • TensorFlow • LIME • Free AI Models
        </div>
        <div class="footer-copyright">
            © 2025 NewsGuard AI. All rights reserved. | Protecting Information Integrity
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()