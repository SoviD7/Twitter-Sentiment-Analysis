import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import re
import eda as e


# Load the pre-trained SVM model
model = joblib.load('SVM.joblib')  # Replace with your model's filename

# Load your Word2Vec model (replace 'your_word2vec_model_filename.model' with your actual model filename)
word2vec_model = Word2Vec.load('w2v.model')

# Text preprocessing functions (clean_text and remove_stopwords)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)   # Remove Special Characters and Punctuation
    text = text.lower()                          # Lowercasing
    text = re.sub(r'@\w+|#\w+', '', text)        # Remove Mentions and Hashtags
    text = re.sub(r'\d+', '', text)              # Remove Numbers
    text = ' '.join(text.split())                # Remove Extra Whitespace
    text = re.sub(r'RT[\s]+','',text)            # Remove RT
    text = re.sub(r'https?:\/\/\S+','',text)     # Remove hyperlinks
    return text
def lower(tweet):
    a = tweet.lower()
    return a
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

image_url = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.dreamstime.com%2Fphotos-images%2Ftwitter-background.html&psig=AOvVaw3r-37KTy-BLiy-5ThLITjr&ust=1697617865261000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCOiwoMLV_IEDFQAAAAAdAAAAABAE"  # Replace with the actual URL of the image

st.markdown(
    f"""
    <style>
    body {{
        background-image: url('{image_url}');
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Streamlit app
st.title("Sentiment Analysis")
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('twitter-users-post-banner.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


user_input = st.text_input("Enter a tweet:")

# Create a button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        user_input = str(user_input)  # Ensure user_input is a string
        
        # Tokenize and preprocess the user input
        tokenized_input = word_tokenize(user_input.lower())
        cleaned_input = clean_text(user_input)  # Only need to preprocess once
        filtered_input = remove_stopwords(tokenized_input)
        
        # Initialize a list to store the Word2Vec vectors
        vectors = []
        
        # Iterate through the tokenized sentences
        for sentence in filtered_input:
            # Check if the sentence is not empty
            if sentence in word2vec_model.wv:
                vectors.append(word2vec_model.wv[sentence])
        
        if vectors:
            
             prediction = model.predict(vectors)[0]
            
            # Map class labels to human-readable categories
             class_mapping = {
                0: "Figurative",
                1: "Irony",
                2: "Regular",
                3: "Sarcasm"
             }
             sentiment = class_mapping.get(prediction, "Unknown")
             st.write(f"Category: {sentiment}")
             
                        
  
            