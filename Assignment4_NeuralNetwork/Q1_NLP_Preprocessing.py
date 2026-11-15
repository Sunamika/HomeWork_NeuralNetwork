
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

def preprocess_text(sentence):
    # Step 1: Tokenize the sentence into words and punctuation
    tokens = word_tokenize(sentence)
    print("Original Tokens:")
    print(tokens)

    # Step 2: Remove English stopwords
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words
        and word.isalpha()
    ]

    print("\nTokens Without Stopwords:")
    print(filtered_tokens)

    # Step 3: Apply Porter stemming
    stemmer = PorterStemmer()

    stemmed_words = [
        stemmer.stem(word) for word in filtered_tokens
    ]

    print("\nStemmed Words:")
    print(stemmed_words)

    return tokens, filtered_tokens, stemmed_words


# Given Input sentence
sentence = (
    "NLP techniques are used in virtual assistants "
    "like Alexa and Siri."
)

preprocess_text(sentence)
