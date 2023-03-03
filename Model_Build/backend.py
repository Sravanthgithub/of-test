import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

nltk.download('popular', quiet=True)
stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
df = pd.read_csv('C:/Users/Sravanth/Downloads/openfabric-test/of-test/Model_Build/of dataset/questions.csv')

def tokenizer(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords, lowercase=True)

matrix = vectorizer.fit_transform(
    tuple(df['question'])
)


def get_response(question):
    question = question.lower()
    similarity, index_similarity = solve(question)
    if similarity[0, index_similarity] < 0.3:
        return 'Sorry, I don\'t understand, Please try once again.'
    else:
        return df['answer'][index_similarity]
    
def solve(question):
    query_vector = vectorizer.transform([question])
    cosine_similarities = cosine_similarity(query_vector, matrix)
    index_similarity = np.argmax(cosine_similarities, axis = None)
    return cosine_similarities, index_similarity
