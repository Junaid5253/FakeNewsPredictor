import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# 1. Load Data
def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    return fake_df, true_df


# 2. Merge DataFrames
def merge_data(df1, df2):
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    return df


# 3. Drop unnecessary columns
def drop_columns(df):
    if 'date' in df.columns:
        df = df.drop('date', axis=1)
    return df


# 4. Combine text + title
def create_content(df):
    df['content'] = df['title'] + " " + df['text']
    return df


# 5. Text Cleaning
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)


# 6. Apply preprocessing
def preprocess(df):
    df['content'] = df['content'].apply(clean_text)
    return df


from pathlib import Path

def preprocessing_script():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    fake_df, true_df = load_data(
        DATA_DIR / "Fake.csv",
        DATA_DIR / "True.csv"
    )

    df = merge_data(fake_df, true_df)
    df = drop_columns(df)
    df = create_content(df)
    df = preprocess(df)

    return df



