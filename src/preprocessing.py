import pandas as pd
from pathlib import Path
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download once (safe check)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Global stopwords (efficient)
stop_words = set(stopwords.words('english'))

# 1. Load Data
def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    return fake_df, true_df


# 2. Merge DataFrames
def merge_data(df1, df2):
    return pd.concat([df1, df2], axis=0).reset_index(drop=True)


# 3. Drop unnecessary columns
def drop_columns(df):
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df


# 4. Combine text + title
def create_content(df):
    df['content'] = df['title'] + " " + df['text']
    return df


# 5. Text Cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# 6. Apply preprocessing
def preprocess(df):
    df['content'] = df['content'].apply(clean_text)
    return df


# 7. Main function to save cleaned data
def save_clean_data():
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

    # Ensure directory exists
    save_path = DATA_DIR / "clean_data.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, index=False)

    print(f"Clean data saved at: {save_path}")


if __name__ == "__main__":
    save_clean_data()