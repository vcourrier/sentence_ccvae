import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def process_IMDB(path, vocab_size = 10000, max_pad=100):
    df_tok = pd.read_csv(path)

    df_tok['review'] = df_tok['review'].apply(lambda x: word_tokenize(x.lower())) #tokenize

    all_words = [word for review in df_tok['review'] for word in review]
    word_counts = Counter(all_words)
    most_common_words = set([word for word, count in word_counts.most_common(vocab_size)])

    df_tok['review'] = df_tok['review'].apply(lambda x: [word if word in most_common_words else '<UNK>' for word in x])

    # Encode reviews
    word2value = {word: idx for idx, word in enumerate(most_common_words, start=1)}
    word2value['<UNK>'] = 0

    df_enc = df_tok.copy()
    df_enc['review'] = df_enc['review'].apply(lambda x: [word2value[word] for word in x])

    # Convert to tensors and pad
    review_tensors = [torch.tensor(encoded_review) for encoded_review in df_enc['review']]
    padded = pad_sequence(review_tensors, batch_first=True, padding_value=0).narrow(1, 0, max_pad)

    # Initialize lists to store values for the DataFrame
    reviews = []
    sentiment_values = []

    # Iterate over each row of the tensor
    for i in range(padded.size(0)):
        # Extract the row from the tensor
        row = padded[i]

        # Convert the tensor row to a list and append it to the 'reviews' list
        reviews.append(row.tolist())
        # Extract the corresponding value of 'sentiment' column from the other DataFrame
        sentiment_value = df_tok.iloc[i]['sentiment']
        # Append the value to the 'sentiment_values' list
        if sentiment_value == "positive" :
            sentiment_values.append(1)
        else :
            sentiment_values.append(0)

    # Final dataframe
    final_df = pd.DataFrame({'review': reviews, 'sentiment': sentiment_values})

    return final_df, word2value


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract features and labels for a single row
        features = torch.tensor(self.dataframe.iloc[idx]['review'], dtype=torch.float32) 
        label = torch.tensor(self.dataframe.iloc[idx]['sentiment'], dtype=torch.long)
        index = torch.tensor(self.dataframe.iloc[idx].name, dtype=torch.long)  
        return index, features, label