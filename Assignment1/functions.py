"""
This script...
"""

import string as st
import re
import nltk
from nltk.tokenize import word_tokenize, PorterStemmer
from nltk import WordNetLemmatizer


# def remove_punct(df):
#     """
#     This function removes punctuation from the 'plot'
#     column of the dataframe.
#     """
#     df['no_punct'] = df['plot'].str.replace('[{}]'.format(st.punctuation), '').str.lower()
#     # replace any double-space by single-space
#     df['no_punct'] = df['no_punct'].str.replace('  ', ' ')
#     # remove leading and trailing whitespaces
#     df['no_punct'] = df['no_punct'].str.strip()
#     return df


def remove_punct(text):
    """This function removes punctuation from the 'plot' column of the dataframe."""
    return (
        text.str.replace("[{}]".format(st.punctuation), "")
        .str.lower()
        .str.replace("  ", " ")
        .str.strip()
    )


def tokenize(text, opt="nlkt"):
    """This function tokenizes the 'no_punct' column of the dataframe."""
    if opt == "re":
        return text.apply(lambda x: re.split("\s+", x))
    elif opt == "nlkt":
        nltk.download("punkt")
        return text.apply(lambda x: word_tokenize(x))


def remove_small_words(text):
    """This function removes words with less than 3 characters."""
    return [x for x in text if len(x) > 3]


def remove_stopwords(text):
    """This function removes stopwords from the 'no_small_words' column of the dataframe."""
    return [word for word in text if word not in nltk.corpus.stopwords.words("english")]


def stemming(text):
    """This function stems the 'no_stopwords' column of the dataframe."""
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]


def lemmatize(text):
    """This function lemmatizes the 'no_stopwords' column of the dataframe."""
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


def get_pos_tag(tokenized_sentence):
    """This function returns the part of speech tag of the tokenized sentence."""
    return nltk.pos_tag(tokenized_sentence)


def return_sentences(tokens):
    """This function returns the tokenized sentence as a sentence."""
    return " ".join([word for word in tokens])


def process_text(df):
    """This function processes the text by removing punctuation, tokenizing,
    removing small words, removing stopwords, and stemming.
    """
    # remove punctuation
    df["no_punct"] = remove_punct(df["plot"])
    # tokenize
    df["tokenized"] = tokenize(df["no_punct"])
    # remove small words
    df["no_small_words"] = df["tokenized"].apply(lambda x: remove_small_words(x))
    # remove stopwords
    df["no_stopwords"] = df["no_small_words"].apply(lambda x: remove_stopwords(x))
    # stemming
    df["stemmed"] = df["no_stopwords"].apply(lambda x: stemming(x))
    # lemmatize
    df["lemmatized"] = df["no_stopwords"].apply(lambda x: lemmatize(x))
    # pos tagging
    df["pos_tagged"] = df["no_stopwords"].apply(lambda x: get_pos_tag(x))
    # return sentences
    df["processed_text"] = df["lemmatized"].apply(lambda x: return_sentences(x))

    return df
