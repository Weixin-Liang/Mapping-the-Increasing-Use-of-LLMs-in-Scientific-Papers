import pandas as pd
import numpy as np
from collections import Counter

def get_vocabulary_intersection(human_counts, ai_counts):
    """
    Gets the intersection of vocabularies from two dictionaries.

    Parameters:
    human_counts (dict): Dictionary of word counts for human data.
    ai_counts (dict): Dictionary of word counts for AI data.

    Returns:
    set: A set containing words that are present in both dictionaries.
    """
    return set(human_counts.keys()).intersection(ai_counts.keys())

def filter_frequent_words(word_counts, min_occurrences):
    """
    Filters words based on minimum occurrence threshold.

    Parameters:
    word_counts (dict): Dictionary of word counts.
    min_occurrences (int): Minimum number of occurrences for a word to be included.

    Returns:
    dict: A filtered dictionary with words meeting the minimum occurrence criterion.
    """
    return {word: count for word, count in word_counts.items() if count >= min_occurrences}

def count_human_binary_word_occurrences(human_data):
    """
    Counts the occurrences of unique words across sentences in the human data.

    Parameters:
    human_data (data frame): A data frame containing a list of sentences under the 'human_sentence' column.
    
    Returns:
    dict: A dictionary with words as keys and the number of sentences each word appears in as values.
    """
    word_counts = Counter(word for sent in human_data['human_sentence'] for word in set(sent))
    return dict(word_counts)

def count_ai_binary_word_occurrences(ai_data):
    """
    Counts the occurrences of unique words across sentences in the ai data.

    Parameters:
    ai_data (data frame): A data frame containing a list of sentences under the 'ai_sentence' column.
    
    Returns:
    dict: A dictionary with words as keys and the number of sentences each word appears in as values.
    """
    word_counts = Counter(word for sent in ai_data['ai_sentence'] for word in set(sent))
    return dict(word_counts)

def calculate_log_probability(human_probs, ai_probs, common_vocab):
    """
    Calculates the log_probability for words in the common vocabulary.

    Parameters:
    human_probs (dict): Dictionary of logarithmic probabilities for human data.
    ai_probs (dict): Dictionary of logarithmic probabilities for AI data.
    common_vocab (set): Set of common words in both vocabularies.

    Returns:
    pd.DataFrame: DataFrame containing probabilities distribution.
    """
    data = []
    for word in common_vocab:
        # default to a very low log probability
        log_human_prob = human_probs.get(word, -np.inf)
        log_ai_prob = ai_probs.get(word, -np.inf)
        
        # calculate log(1-p) and log(1-q) for each word
        log_one_minus_human_prob = np.log1p(-np.exp(log_human_prob))
        log_one_minus_ai_prob = np.log1p(-np.exp(log_ai_prob))
          
        # Calculating log odds
        human_log_odds = log_human_prob - log_one_minus_human_prob
        ai_log_odds = log_ai_prob - log_one_minus_ai_prob
        log_odds_ratio = human_log_odds - ai_log_odds
        
        # Skip if log odds ratio is infinite or NaN
        log_odds_ratio = human_log_odds - ai_log_odds
        if np.isinf(log_odds_ratio) or np.isnan(log_odds_ratio):
            continue

        data.append({"Word": word,  
                     "logP": log_human_prob,
                     'log1-P': log_one_minus_human_prob,
                     "logQ": log_ai_prob,
                     'log1-Q': log_one_minus_ai_prob,
                     "Log Odds Ratio": log_odds_ratio, 
                     })
    
    df = pd.DataFrame(data)
    # Sort words by log odds ratio
    df = df.sort_values(by='Log Odds Ratio', ascending=True)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns=['Log Odds Ratio'])
    return df

def estimate_log_probabilities(word_counts, total_sents):
    """
    Estimates the log probabilities of words based on their occurrence counts and the total number of sentences.
    
    Parameters:
    word_counts (dict): A dictionary with words as keys and the number of sentences each word appears in as values.
    total_sents (int): The total number of sentences considered in the data.
    
    Returns:
    dict: A dictionary with words as keys and their estimated log probabilities as values.
    """
    
    # Calculate log probabilities for each word. For each word in word_counts, divide its count by the total
    # number of sentences to get the probability of the word appearing in any sentence. Then, take the log of
    # this probability. The result is a dictionary where each word is mapped to its log probability.
    log_probabilities = {word: np.log(count / total_sents) for word, count in word_counts.items()}
    return log_probabilities

def estimate_text_distribution(human_source_path, ai_source_path,save_file_path="Word.parquet"):
    """
    Estimates text distribution of human and AI content by calculating log probabilities of word occurrences
    in both human and AI data and saves the results to a Parquet file.

    Parameters:
    human_source_path (str): Path to a Parquet file containing human-generated text data.
    ai_source_path (str): Path to a Parquet file containing AI-generated text data.
    save_file_path (str): The file path where the output Parquet file will be saved.

    """
    # Load the datasets from the provided Parquet files.
    human_data=pd.read_parquet(human_source_path)
    ai_data=pd.read_parquet(ai_source_path)
     # Verify that the expected columns are present in each dataset.
    if 'human_sentence' not in human_data.columns:
        raise ValueError("human_sentence column not found in human data")
    if 'ai_sentence' not in ai_data.columns:
        raise ValueError("ai_sentence column not found in ai data")

    # Filter out records where the sentences are too short (length <= 1) and drop any rows
    # where the sentence is missing (NaN values).
    human_data=human_data[human_data['human_sentence'].apply(len) > 1]
    ai_data=ai_data[ai_data['ai_sentence'].apply(len) > 1]
    human_data.dropna(subset=['human_sentence'], inplace=True)
    ai_data.dropna(subset=['ai_sentence'], inplace=True)
    
    # Count the occurrences of each unique word in both datasets.
    human_word_counts = count_human_binary_word_occurrences(human_data)
    ai_word_counts = count_ai_binary_word_occurrences(ai_data)
    
    # Calculate the total number of sentences in each dataset.
    total_human_sentences = len(human_data)
    total_ai_sentences = len(ai_data)
    
    # Estimate log probabilities of word occurrences in both datasets.
    human_log_probs = estimate_log_probabilities(human_word_counts, total_human_sentences)
    ai_log_probs = estimate_log_probabilities(ai_word_counts, total_ai_sentences)
    
    # Identify common vocabulary and frequent words in both datasets, along with words
    # common across both that meet a minimum frequency criterion.
    common_vocab = get_vocabulary_intersection(human_word_counts, ai_word_counts)
    frequent_human_words = filter_frequent_words(human_word_counts, 5)
    frequent_ai_words = filter_frequent_words(ai_word_counts, 3)
    frequent_common_vocab = common_vocab.intersection(frequent_human_words.keys(), frequent_ai_words.keys())

    # Calculate log(p), log(1-p), log(q) and log(1-q) for each word in the common vocabulary
    # and save the results to a Parquet file.
    # p denotes the probability of a word appearing in a human-generated sentence, while q denotes
    # the probability of a word appearing in an AI-generated sentence.
    log_likelihood_df = calculate_log_probability(human_log_probs, ai_log_probs, frequent_common_vocab)
    log_likelihood_df.to_parquet(save_file_path,index=False)