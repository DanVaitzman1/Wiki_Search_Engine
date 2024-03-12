#------------------Imports--------------------------------
import pandas as pd
import numpy as np
import json
import re
from collections import Counter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pickle
import numpy as np
from google.cloud import storage
from numpy.linalg import norm
from inverted_index_gcp import *
import math
from contextlib import closing
from inverted_index_gcp import MultiFileReader
from inverted_index_gcp import InvertedIndex
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
nltk.download('stopwords')
#--------------------------------------------------
#------------------Helper Functions--------------------------------

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(query):
    """Tokenize the input query.
    This function tokenizes the input query, converts it to lowercase, and stems each token using the Porter stemming algorithm.
    It also filters out stopwords from the token list.
    Args:
    - query (str): The input query to be tokenized.
    Returns:
    - list of str: A list of tokens after tokenization and stemming, with stopwords filtered out.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if
            token.group() not in all_stopwords]

def tokenize_without_stem(query):
    """Tokenize the input query without stemming.
    Args:
    - query (str): The input query.
    Returns:
    - list of str: A list of tokens after tokenization without stemming.
    """
    return [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
#----------------------------------------------------------------------------------------------------

#----------------------Reading Posting Lists ----------------------------
def process_token(token, Inverted_index_body, Inverted_index_anchor, bucket_name, postings_list_body,
                  postings_list_anchor):
    """Process a token and update the posting lists for body and anchor text.
    This function reads the posting lists corresponding to a given token from both the body and anchor indexes and updates the posting lists for body and anchor text.
    Args:
    - token (str): The token to process.
    - Inverted_index_body (InvertedIndex): The inverted index for body text.
    - Inverted_index_anchor (InvertedIndex): The inverted index for anchor text.
    - bucket_name (str): The name of the Google Cloud Storage bucket.
    - postings_list_body (dict): Dictionary to store posting lists for body text.
    - postings_list_anchor (dict): Dictionary to store posting lists for anchor text.
    """
    def read_postings_body():
        postings_list_body[token] = Inverted_index_body.read_a_posting_list("Postings_body_new", token, bucket_name)

    def read_postings_anchor():
        postings_list_anchor[token] = Inverted_index_anchor.read_a_posting_list("Postings_anchor", token, bucket_name)

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2 + 1000) as executor:
        # Submit tasks to the executor
        future_body = executor.submit(read_postings_body)
        future_anchor = executor.submit(read_postings_anchor)

        # Wait for all tasks to complete
        future_body.result()
        future_anchor.result()
def process_token_without_stem(token, bucket_name, Inverted_index_title_without_stem, postings_list_title_without_stem):
    """Process a token without stemming and update the posting list for title text.
    This function reads the posting list corresponding to a given token from the title index without stemming and updates the posting list for title text.
    Args:
    - token (str): The token to process.
    - bucket_name (str): The name of the Google Cloud Storage bucket.
    - Inverted_index_title_without_stem (InvertedIndex): The inverted index for title text without stemming.
    - postings_list_title_without_stem (dict): Dictionary to store posting lists for title text without stemming.
    """
    postings_list_title_without_stem[token] = Inverted_index_title_without_stem.read_a_posting_list(
        "Postings_title_without_stem", token, bucket_name)

def process_token_with_stem(token, bucket_name, Inverted_index_title, postings_list_title):
    """Process a token with stemming and update the posting list for title text.
    This function reads the posting list corresponding to a given token from the title index with stemming and updates the posting list for title text.
    Args:
    - token (str): The token to process.
    - bucket_name (str): The name of the Google Cloud Storage bucket.
    - Inverted_index_title (InvertedIndex): The inverted index for title text with stemming.
    - postings_list_title (dict): Dictionary to store posting lists for title text with stemming.
    """
    postings_list_title[token] = Inverted_index_title.read_a_posting_list("Postings_title", token, bucket_name)
#----------------------------------------------------------------------------------------------------

#----------------------------------Merge Scores Results--------------------------------------------------
def accumulate_scores(scores, weight, final_score):
    """Accumulate scores for documents.
    This function accumulates scores for documents based on the scores provided and the given weight. It updates the final scores dictionary accordingly.
    Args:
    - scores (list of tuples): List of tuples containing document IDs and their corresponding scores.
    - weight (float): Weight to apply to the scores.
    - final_score (Counter): Dictionary to store final scores for documents.
    """
    for doc_id, score in scores:
        if doc_id in final_score:
            final_score[doc_id] += score * weight
        else:
            final_score[doc_id] = score * weight
#----------------------------------------------------------------------------------------------------

#----------------------------------Scores Calculations--------------------------------------------------
def calculate_scores(query_tokens, query_tokens_without_stem, postings_list_body, postings_list_title,postings_list_anchor, Inverted_index_body, Inverted_index_title,
                     doc_id_len_dict, avg_docs_len, id_title_dict,with_stem,CosSim,norm_dict):
    """Calculate scores for documents based on query tokens and postings lists.
    This function calculates scores for documents based on the query tokens and postings lists provided. It supports different scoring methods such as BM25 and cosine similarity for body and title search. The final scores are normalized and returned.
    Args:
    - query_tokens (list): List of query tokens with stemming.
    - query_tokens_without_stem (list): List of query tokens without stemming.
    - postings_list_body (dict): Dictionary containing postings lists for body search.
    - postings_list_title (dict): Dictionary containing postings lists for title search.
    - postings_list_anchor (dict): Dictionary containing postings lists for anchor search.
    - Inverted_index_body (object): Inverted index object for body search.
    - Inverted_index_title (object): Inverted index object for title search.
    - doc_id_len_dict (dict): Dictionary containing document lengths.
    - avg_docs_len (float): Average length of documents.
    - id_title_dict (dict): Dictionary containing document IDs and titles.
    - with_stem (bool): Flag indicating whether stemming is applied.
    - CosSim (bool): Flag indicating whether to use cosine similarity.
    - norm_dict (dict): Dictionary containing normalization factors for documents.
    Returns:
    - future_results (list): List containing the results of future computations.
    """

    Body_Scores = Counter()
    Title_scores = Counter()
    sorted_doc_anchor_score_pairs = Counter()

    # Define functions for different tasks
    def calculate_body_scores():
        """Calculate scores for documents based on the body search.
        This function calculates scores for documents based on the body search using either cosine similarity or BM25.
        Returns:
        - Body_Scores (Counter): Counter containing document scores.
        """
        nonlocal Body_Scores
        if len(postings_list_body) > 0:
            if CosSim:
                Body_Scores = cosSim_search_body(query_tokens, postings_list_body, Inverted_index_body,norm_dict,doc_id_len_dict)[:500]
            else:
                Body_Scores = bm25_search(query_tokens, postings_list_body, Inverted_index_body, doc_id_len_dict,
                                          avg_docs_len)[:500]
            return Body_Scores


    def calculate_title_scores():
        """Calculate scores for documents based on the title search without stemming.
        This function calculates scores for documents based on the title search without stemming.
        Returns:
        - Title_scores (Counter): Counter containing document scores.
        """
        nonlocal Title_scores
        if len(postings_list_title) > 0:
            Title_scores = binary_search_title_without_stem(query_tokens_without_stem, postings_list_title,
                                                            id_title_dict)[:500]
            return Title_scores

    def calculate_title_with_stem_scores():
        """Calculate scores for documents based on the title search with stemming.
        This function calculates scores for documents based on the title search with stemming.
        Returns:
        - Title_scores (Counter): Counter containing document scores.
        """
        nonlocal Title_scores
        if len(postings_list_title) > 0:
            Title_scores = binary_search_title(query_tokens, postings_list_title,
                                                            id_title_dict)[:500]
            return Title_scores

    def calculate_anchor_scores():
        """Calculate scores for documents based on the anchor search.
        This function calculates scores for documents based on the anchor search.
        Returns:
        - sorted_doc_anchor_score_pairs (list): List of tuples containing document ID and anchor score pairs.
        """
        nonlocal sorted_doc_anchor_score_pairs
        if len(postings_list_anchor) > 0:
            sorted_doc_anchor_score_pairs = anchor_score(query_tokens, postings_list_anchor)[:500]
            if len(sorted_doc_anchor_score_pairs) > 0:
                max_value_score_anchor = sorted_doc_anchor_score_pairs[0][1]
                sorted_doc_anchor_score_pairs_norm = [(x[0], x[1] / max_value_score_anchor) for x in
                                                      sorted_doc_anchor_score_pairs]
                return sorted_doc_anchor_score_pairs_norm

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks to the executor
        future_body = executor.submit(calculate_body_scores)
        if with_stem :
            future_title = executor.submit(calculate_title_with_stem_scores)
        else:
            future_title = executor.submit(calculate_title_scores)
        future_anchor = executor.submit(calculate_anchor_scores)
        # Wait for all tasks to complete and retrieve their results
        future_results = [future_body.result(), future_title.result(), future_anchor.result()]
    return future_results

def process_doc_id(doc_id, Final_Score, pageRank, max_page_rank_val, WeightPageRank):
    """Process the document ID to update the final score based on PageRank.
    This function processes the document ID to update the final score based on PageRank.
    Parameters:
    - doc_id (str): The document ID.
    - Final_Score (Counter): Counter containing the final scores for documents.
    - pageRank (dict): Dictionary containing the PageRank scores for documents.
    - max_page_rank_val (float): The maximum PageRank value.
    - WeightPageRank (float): The weight assigned to the PageRank score.
    Returns:
    - None
    """
    try:
        Final_Score[doc_id] += (pageRank[doc_id] / max_page_rank_val) * WeightPageRank
    except:
        pass
def anchor_score(query_list, postings_list_anchor):
    """Calculate anchor scores for documents based on the anchor search.
    This function calculates anchor scores for documents based on the anchor search.
    Parameters:
    - query_list (list): List of tokens in the query.
    - postings_list_anchor (dict): Dictionary containing the posting lists for the anchor search.
    Returns:
    - list: List of tuples containing document ID and anchor score pairs.
    """
    if len(postings_list_anchor) > 0:
        tf_dict = Counter()
        for token in query_list:
            pos_list = postings_list_anchor[token]
            for entry in pos_list:
                if len(entry) == 2:
                    doc_id, tf = entry
                    tf_dict[doc_id] += tf
        return tf_dict.most_common()
    else:
        return []
#------------------------------------------------------------------------------------------------------------

#------------------------------------Search Function Using BM25----------------------------------------------
def bm25_search(query_list, postings_list, Inverted_index_body, doc_id_len_dict, avg_docs_len, B=0.1, K=2.1):
    """Perform BM25 search to rank documents based on the query.
    This function performs BM25 search to rank documents based on the given query.
    Parameters:
    - query_list (list): List of tokens in the query.
    - postings_list (dict): Dictionary containing the posting lists for the query terms.
    - Inverted_index_body (object): Inverted index for the document body.
    - doc_id_len_dict (dict): Dictionary containing document lengths.
    - avg_docs_len (float): Average length of documents in the corpus.
    - B (float): BM25 parameter controlling the impact of document length on term weighting. Default is 0.1.
    - K (float): BM25 parameter controlling the saturation of term frequency normalization. Default is 2.1.
    Returns:
    - list: List of tuples containing document ID and BM25 scores, sorted by score.
    """
    docs_number = len(doc_id_len_dict)
    avg_doc_len = float(avg_docs_len)
    # calc idf for specific token
    scores = Counter()
    if len(postings_list) > 0:
        for token in query_list:
            if Inverted_index_body.df[token] is not None:
                token_df = Inverted_index_body.df[token]
                token_idf = math.log(docs_number / token_df, 10)
                token_pos_list = postings_list[token]
                for doc_id, token_freq in token_pos_list:
                    numerator = token_freq * (K + 1)
                    denominator = token_freq + K * (1 - B + (B * doc_id_len_dict[doc_id]) / avg_doc_len)
                    scores[doc_id] += token_idf * (numerator / denominator)
        sorted_scores = scores.most_common()
        return sorted_scores
    else:
        return scores

#------------------------------------------------------------------------------------------------------------

#------------------------------------Binary Search Function With Stemming ----------------------------------------------

def binary_search_title(query_list, postings_list_title, id_title_dict):
    """Perform binary search on titles to rank documents based on the query.
    This function performs binary search on document titles to rank documents based on the given query.
    Parameters:
    - query_list (list): List of tokens in the query.
    - postings_list_title (dict): Dictionary containing the posting lists for the query terms in document titles.
    - id_title_dict (dict): Dictionary containing document titles indexed by document ID.
    Returns:
    - list: List of tuples containing document ID and binary search scores, sorted by score.
    """
    if len(postings_list_title) > 0:
        tf_dict = Counter()  # To store BM25 scores for each document
        for token in query_list:
            pos_list = postings_list_title[token]
            for doc_id, tf in pos_list:
                tf_dict[doc_id] += 1 / len(tokenize(id_title_dict[doc_id]))
        return tf_dict.most_common()
    else:
        return []
#------------------------------------------------------------------------------------------------------------
def cosSim_search_body(query_tokens, postings_list_body, Inverted_index_body,norm_dict,doc_id_len_dict):
    """Perform cosine similarity search in document bodies to rank documents based on the query.
    This function calculates cosine similarity scores between the query and document bodies to rank documents.
    Parameters:
    - query_tokens (list): List of tokens in the query.
    - postings_list_body (dict): Dictionary containing the posting lists for the query terms in document bodies.
    - Inverted_index_body (Index): Inverted index for document bodies.
    - norm_dict (dict): Dictionary containing the normalization factors for document bodies.
    - doc_id_len_dict (dict): Dictionary containing document lengths indexed by document ID.
    Returns:
    - list: List of tuples containing document ID and cosine similarity scores, sorted by score.
    """
    query_scores = {}
    doc_scores = Counter()
    candidates = get_candidate_documents_and_scores(query_tokens, Inverted_index_body, postings_list_body,doc_id_len_dict)
    query_len = len(query_tokens)
    tf_query_dict = Counter(query_tokens)
    N = len(doc_id_len_dict)  # Make sure this the name of dict

    for token in set(query_tokens):
        # First we are going to compute tf-idf scores for the query
        tf_query = tf_query_dict[token] / query_len
        df_query = Inverted_index_body.df.get(token, 0)
        idf_query = math.log(N / df_query, 10)
        query_scores[token] = tf_query * idf_query

    for token in query_tokens:
        term_posting_list = postings_list_body[token]
        for doc_id, tf in term_posting_list:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (
                        candidates.get((doc_id, token), 0) * query_scores.get(token, 0))

    for doc_id in doc_scores.keys():
        doc_scores[doc_id] = doc_scores[doc_id] * (1 / query_len) * (1 / norm_dict[doc_id])

    return doc_scores.most_common()

def get_candidate_documents_and_scores(query_to_search, index, pls,DL):
    """Get candidate documents and their scores based on the query terms.
    This function retrieves candidate documents and their scores based on the query terms.
    Parameters:
    - query_to_search (list): List of query terms to search for.
    - index (Index): Inverted index object.
    - pls (dict): Dictionary containing posting lists for query terms.
    - DL (dict): Dictionary containing document lengths indexed by document ID.
    Returns:
    - dict: Dictionary containing candidate documents and their scores.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in pls.keys():
            list_of_doc = pls[term]
            normlized_tfidf = []
            for doc_id, freq in list_of_doc:
                if doc_id in DL.keys():
                    if DL[doc_id] == 0:
                        continue
                    else:
                        normlized_tfidf.append(
                            (doc_id, (freq / DL[doc_id]) * np.math.log(len(DL) / index.df[term], 10)))
                else:
                    continue
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates
#------------------------------------Binary Search Function Witout Stemming ----------------------------------------------

def binary_search_title_without_stem(query_list, postings_list_title, id_title_dict):
    """Search for query terms in title without stemming and calculate BM25 scores.
    This function searches for query terms in the title without stemming and calculates BM25 scores for each document.
    Parameters:
    - query_list (list): List of query terms to search for.
    - postings_list_title (dict): Dictionary containing posting lists for query terms in the title.
    - id_title_dict (dict): Dictionary containing document titles indexed by document ID.
    Returns:
    - list: A list of tuples containing document IDs and their corresponding BM25 scores.
    """
    if len(postings_list_title) > 0:
        tf_dict = Counter()  # To store BM25 scores for each document
        for token in query_list:
            pos_list = postings_list_title[token]
            for doc_id, tf in pos_list:
                tf_dict[doc_id] += 1 / len(tokenize_without_stem(id_title_dict[doc_id]))
        return tf_dict.most_common()
    else:
        return []

# ------------------------------------------------------------------------------------------------------------
# -----------------------------------Weight Splits Function --------------------------------------------------------

def split_weights(is_question):
    """Split weights based on whether the input is a question or not.
    This function splits weights into four categories: WeightBody, WeightTitle, WeightAnchor, and WeightPageRank.
    The weights vary based on whether the input is a question or not.
    Parameters:
    - is_question (bool): A flag indicating whether the input is a question or not.
    Returns:
    - tuple: A tuple containing the weights for Body, Title, Anchor, and PageRank in that order.
    """
    if is_question:
        WeightBody ,WeightTitle,WeightAnchor,WeightPageRank= 0.30, 0.23,0.27,0.2
    else:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = 0.27, 0.24, 0.24, 0.25
    return WeightBody, WeightTitle, WeightAnchor, WeightPageRank
# ------------------------------------------------------------------------------------------------------------

#-----------Backend Search Function using : BM25-body , Binary search on Title witout stemming , anchor search , pageRank ---------------------
def backend_search_V1(query, Inverted_index_body, Inverted_index_title, Inverted_index_title_without_stem,
                   Inverted_index_anchor, pageRank, pageView, avg_docs_len, doc_id_len_dict, id_title_dict, norm_dict,
                   bucket_name):
    """Perform a backend search using BM25 for body, binary search on title without stemming,
    anchor search, and pageRank.
    This function takes a query and several indices as input, including inverted indices for body, title, and anchor,
    as well as pageRank data, document lengths, and other necessary information. It calculates scores for body, title,
    and anchor, combines them with weights, and adds pageRank scores. Finally, it returns the top 100 search results.
    Parameters:
    - query (str): The search query.
    - Inverted_index_body (InvertedIndex): Inverted index for the body.
    - Inverted_index_title (InvertedIndex): Inverted index for the title.
    - Inverted_index_title_without_stem (InvertedIndex): Inverted index for the title without stemming.
    - Inverted_index_anchor (InvertedIndex): Inverted index for the anchor.
    - pageRank (dict): PageRank scores for documents.
    - pageView (dict): Page view counts for documents (unused).
    - avg_docs_len (float): Average document length.
    - doc_id_len_dict (dict): Document ID to length mapping.
    - id_title_dict (dict): Document ID to title mapping.
    - norm_dict (dict): Normalization factor for documents.
    - bucket_name (str): Name of the storage bucket.
    Returns:
    - list: A list of tuples containing the top 100 search results. Each tuple contains the document ID and its title.

    """
    global Body_Scores_normalize
    Final_Score = Counter()
    query_after_split = query.split()
    if len(query_after_split) == 0:
        return Final_Score
    # Tokenize query with stemming
    query_tokens = tokenize(query)
    # Tokenize query without stemming
    query_tokens_without_stem = tokenize_without_stem(query)

    if "?" in query:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(True)
    else:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(False)

    Postings_list_body = {}
    Postings_list_anchor = {}
    Postings_list_title_without_stem = {}

    tokens_num = len(query_tokens) * 2 + len(query_tokens_without_stem)
    with ThreadPoolExecutor(max_workers=tokens_num + 1000) as executor:
        # Submit tasks to process each query token concurrently
        token_processing_tasks = []
        for token in query_tokens:
            token_processing_tasks.append(executor.submit(
                partial(process_token, token, Inverted_index_body, Inverted_index_anchor, bucket_name,
                        postings_list_body=Postings_list_body, postings_list_anchor=Postings_list_anchor)))
        for token in query_tokens_without_stem:
            token_processing_tasks.append(executor.submit(
                partial(process_token_without_stem, token, bucket_name, Inverted_index_title_without_stem,
                        postings_list_title_without_stem=Postings_list_title_without_stem)))
        for task in token_processing_tasks:
            task.result()
    calculate_scores_results = calculate_scores(query_tokens, query_tokens_without_stem, Postings_list_body, Postings_list_title_without_stem, Postings_list_anchor, Inverted_index_body,Inverted_index_title, doc_id_len_dict, avg_docs_len, id_title_dict,False,False,norm_dict)


    Body_Scores_normalize = calculate_scores_results[0]
    Title_scores = calculate_scores_results[1]
    sorted_doc_anchor_score_pairs_norm = calculate_scores_results[2]

    with ThreadPoolExecutor(max_workers=1000) as executor:
        accumulate_body_scores_task = executor.submit(accumulate_scores, Body_Scores_normalize, WeightBody, Final_Score)
        accumulate_title_scores_task = executor.submit(accumulate_scores, Title_scores, WeightTitle, Final_Score)
        accumulate_anchor_scores_task = executor.submit(accumulate_scores, sorted_doc_anchor_score_pairs_norm,
                                                        WeightAnchor, Final_Score)


        # Wait for the tasks to complete
        accumulate_body_scores_task.result()
        accumulate_title_scores_task.result()
        accumulate_anchor_scores_task.result()

    max_page_rank_val = next(iter(pageRank.values()))
    with ThreadPoolExecutor(max_workers=1500) as executor:
        # Submit tasks to process each doc_id concurrently
        doc_id_tasks = [
            executor.submit(process_doc_id, doc_id, Final_Score, pageRank, max_page_rank_val, WeightPageRank) for doc_id in Final_Score]
        # Wait for all tasks to complete
        for task in doc_id_tasks:
            task.result()

    most_common_Final_Score = Final_Score.most_common(100)
    Final_Results = []
    for tup in most_common_Final_Score:
        Final_Results.append((str(tup[0]), id_title_dict[tup[0]]))

    return Final_Results
# ------------------------------------------------------------------------------------------------------------

# -----------Backend Search Function using : BM25-body , Binary search on Title with stemming , anchor search , pageRank ---------------------
def backend_search_V2(query, Inverted_index_body, Inverted_index_title, Inverted_index_title_without_stem,
                      Inverted_index_anchor, pageRank, pageView, avg_docs_len, doc_id_len_dict, id_title_dict,
                      norm_dict,
                      bucket_name):
    """
    Backend search function version 2.
    This function performs a backend search using BM25 scoring for the body, binary search on titles with stemming,
    anchor search, and incorporates pageRank.
    Args:
        query (str): The search query.
        Inverted_index_body (InvertedIndex): Inverted index for the body.
        Inverted_index_title (InvertedIndex): Inverted index for the title.
        Inverted_index_title_without_stem (InvertedIndex): Inverted index for the title without stemming.
        Inverted_index_anchor (InvertedIndex): Inverted index for anchor text.
        pageRank (dict): Dictionary containing pageRank scores for documents.
        pageView (dict): Dictionary containing pageView scores for documents.
        avg_docs_len (float): Average document length.
        doc_id_len_dict (dict): Dictionary containing document lengths.
        id_title_dict (dict): Dictionary containing document titles.
        norm_dict (dict): Dictionary containing normalization factors for documents.
        bucket_name (str): Name of the bucket.
    Returns:
        list: List of tuples containing document IDs and titles.
    """
    global Body_Scores_normalize
    Final_Score = Counter()
    query_after_split = query.split()
    if len(query_after_split) == 0:
        return Final_Score
    # Tokenize query with stemming
    query_tokens = tokenize(query)
    # Tokenize query without stemming
    query_tokens_without_stem = tokenize_without_stem(query)

    if "?" in query:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(True)
    else:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(False)

    Postings_list_body = {}
    Postings_list_anchor = {}
    Postings_list_title = {}

    tokens_num = len(query_tokens) * 2 + len(query_tokens_without_stem)
    with ThreadPoolExecutor(max_workers=tokens_num + 1000) as executor:
        # Submit tasks to process each query token concurrently
        token_processing_tasks = []
        for token in query_tokens:
            token_processing_tasks.append(executor.submit(
                partial(process_token, token, Inverted_index_body, Inverted_index_anchor, bucket_name,
                        postings_list_body=Postings_list_body, postings_list_anchor=Postings_list_anchor)))
            token_processing_tasks.append(executor.submit(
                partial(process_token_with_stem, token, bucket_name, Inverted_index_title,
                        postings_list_title=Postings_list_title)))

        for task in token_processing_tasks:
            task.result()

    calculate_scores_results = calculate_scores(query_tokens, query_tokens_without_stem, Postings_list_body,
                                                Postings_list_title, Postings_list_anchor,
                                                Inverted_index_body,Inverted_index_title, doc_id_len_dict, avg_docs_len, id_title_dict,True,False,norm_dict)


    Body_Scores_normalize = calculate_scores_results[0]
    Title_scores = calculate_scores_results[1]
    sorted_doc_anchor_score_pairs_norm = calculate_scores_results[2]

    with ThreadPoolExecutor(max_workers=1000) as executor:
        accumulate_body_scores_task = executor.submit(accumulate_scores, Body_Scores_normalize, WeightBody, Final_Score)
        accumulate_title_scores_task = executor.submit(accumulate_scores, Title_scores, WeightTitle, Final_Score)
        accumulate_anchor_scores_task = executor.submit(accumulate_scores, sorted_doc_anchor_score_pairs_norm,
                                                        WeightAnchor, Final_Score)

        # Wait for the tasks to complete
        accumulate_body_scores_task.result()
        accumulate_title_scores_task.result()
        accumulate_anchor_scores_task.result()

    max_page_rank_val = next(iter(pageRank.values()))
    with ThreadPoolExecutor(max_workers=1500) as executor:
        # Submit tasks to process each doc_id concurrently
        doc_id_tasks = [
            executor.submit(process_doc_id, doc_id, Final_Score, pageRank, max_page_rank_val, WeightPageRank) for doc_id
            in Final_Score]
        # Wait for all tasks to complete
        for task in doc_id_tasks:
            task.result()

    most_common_Final_Score = Final_Score.most_common(100)
    Final_Results = []
    for tup in most_common_Final_Score:
        Final_Results.append((str(tup[0]), id_title_dict[tup[0]]))
    return Final_Results
# ------------------------------------------------------------------------------------------------------------

# -----------Backend Search Function using :  CosSIM body , Title with stem , pageRank ---------------------

def backend_search_V3(query, Inverted_index_body, Inverted_index_title, Inverted_index_title_without_stem,
                   Inverted_index_anchor, pageRank, pageView, avg_docs_len, doc_id_len_dict, id_title_dict, norm_dict,
                   bucket_name):
    """
    Backend search function version 3.
    This function performs a backend search using Cosine Similarity scoring for the body, binary search on titles
    with stemming, and incorporates pageRank.
    Args:
        query (str): The search query.
        Inverted_index_body (InvertedIndex): Inverted index for the body.
        Inverted_index_title (InvertedIndex): Inverted index for the title.
        Inverted_index_title_without_stem (InvertedIndex): Inverted index for the title without stemming.
        Inverted_index_anchor (InvertedIndex): Inverted index for anchor text.
        pageRank (dict): Dictionary containing pageRank scores for documents.
        pageView (dict): Dictionary containing pageView scores for documents.
        avg_docs_len (float): Average document length.
        doc_id_len_dict (dict): Dictionary containing document lengths.
        id_title_dict (dict): Dictionary containing document titles.
        norm_dict (dict): Dictionary containing normalization factors for documents.
        bucket_name (str): Name of the bucket.
    Returns:
        list: List of tuples containing document IDs and titles.
    """
    global Body_Scores_normalize
    Final_Score = Counter()
    query_after_split = query.split()
    if len(query_after_split) == 0:
        return Final_Score
    # Tokenize query with stemming
    query_tokens = tokenize(query)
    # Tokenize query without stemming
    query_tokens_without_stem = tokenize_without_stem(query)

    if "?" in query:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(True)
    else:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(False)

    Postings_list_body = {}
    Postings_list_anchor = {}
    Postings_list_title_without_stem = {}

    tokens_num = len(query_tokens) * 2 + len(query_tokens_without_stem)
    with ThreadPoolExecutor(max_workers=tokens_num + 1000) as executor:
        # Submit tasks to process each query token concurrently
        token_processing_tasks = []
        for token in query_tokens:
            token_processing_tasks.append(executor.submit(
                partial(process_token, token, Inverted_index_body, Inverted_index_anchor, bucket_name,
                        postings_list_body=Postings_list_body, postings_list_anchor=Postings_list_anchor)))
        for token in query_tokens_without_stem:
            token_processing_tasks.append(executor.submit(
                partial(process_token_without_stem, token, bucket_name, Inverted_index_title_without_stem,
                        postings_list_title_without_stem=Postings_list_title_without_stem)))
        for task in token_processing_tasks:
            task.result()
    calculate_scores_results = calculate_scores(query_tokens, query_tokens_without_stem, Postings_list_body, Postings_list_title_without_stem, Postings_list_anchor, Inverted_index_body,Inverted_index_title, doc_id_len_dict, avg_docs_len, id_title_dict,False,True,norm_dict)


    Body_Scores_normalize = calculate_scores_results[0]
    Title_scores = calculate_scores_results[1]
    sorted_doc_anchor_score_pairs_norm = calculate_scores_results[2]

    with ThreadPoolExecutor(max_workers=1000) as executor:
        accumulate_body_scores_task = executor.submit(accumulate_scores, Body_Scores_normalize, WeightBody, Final_Score)
        accumulate_title_scores_task = executor.submit(accumulate_scores, Title_scores, WeightTitle, Final_Score)
        accumulate_anchor_scores_task = executor.submit(accumulate_scores, sorted_doc_anchor_score_pairs_norm,
                                                        WeightAnchor, Final_Score)


        # Wait for the tasks to complete
        accumulate_body_scores_task.result()
        accumulate_title_scores_task.result()
        accumulate_anchor_scores_task.result()

    max_page_rank_val = next(iter(pageRank.values()))
    with ThreadPoolExecutor(max_workers=1500) as executor:
        # Submit tasks to process each doc_id concurrently
        doc_id_tasks = [
            executor.submit(process_doc_id, doc_id, Final_Score, pageRank, max_page_rank_val, WeightPageRank) for doc_id in Final_Score]
        # Wait for all tasks to complete
        for task in doc_id_tasks:
            task.result()

    most_common_Final_Score = Final_Score.most_common(100)
    Final_Results = []
    for tup in most_common_Final_Score:
        Final_Results.append((str(tup[0]), id_title_dict[tup[0]]))

    return Final_Results
# ------------------------------------------------------------------------------------------------------------

#-----------Backend Search Function using : BM25-body , Binary search on Title witout stemming , anchor search , pageRank ---------------------
def backend_search_V4(query, Inverted_index_body, Inverted_index_title, Inverted_index_title_without_stem,
                   Inverted_index_anchor, pageRank, pageView, avg_docs_len, doc_id_len_dict, id_title_dict, norm_dict,
                   bucket_name):
    """Perform a backend search using BM25 for body, binary search on title without stemming,
    anchor search, and pageRank.
    This function takes a query and several indices as input, including inverted indices for body, title, and anchor,
    as well as pageRank data, document lengths, and other necessary information. It calculates scores for body, title,
    and anchor, combines them with weights, and adds pageRank scores. Finally, it returns the top 100 search results.
    Parameters:
    - query (str): The search query.
    - Inverted_index_body (InvertedIndex): Inverted index for the body.
    - Inverted_index_title (InvertedIndex): Inverted index for the title.
    - Inverted_index_title_without_stem (InvertedIndex): Inverted index for the title without stemming.
    - Inverted_index_anchor (InvertedIndex): Inverted index for the anchor.
    - pageRank (dict): PageRank scores for documents.
    - pageView (dict): Page view counts for documents (unused).
    - avg_docs_len (float): Average document length.
    - doc_id_len_dict (dict): Document ID to length mapping.
    - id_title_dict (dict): Document ID to title mapping.
    - norm_dict (dict): Normalization factor for documents.
    - bucket_name (str): Name of the storage bucket.
    Returns:
    - list: A list of tuples containing the top 100 search results. Each tuple contains the document ID and its title.

    """
    global Body_Scores_normalize
    Final_Score = Counter()
    query_after_split = query.split()
    if len(query_after_split) == 0:
        return Final_Score
    # Tokenize query with stemming
    query_tokens = tokenize(query)
    # Tokenize query without stemming
    query_tokens_without_stem = tokenize_without_stem(query)

    if "?" in query:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(True)
    else:
        WeightBody, WeightTitle, WeightAnchor, WeightPageRank = split_weights(False)

    Postings_list_body = {}
    Postings_list_anchor = {}
    Postings_list_title_without_stem = {}

    tokens_num = len(query_tokens) * 2 + len(query_tokens_without_stem)
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to process each query token concurrently
        token_processing_tasks = []
        for token in query_tokens:
            token_processing_tasks.append(executor.submit(
                partial(process_token, token, Inverted_index_body, Inverted_index_anchor, bucket_name,
                        postings_list_body=Postings_list_body, postings_list_anchor=Postings_list_anchor)))
        for token in query_tokens_without_stem:
            token_processing_tasks.append(executor.submit(
                partial(process_token_without_stem, token, bucket_name, Inverted_index_title_without_stem,
                        postings_list_title_without_stem=Postings_list_title_without_stem)))
        for task in token_processing_tasks:
            task.result()
    calculate_scores_results = calculate_scores(query_tokens, query_tokens_without_stem, Postings_list_body, Postings_list_title_without_stem, Postings_list_anchor, Inverted_index_body,Inverted_index_title, doc_id_len_dict, avg_docs_len, id_title_dict,False,False,norm_dict)


    Body_Scores_normalize = calculate_scores_results[0]
    Title_scores = calculate_scores_results[1]
    sorted_doc_anchor_score_pairs_norm = calculate_scores_results[2]

    with ThreadPoolExecutor(max_workers=6) as executor:
        accumulate_body_scores_task = executor.submit(accumulate_scores, Body_Scores_normalize, WeightBody, Final_Score)
        accumulate_title_scores_task = executor.submit(accumulate_scores, Title_scores, WeightTitle, Final_Score)
        accumulate_anchor_scores_task = executor.submit(accumulate_scores, sorted_doc_anchor_score_pairs_norm,
                                                        WeightAnchor, Final_Score)


        # Wait for the tasks to complete
        accumulate_body_scores_task.result()
        accumulate_title_scores_task.result()
        accumulate_anchor_scores_task.result()

    max_page_rank_val = next(iter(pageRank.values()))
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to process each doc_id concurrently
        doc_id_tasks = [executor.submit(process_doc_id, doc_id, Final_Score, pageRank, max_page_rank_val, WeightPageRank) for doc_id in Final_Score]
        # Wait for all tasks to complete
        for task in doc_id_tasks:
            task.result()

    most_common_Final_Score = Final_Score.most_common(100)
    Final_Results = []
    for tup in most_common_Final_Score:
        Final_Results.append((str(tup[0]), id_title_dict[tup[0]]))

    return Final_Results
# ------------------------------------------------------------------------------------------------------------
