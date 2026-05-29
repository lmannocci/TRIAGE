import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.tokenize import word_tokenize, wordpunct_tokenize, TweetTokenizer, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from iso_language_codes import *
from langdetect import detect
import os
import sys


def detectLanguage(text):
    """
        Detect the language of text
        :param text: [str] text of which detecting language
        :return: [str] detected language
    """
    lang = detect(text)
    if lang == "zh-cn" or lang == "zh-tw":
        lang = "zh"
    try:
        language = language_name(lang).lower()
    except:
        language = "multilingual"
        logger.info("Error language detection")
    logger.info("language=" + str(language))
    return language


def transformDocs2Sentences(docs):
    """
         Convert a list of posts to a list of sentence. The docs are the sentence and not the posts
        :param docs: [list] list of documents
        :return: [list] list of sentences
    """
    concat_posts, posts = preprocessingPosts(docs)
    sent = sent_tokenize(concat_posts)
    return sent

def transformDoc2Sentences(doc):
    """
         Convert a post to a list of sentences.
        :param docs: [str] text of the post
        :return: [list] list of sentences
    """
    final_sent = []
    if doc != '':
        doc_lower = doc.lower().strip()
        sent = sent_tokenize(doc_lower)

        # nel dividere in frasi, commette degli errori, ovvero ci sono frasi composte solamente dalla punteggiatura di lunghezza 1, che vanno rimosse
        for s in sent:
            if len(s) == 1:
                continue
            final_sent.append(s)

    return final_sent

# scegli il miglior tokenize
def tokenizeWords(text, type_tokenizer):
    """
         Tokenize the text, given an input tokenizer
        :param text: [str] text to be tokenized
        :return: [str] the type of tokenizer you want to use
    """
    temp = []
    if type_tokenizer == "whitespace":
        for elem in text:
            tokens = elem.split()
            temp.append(tokens)
    elif type_tokenizer == "punct":
        for elem in text:
            tokens = nltk.wordpunct_tokenize(elem)
            temp.append(tokens)
    elif type_tokenizer == "regex":
        tokenizer = RegexpTokenizer(r'\w+')
        # tokenizer = nltk.tokenize.RegexpTokenizer ('\w+')
        for elem in text:
            tokens =  tokenizer.tokenize(elem)
            temp.append(tokens)
    elif type_tokenizer == "tweet":
        tokenizer = TweetTokenizer()
        for elem in text:
            tokens = tokenizer.tokenize(elem)
            temp.append(tokens)
    return temp

def tokenizeWords(text, type_tokenizer):
    """
         Tokenize the text, given an input tokenizer
        :param text: [str] text to be tokenized
        :param text [str] the type of tokenizer you want to use
        :return: [list] list of tokens
    """
    tokens = []
    if type_tokenizer == "whitespace":
        tokens = text.split()
    elif type_tokenizer == "punct":
        tokens = nltk.wordpunct_tokenize(text)
    elif type_tokenizer == "regex":
        tokenizer = RegexpTokenizer(r'\w+')
        # tokenizer = nltk.tokenize.RegexpTokenizer ('\w+')
        tokens =  tokenizer.tokenize(text)
    elif type_tokenizer == "tweet":
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(text)
    return tokens

# def tokenizeWords(text, type_tokenizer):
#     """
#          Tokenize the text, given an input tokenizer
#         :param text: [str] text to be tokenized
#         :param text [str] the type of tokenizer you want to use
#         :return: [list] list of tokens
#     """
#     temp = []
#     if type_tokenizer == "whitespace":
#         for elem in text:
#             tokens = elem.split()
#             temp.append(tokens)
#     elif type_tokenizer == "punct":
#         for elem in text:
#             tokens = nltk.wordpunct_tokenize(elem)
#             temp.append(tokens)
#     elif type_tokenizer == "regex":
#         tokenizer = RegexpTokenizer(r'\w+')
#         # tokenizer = nltk.tokenize.RegexpTokenizer ('\w+')
#         for elem in text:
#             tokens =  tokenizer.tokenize(elem)
#             temp.append(tokens)
#     elif type_tokenizer == "tweet":
#         tokenizer = TweetTokenizer()
#         for elem in text:
#             tokens = tokenizer.tokenize(elem)
#             temp.append(tokens)
#     return temp

def lower(token_posts):
    """
         Transform to lower case the tokens of the documents in input
        :param token_posts: [list of list] list of tokenized documents. Each document in token_posts is a list of tokens
        :return: [list of list] list of tokenized documents with lower tokens
    """
    temp = []
    for elem in token_posts:
        words_post = []
        for token in elem:
            words_post.append(token.lower())
        temp.append(words_post)
    return temp

#concatena post target
def preprocessingPosts(docs):
    """
         Convert docs to lower case, discard empty docs and remove white spaces and returns the list of docs and the
         joine docs in a unique text
        :param posts: [list] list of documents to be processed
        :return: [str, list] text concatenated, list of processed documents
    """
    docs[:] = (value.lower() for value in docs if value != "")

    if len(docs) != 0:
        text = '. '.join(docs)
    else:
        text = ''
    return text.strip(), docs


def remove_URL(text):
    """
         Remove URLs from a document
        :param text: [str] text from which remove URLs
        :return: [str] text without URls
    """
    s = re.sub(r"http\S+", "", text)
    return s

def remove_hashtags(text):
    """
         Remove hashtags from a document
        :param text: [str] text from which remove hashtags
        :return: [str] text without hashtags
    """
    s = re.sub(r"#(\w+)", "", text)
    return s

def extract_URL(text):
    """
         Extract URLs from a document
        :param text: [str] text from which remove URLs
        :return: [str] text without URls
    """
    return re.findall(r"http\S+", text)

def extract_hashtags(text):
    """
         Extract hashtags from a document
        :param text: [str] text from which remove hashtags
        :return: [str] text without hashtags
    """
    return re.findall(r"#(\w+)", s)



# def remove_URL(docs):
#     """
#          Remove URLs from a list of documents
#         :param docs: [list] list of documents from which remove URLs
#         :return: [list] list of documents without URls
#     """
#     temp = []
#     for elem in docs:
#         s = re.sub(r"http\S+", "", elem)
#         if s!=[]:
#             temp.append(s)
#     return temp
#
# def remove_hashtags(docs):
#     """
#          Remove hashtags from a list of documents
#         :param docs: [list] list of documents from which remove hashtags
#         :return: [list] list of documents without hashtags
#     """
#     temp = []
#     for elem in docs:
#         s = re.sub(r"#(\w+)", "", elem)
#         if s!= []:
#             temp.append(s)
#     return temp

def remove_punctuation(tokens_posts):
    """
         Remove punctuation from list of tokenized posts
        :param tokens_posts: [list of list] list of list of tokens
        :return: [list of list] list of list of tokens without punctuation
    """
    temp = []
    for elem in tokens_posts:
        new_words = []
        for token in elem:
            new_word = re.sub(r'[^\w\s]', '', token)
            if new_word != '':
                new_words.append(new_word)
        if new_words!= []:
            temp.append(new_words)
    return temp



# def remove_stop_words(token_posts, language):
#     """
#          Remove stop words from list of tokenized posts, given the language
#         :param tokens_posts: [list of list] list of list of tokens from which remove stopwords
#         :param language: [str] language of the tokenized posts given in input
#         :return: [list of list] list of list of tokens without punctuation
#     """
#     temp = []
#     sw = stopwords.words(language)
#     for elem in token_posts:
#         token_ns = []
#         for word in elem:
#             if word not in sw:
#                 token_ns.append(word)
#         if token_ns!=[]:
#             temp.append(token_ns)
#     return temp

def remove_stop_words(token_posts, language):
    """
         Remove stop words from list of tokenized posts, given the language
        :param tokens_posts: [list of list] list of list of tokens from which remove stopwords
        :param language: [str] language of the tokenized posts given in input
        :return: [list of list] list of list of tokens without punctuation
    """
    sw = stopwords.words(language)
    token_ns = []
    for word in token_posts:
        if word not in sw or word == '':
            token_ns.append(word)
    return token_ns