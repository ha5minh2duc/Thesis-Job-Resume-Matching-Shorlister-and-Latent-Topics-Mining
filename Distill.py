import nltk
import spacy
import re
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import streamlit as st
import pandas as pd
from fileReader import Reader
import datetime
# Define english stopwords
stop_words = stopwords.words('english')

# load the spacy module and create a nlp object
# This need the spacy en module to be present on the system.
nlp = spacy.load('en_core_web_sm')
# proces to remove stopwords form a file, takes an optional_word list
# for the words that are not present in the stop words but the user wants them deleted.


def remove_stopwords(text, stopwords=stop_words, optional_params=False, optional_words=[]):
    if optional_params:
        stopwords.append([a for a in optional_words])
    return [word for word in text if word not in stopwords]


def tokenize(text):
    # Removes any useless punctuations from the text
    text = re.sub(r'[^\w\s]', '', text)
    return word_tokenize(text)


def lemmatize(text):
    # the input to this function is a list
    str_text = nlp(" ".join(text))
    lemmatized_text = []
    for word in str_text:
        lemmatized_text.append(word.lemma_)
    return lemmatized_text

# internal fuction, useless right now.


def _to_string(List):
    # the input parameter must be a list
    string = " "
    return string.join(List)


def remove_tags(text, postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Takes in Tags which are allowed by the user and then elimnates the rest of the words
    based on their Part of Speech (POS) Tags.
    """
    filtered = []
    str_text = nlp(" ".join(text))
    for token in str_text:
        if token.pos_ in postags:
            filtered.append(token.text)
    return filtered

def save_uploaded_file(uploadedfile, path_folder):
    with open(os.path.join(path_folder, uploadedfile.name),"bw") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file :{} in tempDir".format(uploadedfile.name))

def upload_file_Resumes_csv():
    upload_file = st.file_uploader("Choose a file Resumes type csv", key="1")
    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file)
            return df
        except:
            return []
        
def upload_file_resumes_csv():
    upload_file = st.file_uploader("Choose a file Resumes type csv", key = "2")
    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file, sep=',', header=None)
            return df
        except:
            return []

def upload_file_resumes_docx(config):
    data = []
    upload_files = st.file_uploader("Choose a file Resumes type docx", key = "5", accept_multiple_files = True)
    name_folder = datetime.datetime.now()
    name_folder = str(name_folder.year)+ "_" + str(name_folder.month)+ "_" + str(name_folder.day) +"_" + str(name_folder.hour)+ "_" + str(name_folder.minute)+ "_" + str(name_folder.second)
    path_folder = os.path.join(config.save_data.resume, name_folder)
    if upload_files:
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)
        for file in upload_files:
            save_uploaded_file(file, path_folder)
        data = Reader(path_folder)
    return data


def upload_file_jd_docx(config):
    data = []
    upload_files = st.file_uploader("Choose a file Job type docx", key = "7", accept_multiple_files = True)
    name_folder = datetime.datetime.now()
    name_folder = str(name_folder.year)+ "_" + str(name_folder.month)+ "_" + str(name_folder.day) +"_" + str(name_folder.hour)+ "_" + str(name_folder.minute)+ "_" + str(name_folder.second)
    path_folder = os.path.join(config.save_data.job, name_folder)
    if upload_files:
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)
        for file in upload_files:
            save_uploaded_file(file, path_folder)
        data = Reader(path_folder)
    return data
        
def upload_file_jobs_csv():
    upload_file = st.file_uploader("Choose a file Jobs type csv", key = "3")
    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file)
            return df
        except:
            return []
