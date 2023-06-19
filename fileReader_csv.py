from operator import index
from pandas._config.config import options
import Cleaner
import textract as tx
import pandas as pd
import os
import tf_idf

def read_resumes(file_csv):
    placeholder = []
    for i in range(len(file_csv["Context"])):
        temp = []
        temp.append(file_csv["Name"][i])
        context = file_csv["Context"][i]
        temp.append(context)
        placeholder.append(temp)
    return placeholder

def get_cleaned_words(document):
    for i in range(len(document)):
        raw = Cleaner.Cleaner(document[i][1])
        document[i].append(" ".join(raw[0]))
        document[i].append(" ".join(raw[1]))
        document[i].append(" ".join(raw[2]))
        sentence = tf_idf.do_tfidf(document[i][3].split(" "))
        document[i].append(sentence)
    return document

def file_Readert(file_csv):
    document = read_resumes(file_csv)
    doc = get_cleaned_words(document)
    Database = pd.DataFrame(document, columns=[
                        "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])

    # Database.to_csv("Resume_Data_test.csv", index=False)
    return Database
