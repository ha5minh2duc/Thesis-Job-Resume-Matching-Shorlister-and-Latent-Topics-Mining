import matplotlib.colors as mcolors
import gensim
import gensim.corpora as corpora
from wordcloud import WordCloud
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import Similar
from PIL import Image
import time
from Job_resume_matching.extraction import extraction
from Job_resume_matching.extraction_resume import extraction_resume
from Job_resume_matching.matching import matching
from Distill import upload_file_jobs_csv, upload_file_resumes_csv, upload_file_Resumes_csv, upload_file_resumes_docx, upload_file_jd_docx
import asyncio
from Database.connect_database import QueryDatabase
import copy
from fileReader_csv import file_Readert
import yacs.config
from config import get_default_config


def load_config() -> yacs.config.CfgNode:
    config = get_default_config()
    return config

config = load_config()
def load_title_dasboard():
    image = Image.open('Images//logo.png')
    st.image(image, use_column_width=True)
    st.title("Resume Matcher")
    option_yn = st.selectbox(
            "Choose type data input?", options=['CSV', 'DOC'])
    return option_yn

def load_input_data_Resumes():
    Resumes = upload_file_Resumes_csv()
    return Resumes

def load_input_data_Jobs():
    Jobs = upload_file_jobs_csv()
    return Jobs

def load_input_data_Resumes_docx():
    Resumes = upload_file_resumes_docx(config)
    return Resumes

def load_input_data_Jobs_docx():
    Resumes = upload_file_jd_docx(config)
    return Resumes

async def load_input_data_resumes():
    resumes = upload_file_resumes_csv()
    return resumes

async def show_description_name(_Jobs):
    # """ Show all description with data input"""
    if len(_Jobs['Name']) <= 1:
        st.write(
            "There is only 1 Job Description present. It will be used to create scores.")
    else:
        st.write("There are ", len(_Jobs['Name']),
                "Job Descriptions available. Please select one.")
    # Asking to Print the Job Desciption Names
    if len(_Jobs['Name']) > 1:
        option_yn = st.selectbox(
            "Show the Job Description Names?", options=['YES', 'NO'])
        if option_yn == 'YES':
            index = [a for a in range(len(_Jobs['Name']))]
            fig = go.Figure(data=[go.Table(header=dict(values=["Job No.", "Job Desc. Name"], line_color='darkslategray',
                                                    fill_color='lightskyblue'),
                                        cells=dict(values=[index, _Jobs['Name']], line_color='darkslategray',
                                                    fill_color='#f4f4f4'))
                                ])
            fig.update_layout(width=700, height=400)
            st.write(fig)

async def show_description(_Jobs):
    # """Choose index description before use model matching rule"""
    # Asking to chose the Job Description
    index = st.slider("Which JD to select ? : ", 0,
                    len(_Jobs['Name'])-1, 1)
    option_yn = st.selectbox("Show the Job Description ?", options=['YES', 'NO'])
    if option_yn == 'YES':
        st.markdown("---")
        st.markdown("### Job Description :")
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Job Description"],
                        fill_color='#f0a500',
                        align='center', font=dict(color='white', size=16)),
            cells=dict(values=[_Jobs['Context'][index]],
                    fill_color='#f4f4f4',
                    align='left'))])

        fig.update_layout(width=800, height=500)
        st.write(fig)
        st.markdown("---")
    return index

async def show_information_retrieval(_Jobs, _index):
    # """Extraction informaion JD and show Information JD by index in table"""
    info_retrieval = extraction(_Jobs, _index)
    option_yn = st.selectbox("Information Retrieval ?", options=['YES', 'NO'])
    if option_yn == 'YES':
        st.markdown("---")
        st.markdown("### Information Retrieval :")
        fig = go.Figure(data=[go.Table(columnwidth = [1, 1, 2 , 3], header=dict(values=["Index", "Minimum degree level", "Acceptable majors", "Skills"], line_color='darkslategray',
                                                    fill_color='#f0a500'),
                                        cells=dict(values=[_index, info_retrieval["Minimum degree level"], info_retrieval["Acceptable majors"], info_retrieval["Skills"]], line_color='darkslategray',
                                                    fill_color='#f4f4f4'))
                                ])

        fig.update_layout(width=800, height=500)
        st.write(fig)
        st.markdown("---")
    return info_retrieval

async def show_information_retrieval_resumes(_Resumes):
    # """Extraction informaion Resume and show Information Resume in table"""
    info_retrieval_resume = extraction_resume(_Resumes)
    index = [a for a in range(len(info_retrieval_resume['skills']))]
    option_yn = st.selectbox("Information Retrieval Resumes ?", options=['YES', 'NO'])
    if option_yn == 'YES':
        st.markdown("---")
        st.markdown("### Information Retrieval Resumes:")
        fig = go.Figure(data=[go.Table(columnwidth = [1, 1, 2 , 3], header=dict(values=["Index", "Degrees", "Majors", "Skills"], line_color='darkslategray',
                                                    fill_color='#f0a500'),
                                        cells=dict(values=[index, info_retrieval_resume["degrees"], info_retrieval_resume["majors"], info_retrieval_resume["skills"]], line_color='darkslategray',
                                                    fill_color='#f4f4f4'))
                                ])

        fig.update_layout(width=800, height=500)
        st.write(fig)
        st.markdown("---")
    for i in range(len(info_retrieval_resume)):
        info_retrieval_resume["degrees"][i] = str([info_retrieval_resume["degrees"][i]])
        info_retrieval_resume["majors"][i] = str(info_retrieval_resume["majors"][i]) 
        info_retrieval_resume["skills"][i] = str(info_retrieval_resume["skills"][i])
    return info_retrieval_resume

async def find_JD_by_keyword():
    # """Fillter JB by keyword input"""
    option_yn = st.selectbox("Find Resumes by keyword ?", options=['YES', 'NO'])
    if option_yn == 'YES':
        keyword = st.text_area("Enter keywords")
        if keyword:
            database = QueryDatabase(keyword, config)
            result = database.get_resumes_by_keyword(keyword)
            _indexs = [a for a in range(len(result["Name"]))]
            st.markdown("---")
            st.markdown("### Resumes by keyword:")
            fig = go.Figure(data=[go.Table(columnwidth = [1, 2, 2 ], header=dict(values=["Index", "Name", "Context"], line_color='darkslategray',
                                                        fill_color='#f0a500'),
                                            cells=dict(values=[_indexs, result["Name"], result["Context"]], line_color='darkslategray',
                                                        fill_color='#f4f4f4'))
                                    ])
            fig.update_layout(width=800, height=500)
            st.write(fig)
            st.markdown("---")


async def show_matching_rule(_indexs, _info_retrieval, _resumes, _Jobs, _Resumes):
    # """ Processing mathching rule by 5 model:All-mpnet-base-v2, Paraphrase-MiniLM-L6-v2, All-MiniLM-L12-v1, GPT3, TF-IDF, Final_score
    #     with: _indexs: choose index JD
    #           _info_retieval: JD after extraction by index
    #           _resumes: Resumes after extraciton
    #           _Jobs: Jobs in type fileReder
    #           _Resume: Resume in type fileReder

    #     output: Dataframe
    #     Show output in table on streamlit
    # """
    results_matching = await asyncio.gather(matching(_info_retrieval, _resumes, config))
    results_matching = results_matching[0]
    score_tfidf = calculate_scores(_Resumes,  _Jobs, _indexs)
    final_score = []
    for i in range(len(results_matching[0]["matching score job 0"])):
        score = results_matching[0]["matching score job 0"][i] + results_matching[1]["matching score job 0"][i] + results_matching[2]["matching score job 0"][i] \
                + results_matching[3]["matching score job 0"][i] + score_tfidf[i] 
        final_score.append(score)
    data = {
        "All-mpnet-base-v2": results_matching[0]["matching score job 0"],
        "Paraphrase-MiniLM-L6-v2": results_matching[1]["matching score job 0"],
        "All-MiniLM-L12-v1" : results_matching[2]["matching score job 0"],
        "GPT3": results_matching[3]["matching score job 0"],
        "TF-IDF": score_tfidf,
        "Final_score": final_score
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by=["Final_score"], ascending=False)
    option_yn = st.selectbox("Matching Ruler by model Sentence Transformer", options=['YES', 'NO'])
    if option_yn == 'YES':
        _indexs = [a for a in range(len(results_matching[0]["degrees"]))]
        st.markdown("---")
        st.markdown("### Matching Ruler by model Sentence Transformer:")
        fig = go.Figure(data=[go.Table(columnwidth = [1, 2, 2 , 2, 2, 2, 2], header=dict(values=["Index", "All-mpnet-base-v2", "Paraphrase-MiniLM-L6-v2", "All-MiniLM-L12-v1","GPT3", "TF-IDF", "Final Score"], line_color='darkslategray',
                                                    fill_color='#f0a500'),
                                        cells=dict(values=[_indexs, df["All-mpnet-base-v2"], df["Paraphrase-MiniLM-L6-v2"], df["All-MiniLM-L12-v1"], df["GPT3"], df["TF-IDF"], df["Final_score"]], line_color='darkslategray',
                                                    fill_color='#f4f4f4'))
                                ])
        fig.update_layout(width=800, height=500)
        st.write(fig)
        st.markdown("---")



@st.cache_data()
def calculate_scores(_resumes, _job_description, index):
    scores = []
    for x in range(_resumes.shape[0]):
        score = Similar.match(
            _resumes['TF_Based'][x], _job_description['TF_Based'][index])
        scores.append(score)
    return scores

def ranked_resumes(_Resumes, _Jobs, _index):
    _Resumes['Scores'] = calculate_scores(_Resumes, _Jobs, _index)
    Ranked_resumes = _Resumes.sort_values(
        by=['Scores'], ascending=False).reset_index(drop=True)

    Ranked_resumes['Rank'] = pd.DataFrame(
        [i for i in range(1, len(Ranked_resumes['Scores'])+1)])
    return Ranked_resumes

def score_table_plot(_Ranked_resumes):
    fig1 = go.Figure(data=[go.Table(
    header=dict(values=["Rank", "Name", "Scores"],
                fill_color='#00416d',
                align='center', font=dict(color='white', size=16)),
    cells=dict(values=[_Ranked_resumes.Rank, _Ranked_resumes.Name, _Ranked_resumes.Scores],
            fill_color='#d6e0f0',
            align='left'))])
    fig1.update_layout(title="Top Ranked Resumes", width=700, height=1100)
    st.write(fig1)
    st.markdown("---")
    fig2 = px.bar(_Ranked_resumes,
                x=_Ranked_resumes['Name'], y=_Ranked_resumes['Scores'], color='Scores',
                color_continuous_scale='haline', title="Score and Rank Distribution")
    # fig.update_layout(width=700, height=700)
    st.write(fig2)
    st.markdown("---")


@st.cache_data()
def get_list_of_words(document):
    Document = []

    for a in document:
        raw = a.split(" ")
        Document.append(raw)

    return Document

def tfidf(_Resumes):
    document = get_list_of_words(_Resumes['Cleaned'])

    id2word = corpora.Dictionary(document)
    corpus = [id2word.doc2bow(text) for text in document]


    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                                update_every=3, chunksize=100, passes=50, alpha='auto', per_word_topics=True)
    return lda_model, corpus

# @st.cache_data  # Trying to improve performance by reducing the rerun computations
def format_topics_sentences(ldamodel, corpus):
    sent_topics_df = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.append(
                    [i, int(topic_num), round(prop_topic, 4)*100, topic_keywords])
            else:
                break

    return sent_topics_df

def topic_word_clound(lda_model):
    st.markdown("## Topics and Topic Related Keywords ")
    st.markdown(
        """This Wordcloud representation shows the Topic Number and the Top Keywords that contstitute a Topic.
        This further is used to cluster the resumes.      """)

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    collocations=False,
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown("---")

def show_sunburst_graph(_lda_model, _corpus, _Resumes):
    df_topic_sents_keywords = format_topics_sentences(
    ldamodel=_lda_model, corpus=_corpus)
    df_some = pd.DataFrame(df_topic_sents_keywords, columns=[
                        'Document No', 'Dominant Topic', 'Topic % Contribution', 'Keywords'])
    df_some['Names'] = _Resumes['Name']

    df = df_some

    st.markdown("## Topic Modelling of Resumes ")
    st.markdown(
        "Using LDA to divide the topics into a number of usefull topics and creating a Cluster of matching topic resumes.  ")
    fig3 = px.sunburst(df, path=['Dominant Topic', 'Names'], values='Topic % Contribution',
                    color='Dominant Topic', color_continuous_scale='viridis', width=800, height=800, title="Topic Distribution Graph")
    st.write(fig3)

def format_topics_sentences_module(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = pd.concat([sent_topics_df, pd.DataFrame([pd.Series([int(topic_num), round(prop_topic,4), topic_keywords])])], ignore_index=True )
                # sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def show_topics_in_sentences(_ldamodel, _corpus, _text):
    _text = list(_text["Context"])
    sent_topic = format_topics_sentences_module(ldamodel=_ldamodel, corpus=_corpus, texts=_text)
    df_dominant_topic = sent_topic.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']
    df_dominant_topic.head(10)
    st.markdown("---")
    st.markdown("### Show Topic modeling each Resumes:")
    fig = go.Figure(data=[go.Table(columnwidth = [1, 2, 2 , 2, 2], header=dict(values=["Document_No", "Dominant_Topic", "Topic_Perc_Contrib","Keywords"], line_color='darkslategray',
                                                fill_color='#f0a500'),
                                    cells=dict(values=[df_dominant_topic["Document_No"], df_dominant_topic["Dominant_Topic"], df_dominant_topic["Topic_Perc_Contrib"], df_dominant_topic["Keywords"]], line_color='darkslategray',
                                                fill_color='#f4f4f4'))
                            ])
    fig.update_layout(width=800, height=500)
    st.write(fig)
    st.markdown("---")


def resume_printing(Ranked_resumes):
    option_2 = st.selectbox("Show the Best Matching Resumes?", options=[
    'YES', 'NO'])
    if option_2 == 'YES':
        indx = st.slider("Which resume to display ?:",
                        1, Ranked_resumes.shape[0], 1)

        st.write("Displaying Resume with Rank: ", indx)
        st.markdown("---")
        st.markdown("## **Resume** ")
        value = Ranked_resumes.iloc[indx-1, 2]
        st.markdown("#### The Word Cloud For the Resume")
        wordcloud = WordCloud(width=800, height=800,
                            background_color='white',
                            colormap='viridis', collocations=False,
                            min_font_size=10).generate(value)
        plt.figure(figsize=(7, 7), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)

        # st.write("With a Match Score of :", Ranked_resumes.iloc[indx-1, 6])
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Resume"],
                        fill_color='#f0a500',
                        align='center', font=dict(color='white', size=16)),
            cells=dict(values=[str(value)],
                    fill_color='#f4f4f4',
                    align='left'))])

        fig.update_layout(width=800, height=1200)
        st.write(fig)
        # st.text(df_sorted.iloc[indx-1, 1])
        st.markdown("---")

async def main():
    # load title dasboard
    choose_type = load_title_dasboard()
    # load data input
    if choose_type == 'CSV':
        Resumes, Jobs = load_input_data_Resumes(), load_input_data_Jobs()
    else:
        Resumes, Jobs = load_input_data_Resumes_docx(), load_input_data_Jobs_docx()
    try:
        Resumes_origin = copy.copy(Resumes)
        Jobs_origin = copy.copy(Jobs)
        # show description name ny data origin
        _ , index  = await asyncio.gather(show_description_name(Jobs), show_description(Jobs))
        # Extraction information from JD
        info_retrieval = await asyncio.gather(show_information_retrieval(Jobs, index))
        # Extraction information from Resumes
        info_retrieval_resumes = await asyncio.gather(show_information_retrieval_resumes(Resumes))
        # Fillter JD by input keyword
        await asyncio.gather(find_JD_by_keyword())
        # Convert type Resume and JB origin to type use TF-IDF
        Jobs_origin = file_Readert(Jobs_origin)
        Resumes_origin = file_Readert(Resumes_origin)
        # Matching Rule use 5 model 
        await asyncio.gather(show_matching_rule(index, info_retrieval[0], info_retrieval_resumes[0], Jobs_origin, Resumes_origin))
        # Topic modeling
        Ranked_resumes = ranked_resumes(Resumes_origin, Jobs_origin, index)
        lda_model, corpus = tfidf(Resumes_origin)
        topic_word_clound(lda_model)
        show_topics_in_sentences(lda_model, corpus, Resumes_origin)
        show_sunburst_graph(lda_model, corpus, Resumes_origin)
        resume_printing(Ranked_resumes)
    except Exception as e:
        print ("Warring:", e)
asyncio.run(main(), debug=False) 



# QueryDatabase("a", load_config()).insert_resume_database(r"C:\Users\huuph\OneDrive\Documents\resume_matching\Resume_matching\Resume_Data.csv")
# QueryDatabase("a", load_config()).insert_resume_it_viec_database(r"C:\Users\huuph\OneDrive\Documents\resume_matching\Resume_matching\Data\IT_viec\ResumeDataSet.csv")
# QueryDatabase("a", load_config()).insert_job_it_viec_database(r"C:\Users\huuph\OneDrive\Documents\resume_matching\Resume_matching\Data\IT_viec\jobs.csv")
