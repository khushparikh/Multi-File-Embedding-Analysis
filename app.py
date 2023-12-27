import streamlit as st
import openai
import numpy as np
import docx2txt
import pandas as pd
from PyPDF2 import PdfReader

# Define the cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define the function that generates the embeddings
def generate_embeddings(prompt):
    openai.api_key = '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'
    response = openai.Embedding.create(
        input= prompt,
        model="text-embedding-ada-002"
    )
    # print(response)
    embeddings = response['data'][0]['embedding']
    return embeddings

# Define the output interface
def compare_files(files):
    fileName = files[0].name
    print(files)
    # Extraction of text from model source
    text1 = ""
    if fileName[len(fileName)-4:len(fileName)] == ".txt":
        for line in files[0]:
            text1 += line
    elif fileName[len(fileName)-5:len(fileName)] == ".docx":
        text1 = docx2txt.process(files[0])
    else:
        reader = PdfReader(files[0])
        for pageNum in range(len(reader.pages)):
            page = reader.pages[pageNum]
            text1 += page.extract_text()
    
    # Extraction of text from every other source 
    text2 = ""
    scatterData = {'Article Name': [], 'Similarity Percentage in Relation to Model Article': []}
    for i in range(len(files)):
        fileName = files[i].name
        if fileName[len(fileName)-4:len(fileName)] == ".txt":
            for line in files[i]:
                text2 += line
        elif fileName[len(fileName)-5:len(fileName)] == ".docx":
            text2 = docx2txt.process(files[i])
        else:
            reader = PdfReader(files[i])
            for pageNum in range(len(reader.pages)):
                page = reader.pages[pageNum]
                text2 += page.extract_text()
        
        embedding1 = generate_embeddings(text1)
        embedding2 = generate_embeddings(text2)
        similarity = cosine_similarity(embedding1, embedding2)
        # Addition of Article Name and Similarity Score to Dictionary
        scatterData["Article Name"].append(fileName)
        scatterData["Similarity Percentage in Relation to Model Article"].append(similarity)
        print(scatterData)
        
    # Initialization of DataFrame using scatterData data
    df = pd.DataFrame(scatterData)
    return df

# Streamlit Input Initialization

with st.form("Multiple File Upload"):   
    # st.header('Multiple File Upload')
    uploaded_files = st.file_uploader('Upload your files', accept_multiple_files= True, type = ['pdf', 'docx', 'txt'])
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(compare_files(uploaded_files))
        st.bar_chart(data = compare_files(uploaded_files), x = "Article Name", y = "Similarity Percentage in Relation to Model Article")
