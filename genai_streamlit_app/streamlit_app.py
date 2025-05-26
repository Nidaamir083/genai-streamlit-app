import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"


import streamlit as st
from PIL import Image

# Set page config first
st.set_page_config(page_title="Find Your Research", layout="wide")

# Add background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://astrixinc.com/wp-content/uploads/2025/04/AI-Image-1.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         .main {{
             background-color: rgba(255, 255, 255, 0.9);
             padding: 2rem;
             border-radius: 10px;
             margin: 2rem 0;
         }}
         h1 {{
             color: #4a4a4a;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# App content
st.title("Find Your Research")
st.write("Welcome to FYR!")

# Add a sidebar
with st.sidebar:
    st.header("Settings")
    option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])
    
# Main content area
col1, col2 = st.columns(2)
with col1:
    st.subheader("Feature 1")
    st.write("This is some content in the first column.")

with col2:
    st.subheader("Feature 2")
    st.write("This is some content in the second column.")

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Find Your Research", 
    layout="wide",
    page_icon="🔬"
)
# Set background image (must come right after page config)
background_url = "https://astrixinc.com/wp-content/uploads/2025/04/AI-Image-1.jpg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
import arxiv
import requests
from Bio import Entrez
from transformers import pipeline

Entrez.email = "nida.amir@gmail.com"

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_model()

def fetch_pubmed_articles(query, start_year=2015, end_year=2024, max_results=20):
    handle = Entrez.esearch(db="pubmed", term=query, mindate=f"{start_year}/01/01",
                            maxdate=f"{end_year}/12/31", retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    abstracts = [a.strip() for a in handle.read().split("\n\n") if len(a.strip()) > 100]
    return pd.DataFrame({"abstract": abstracts, "source": ["PubMed"] * len(abstracts)})

def get_wikipedia_background(topic):
    try:
        summary = wikipedia.summary(topic, sentences=5)
        return [{"source": "Wikipedia", "title": topic, "date": topic, "summary": summary}]
    except Exception:
        return []

def fetch_arxiv_articles(query, max_results=5):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    articles = []
    for result in search.results():
        if 2015 <= result.published.year <= 2024:
            articles.append({
                "source": "arXiv",
                "title": result.title,
                "date": result.published,
                "summary": result.summary
            })
    return articles

def build_merged_report(topic, pubmed_limit=5, arxiv_limit=5):
    pubmed = fetch_pubmed_articles(topic, max_results=pubmed_limit)
    arxiv_articles = fetch_arxiv_articles(topic, max_results=arxiv_limit)
    wiki = get_wikipedia_background(topic)
    return pubmed.to_dict('records') + arxiv_articles + wiki

def visualize_results(data):
    for doc in data:
        doc['source'] = doc.get('source', 'Unknown')
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='source', order=df['source'].value_counts().index, palette='pastel', ax=ax)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def ask_scientific_question(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return qa_pipeline(prompt, max_new_tokens=300)[0]["generated_text"].strip()

st.title("🔬 GenAI for Scientific QA and Drug Repurposing")
topic = st.text_input("Enter a research topic:", "drug repurposing for anaplastic thyroid cancer")


        

st.markdown("<h1 style='color: #4CAF50;'>🔬 Find Your Research</h1>", unsafe_allow_html=True)

if topic:
    with st.spinner("Fetching data..."):
        data = build_merged_report(topic)
    st.success("Data fetched successfully!")

    st.subheader("📊 Source Distribution")
    visualize_results(data)

    st.subheader("📚 Sources")
    for doc in data:
        st.markdown(f"- **{doc.get('source')}**: {doc.get('title', doc.get('abstract', doc.get('summary', 'N/A')))[:80]}...")

    st.subheader("🧠 Ask a Scientific Question")
    question = st.text_input("What would you like to ask?", "What AI tools are used in the diagnosis of Thyroid cancer?")
    if st.button("Get Answer"):
        context_texts = " ".join(doc.get('summary', '') or doc.get('abstract', '') for doc in data)[:4000]
        answer = ask_scientific_question(question, context_texts)
        st.markdown("### 🤖 Answer")
        st.write(answer)
