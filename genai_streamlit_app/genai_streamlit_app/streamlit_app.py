
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
import arxiv
import requests
from Bio import Entrez
from Bio import Medline
from transformers import pipeline

Entrez.email = "nida.amir0083@example.com"

# Add background image using CSS
st.markdown(
    """
    <style>
    body {
        background-image: url('https://huuray.com/wp-content/uploads/2024/01/AI-research-tool-comp.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85); /* semi-transparent white */
        border-radius: 15px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_model()

from Bio import Medline

def fetch_pubmed_articles(query, start_year=2015, end_year=2024, max_results=20):
    handle = Entrez.esearch(db="pubmed", term=query, mindate=f"{start_year}/01/01",
                            maxdate=f"{end_year}/12/31", retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    
    articles = []
    for rec in records:
        abstract = rec.get("AB", "")
        pubdate = rec.get("DP", "Unknown")
        if len(abstract.strip()) > 100:
            articles.append({
                "abstract": abstract,
                "source": "PubMed",
                "title": rec.get("TI", "No title"),
                "date": pubdate
            })
    return pd.DataFrame(articles)

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
        doc['date'] = str(doc.get('date', 'Unknown'))[:4]  # Extract year only
    
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='date', hue='source', palette='pastel', ax=ax)
    ax.set_title("Article Counts by Year and Source")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def ask_scientific_question(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return qa_pipeline(prompt, max_new_tokens=300)[0]["generated_text"].strip()

st.title("🔬 Find Your Research")
topic = st.text_input("Enter a research topic:", "AI in thyroid Cancer")

if topic:
    with st.spinner("Fetching data..."):
        data = build_merged_report(topic)
    st.success("Data fetched successfully!")

    st.subheader("📊 Source Distribution")
    visualize_results(data)

    st.subheader("📚 Sources")
    for doc in data:
        title_or_text = doc.get('title', doc.get('abstract', doc.get('summary', 'N/A')))[:80]
        date = doc.get('date', 'N/A')
        st.markdown(f"- **{doc.get('source')}** ({date}): {title_or_text}...")

    # After visualizations and source listing
st.subheader("📚 Sources")
for doc in data:
    title = doc.get('title', 'No title')
    summary = doc.get('summary', doc.get('abstract', 'No summary available'))
    date = doc.get('date', 'N/A')
    if isinstance(date, pd.Timestamp) or hasattr(date, 'strftime'):
        date = date.strftime('%Y-%m-%d')
    else:
        date = str(date)
    st.markdown(f"**Source:** {doc.get('source', 'Unknown')}")
    st.markdown(f"**Title:** {title}")
    st.markdown(f"**Date:** {date}")
    st.markdown(f"**Summary:** {summary[:300]}...")  # show first 300 chars
    st.markdown("---")

# NEW SECTION: Overall Summary
st.subheader("📝 Overall Summary of All Articles")

# Combine summaries/abstracts
combined_texts = " ".join(doc.get('summary', '') or doc.get('abstract', '') for doc in data)

# Ask the QA model to summarize
if st.button("Generate Overall Summary"):
    with st.spinner("Generating summary..."):
        prompt = f"Summarize the key points and findings from the following articles:\n\n{combined_texts[:4000]}"
        overall_summary = qa_pipeline(prompt, max_new_tokens=500)[0]["generated_text"].strip()
    st.markdown("### 🧩 Combined Summary")
    st.write(overall_summary)


    st.subheader("🧠 Ask a Scientific Question")
    question = st.text_input("What would you like to ask?", "What AI tools are used in the diagnosis of Thyroid cancer?")
    if st.button("Get Answer"):
        context_texts = " ".join(doc.get('summary', '') or doc.get('abstract', '') for doc in data)[:4000]
        answer = ask_scientific_question(question, context_texts)
        st.markdown("### 🤖 Answer")
        st.write(answer)
