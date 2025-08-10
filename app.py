
import streamlit as st
import pandas as pd
from resume_matcher import (
    df, job_descriptions_df, tfidf_vectorizer, resume_vectors,
    find_resumes_for_job, find_resumes_for_job_category
)

st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("📄 Resume Matcher")
st.write("Match job descriptions with the most relevant resumes from the dataset.")

# Sidebar for options
st.sidebar.header("Settings")
num_matches = st.sidebar.slider("Number of Matches", min_value=1, max_value=10, value=5)

# Option 1: Match by job category
st.header("1️⃣ Match by Job Category")
category_list = job_descriptions_df['category'].tolist()
selected_category = st.selectbox("Select a Job Category", category_list)

if st.button("Find Matches by Category"):
    results = find_resumes_for_job_category(
        selected_category,
        job_descriptions_df,
        df,
        tfidf_vectorizer,
        resume_vectors,
        num_matches=num_matches
    )
    if results:
        st.write(pd.DataFrame(results))
    else:
        st.info("No matches found for the selected category.")

# Option 2: Match by custom job description
st.header("2️⃣ Match by Custom Job Description")
custom_desc = st.text_area("Enter your Job Description here:")

if st.button("Find Matches by Custom Description"):
    if custom_desc.strip():
        results = find_resumes_for_job(
            custom_desc,
            df,
            tfidf_vectorizer,
            resume_vectors,
            num_matches=num_matches
        )
        if results:
            st.write(pd.DataFrame(results))
        else:
            st.info("No matches found for your custom description.")
    else:
        st.warning("Please enter a job description to match.")
