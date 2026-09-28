import streamlit as st
import pandas as pd
import joblib
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import re
import os


st.set_page_config(page_title='AI Job Matcher', page_icon="💼👔", layout="wide")

key = "AIzaSyBy6DnKqeIih0CS-E3oZkLrNTYPqiu9bQ0"
client = genai.Client(api_key=key)

if "role" not in st.session_state:
    st.session_state.role = None

ROLES = [None, "Employer","Job Seeker"]
@st.cache_resource
def load_models():
    vect =joblib.load('keywords_vectorizer.pkl')
    return vect
    
def login():
    if st.session_state.role == None:
        st.header("Welcome To Job Matching 💼👔")
        role = st.selectbox("Choose a role", ROLES)
        if st.button("Confirm"):
            st.session_state.role = role
            st.rerun()

def extract_data(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def gemini_extractor(raw_data):
    prompt= f"""Extract the following from this resume. Return only the data extacly like the this:
    
    Title:[Job Title]
    Years: [Integer of the years of experience]
    Level: [either entry level or senior level]
    Keywords: [Seperate diffrent skills with comma]

    Resume: {raw_data}
    """
    response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
    )

    user_data = {"Title":"","Years":0, "Level": "entry level", "Keywords":""}
    for line in response.text.split('\n'):
        if line.startswith("Title"): user_data["Title"] = line.replace("Title:", "").replace("Title","").strip().lower()
        elif line.startswith("Years:"):
            nums = re.findall(r'\d+', line)
            user_data["Years"]= int(nums[0]) if nums else 0
        elif line.startswith("Level:"): 
            user_data["Level"] = line.replace("Level:", "").replace("Level", "").strip().lower()
        elif line.startswith("Keywords:"): 
            user_data["Keywords"] = line.replace("Keywords:", "").replace("Keywords", "").strip().lower()
    return user_data

def job_matching(user_data):
    jobsDB = pd.read_csv('cleaned_job_dataset.csv')
    user_vec = vect.transform([user_data["Keywords"]])
    job_vec = vect.transform(jobsDB['Cleaned_Keywords'].fillna(''))
    similatrity_scr = cosine_similarity(user_vec, job_vec).flatten()

    results =jobsDB.copy()
    results['Match_Score'] = similatrity_scr *100 

    total_scr = []
    for i, row in results.iterrows():
        score = row['Match_Score']

        if user_data['Years']>= row['Cleaned_YearsOfExperience']:
            score +=15
        else:
            score -=20
        
        if user_data['Title'] in str(row['Cleaned_Title']):
            score +=5

        total_scr.append(min(max(score,0),99.9))

    results['Final_Score'] =total_scr
    return results.sort_values(by='Final_Score', ascending=False)

vect = load_models()
login()

role = st.session_state.role
page_dict={}

if st.session_state.role != None:
    st.sidebar.title("Chatbot Advisory")
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []
    
    for chats in st.session_state.chatHistory:
        with st.sidebar.chat_message(chats["role"]):
            st.markdown(chats["content"])

    prompt = st.sidebar.chat_input("Chat with HelpBot for advice")
    if prompt:
        st.session_state.chatHistory.append({"role":"user", "content":prompt})
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            reply = response.text
            st.session_state.chatHistory.append({"role":"assistant", "content":reply})
            with st.sidebar.chat_message("assistant"):
                st.markdown(reply)
        except:
            st.sidebar.error("Oops looks like out HelpBot is out for a while...")

    if st.session_state.role in ["Employer"]:
        st.title("Employer")
        st.write("Welcome Employer! Ready to find your new employees?")

        comp_name  = st.text_input("Company Name")
        comp_email  = st.text_input("Company Email")
        job_title = st.text_input("Job Title")
        job_desc = st.text_area("Job Description (Enter in duties, skills and requiremnts for the job)")
        job_level = st.selectbox("Choose the experience level", ['Entry Level','Senior Level']).lower()
        job_years = st.number_input("Choose the number of years for the job",min_value=0, max_value=20, value=0)
        if st.button("Find Candidates"):
            if job_desc:
                with st.spinner("Analyzing requirements and searching candidates"):
                    prompt = f"Extract only the core technical skills from the job secription and use comma to seperate every skill Job Description:{job_desc}"
                    response = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt
                            )
                    job_keywords = response.text.strip().lower()    

                    new_job = {
                            "Cleaned_Title":job_title.lower(),
                            "Cleaned_ExperienceLevel": job_level,
                            "Cleaned_YearsOfExperience":job_years,
                            "Cleaned_Keywords":job_keywords,
                            "Company_Name": comp_name,
                            "Company_Email": comp_email
                        }

                    df_new_job = pd.DataFrame([new_job])
                    df_new_job.to_csv("cleaned_job_dataset.csv", mode='a', index=False, header=False)              
                    try:
                        candidates = pd.read_csv('candidates.csv')
                        job_vec = vect.transform([job_keywords])
                        candidate_matrix = vect.transform(candidates['Keywords'].fillna(''))
                        similarity_scr = cosine_similarity(job_vec, candidate_matrix).flatten()

                        candidates['Match_Score'] = similarity_scr*100
                        results = candidates.sort_values(by='Match_Score', ascending= False)


                        for i, row in results.iterrows():
                            with st.container():
                                st.markdown(f"{row['Name']} - {row['Title'].title()} (Score:{row['Match_Score']:.1f}%)")
                                st.write(f"Skills:{row['Keywords']}")
                                st.link_button(f"Email: {row['Name']}", f"mailto:{row['Email']}")
                                st.markdown("___________")
                    except:
                        st.error("Oops looks like there are no candidates in the database that match well enough for what you are looking for")

    elif st.session_state.role in ["Job Seeker"]:
        st.title("Job Seeker 💼")
        st.write("Welcome Job Seeker! Ready to find your dream job?")

        candid_name = st.text_input("Name:")
        candid_email = st.text_input("Email Adress:")
        
        upload_file = st.file_uploader("Upload Resume in PDF", type=['pdf'])
        if upload_file is not None and candid_name and candid_email:
            if st.button("Confirm Upload"):
                with st.spinner("We are analyzing your resume..."):
                    raw_text = extract_data(upload_file)
                    user_data = gemini_extractor(raw_text)
                    
                    matches = job_matching(user_data)

                    user_data["Name"] = candid_name
                    user_data["Email"] = candid_email
                    df_new_candidate = pd.DataFrame([user_data])

                    df_new_candidate.to_csv("candidates.csv", mode='a', index=False, header=not os.path.exists('candidates.csv'))
                    st.subheader("Top 5 Job Matches For You:")
                    st.dataframe(matches[['Cleaned_Title','Cleaned_ExperienceLevel','Cleaned_YearsOfExperience','Final_Score','Company_Name',"Company_Email"]].head(5))


    if st.button("Log Out"):
        st.session_state.role = None
        st.rerun()

