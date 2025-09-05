import nltk
import re
import PyPDF2
from PyPDF2.filters import JPXDecode
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from numpy.linalg import norm
pdf = PyPDF2.PdfReader('Akshay_Srimatrix.pdf')
resume = ""
for i in range(len(pdf.pages)):
    pageObj = pdf.pages[i]
    resume += pageObj.extract_text()
# JD by input text:
# jd = input("Paste your JD here: ")

# jd = "Job Description – Data Scientist: Position: Data Scientist | Location: [City, Country or Remote] | Employment Type: Full-time | About the Role: We are seeking a highly motivated Data Scientist to join our team. The ideal candidate will be passionate about leveraging data to solve business problems, build predictive models, and deliver actionable insights. You will work closely with cross-functional teams including engineering, product, and business stakeholders to design data-driven solutions. | Key Responsibilities: Collect, clean, and analyze structured and unstructured data from multiple sources; Develop and deploy machine learning and statistical models to support business decisions; Perform exploratory data analysis (EDA) and visualize insights using dashboards and reports; Work with engineers to design scalable data pipelines and solutions; Communicate findings and recommendations to both technical and non-technical stakeholders; Stay updated with the latest advancements in machine learning, AI, and big data technologies. | Qualifications: Bachelor’s or Master’s degree in Computer Science, Statistics, Mathematics, Data Science, or a related field; Strong programming skills in Python or R with experience using libraries like Pandas, NumPy, Scikit-learn, PyTorch, or TensorFlow; Solid understanding of statistics, probability, and machine learning algorithms; Proficiency in SQL and relational databases; Hands-on experience with data visualization tools (Tableau, Power BI, or Matplotlib/Seaborn); Strong problem-solving skills; Excellent communication skills. | Preferred Qualifications: Experience with big data frameworks (Hadoop, Spark); Knowledge of cloud platforms (AWS, GCP, Azure); Familiarity with MLOps, CI/CD, and model deployment practices; Experience in NLP, computer vision, or recommender systems. | What We Offer: Competitive salary and performance-based bonuses; Flexible working hours and remote options; Opportunities for professional growth and continuous learning; Collaborative and innovative work environment."
jd = "Job Description – Oracle PL/SQL Developer: Position: Oracle PL/SQL Developer | Location: [City, State or Remote] | Employment Type: Full-time | About the Role: We are looking for an experienced Oracle PL/SQL Developer with strong expertise in designing, developing, and optimizing complex database solutions. The ideal candidate will have hands-on experience in writing efficient SQL/PLSQL code, creating stored procedures, triggers, and packages, and supporting large-scale enterprise applications in sectors such as Banking, Insurance, or Finance. | Key Responsibilities: Design, develop, and maintain PL/SQL stored procedures, functions, packages, and triggers; Create and manage database objects including tables, indexes, materialized views, and constraints; Perform SQL/PLSQL performance tuning and optimization using AWR, ADDM, Explain Plan, Hints, and Tkprof; Develop and maintain data migration scripts using SQL*Loader, Data Pump (expdp/impdp), external tables, and UTL_FILE; Work with collections (associative arrays, nested tables, VARRAYs) and cursors (explicit, parameterized, bulk collect); Implement business rules using row-level, statement-level, instead-of, and compound triggers; Collaborate with functional/business teams to understand requirements and translate them into technical solutions; Participate in the full software development lifecycle (SDLC) including requirement gathering, design, development, testing, and deployment; Provide production support for backend database systems and troubleshoot performance issues; Work in Agile/Scrum environments with tools like JIRA, Confluence, Bitbucket, GitHub, Jenkins, Bamboo. | Qualifications: Bachelor’s or Master’s degree in Computer Science, Engineering, or related field; 5+ years of professional experience as an Oracle PL/SQL Developer; Strong expertise in Oracle 11g/12c database development; Experience in database design, ER modeling, normalization, and data flow diagrams; Proficiency in SQL performance tuning, query optimization, and partitioning; Experience with Oracle utilities such as Export/Import, Data Pump, and scheduling jobs with DBMS_JOB or DBMS_SCHEDULER; Hands-on experience with UNIX shell scripting for automation; Good knowledge of data migration techniques and working with flat files (CSV, XML); Excellent communication skills and ability to work both independently and in a team. | Preferred Qualifications: Exposure to Informatica or other ETL tools; Experience with version control and CI/CD tools (Git, Jenkins, Bamboo); Knowledge of data privacy regulations (GDPR, HIPAA) and compliance-related projects. | What We Offer: Competitive compensation and benefits package; Opportunity to work on large-scale enterprise database systems; Collaborative and innovative team environment; Professional growth and skill development opportunities."

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)

    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text
# Apply to CV and JD
input_CV = preprocess_text(resume)
input_JD = preprocess_text(jd)
# Model evaluation
model = Doc2Vec.load('cv_job_maching.model')
v1 = model.infer_vector(input_CV.split())
v2 = model.infer_vector(input_JD.split())
similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
print(round(similarity, 2))