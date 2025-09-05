# Import libraries
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
df = pd.read_csv('nyc-jobs-1.csv', encoding="utf-8")

resume = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]
resume = resume[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]

# Create a new column called 'data' and merge the values of the other columns into it
df['data'] = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
# Drop the individual columns if you no longer need them
df.drop(['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills'], axis=1, inplace=True)

data = list(df['data'])
tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(data)]

# Model initialization
model = Doc2Vec(vector_size = 50,
min_count = 5,
epochs = 50,
alpha = 0.05,
workers = 8,
)
# Vocabulary building
model.build_vocab(tagged_data)
# Get the vocabulary keys
keys = model.wv.key_to_index.keys()
# Print the length of the vocabulary keys
print(len(keys))
# Train the model
for epoch in range(model.epochs):
    print(f"Training epoch {epoch+1}/{model.epochs}")
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    if epoch % 10 == 0:
        model.save(f"models/model_{epoch}.model")

model.save('cv_job_maching.model')
print("Model saved")
pdf = PyPDF2.PdfReader('Akshay_Srimatrix.pdf')
resume = ""
for i in range(len(pdf.pages)):
    pageObj = pdf.pages[i]
    resume += pageObj.extract_text()
# JD by input text:
# jd = input("Paste your JD here: ")

jd = "Job Description – Data Scientist: Position: Data Scientist | Location: [City, Country or Remote] | Employment Type: Full-time | About the Role: We are seeking a highly motivated Data Scientist to join our team. The ideal candidate will be passionate about leveraging data to solve business problems, build predictive models, and deliver actionable insights. You will work closely with cross-functional teams including engineering, product, and business stakeholders to design data-driven solutions. | Key Responsibilities: Collect, clean, and analyze structured and unstructured data from multiple sources; Develop and deploy machine learning and statistical models to support business decisions; Perform exploratory data analysis (EDA) and visualize insights using dashboards and reports; Work with engineers to design scalable data pipelines and solutions; Communicate findings and recommendations to both technical and non-technical stakeholders; Stay updated with the latest advancements in machine learning, AI, and big data technologies. | Qualifications: Bachelor’s or Master’s degree in Computer Science, Statistics, Mathematics, Data Science, or a related field; Strong programming skills in Python or R with experience using libraries like Pandas, NumPy, Scikit-learn, PyTorch, or TensorFlow; Solid understanding of statistics, probability, and machine learning algorithms; Proficiency in SQL and relational databases; Hands-on experience with data visualization tools (Tableau, Power BI, or Matplotlib/Seaborn); Strong problem-solving skills; Excellent communication skills. | Preferred Qualifications: Experience with big data frameworks (Hadoop, Spark); Knowledge of cloud platforms (AWS, GCP, Azure); Familiarity with MLOps, CI/CD, and model deployment practices; Experience in NLP, computer vision, or recommender systems. | What We Offer: Competitive salary and performance-based bonuses; Flexible working hours and remote options; Opportunities for professional growth and continuous learning; Collaborative and innovative work environment."

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