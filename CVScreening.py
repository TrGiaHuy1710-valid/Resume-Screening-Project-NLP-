import pandas as pd

df = pd.read_csv('nyc-jobs-1.csv')

resume = df[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]
resume = resume[['Business Title', 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills']]
print(resume)