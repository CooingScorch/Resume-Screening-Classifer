import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np

try:
    df = pd.read_csv('cleaned_job_dataset.csv')
except FileNotFoundError:
    print("Error could not find file")
    exit()

# (BoW)
vectorizer = CountVectorizer(stop_words='english') 
vectorizer.fit_transform(df['Cleaned_Keywords'])


X_trainText, X_testText, y_trainTitle, y_testTitle = train_test_split(df['Cleaned_Keywords'], df['Cleaned_Title'], test_size=0.2, random_state=42)

X_trainVec = vectorizer.transform(X_trainText)
X_testVec = vectorizer.transform(X_testText)

similarity_matrix = cosine_similarity(X_testVec, X_trainVec)

top1 =0
top5 =0
precision_scr =0
K=5

for i in range(len(X_testText)):
    scores = similarity_matrix[i]
    topIndex = np.argsort(scores)[::-1]

    actual_title = y_testTitle.iloc[i]
    topK_title = [y_trainTitle.iloc[idx] for idx in topIndex[:K]]


    if y_trainTitle.iloc[topIndex[0]]== actual_title:
        top1 +=1
    if actual_title in topK_title:
        top5 +=1
    topK_match_count = sum(1 for title in topK_title if title == actual_title)
    precision_scr += (topK_match_count/K)

totalTests = len(X_testText)

print("Performance Metrics (BoW)")
print(f"Total Test Samples Evluated: {totalTests}")
print(f" Top 1 Accuracy:{(top1/totalTests)}")
print(f" Top 5 Accuracy:{(top5/totalTests)}")
print(f"Precision:{(precision_scr/totalTests)}")

