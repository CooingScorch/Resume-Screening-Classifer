import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('cleaned_job_dataset.csv')
except FileNotFoundError:
    print("Error could not find file")
    exit()

# Throw away useless terms (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english') 
vectorizer.fit_transform(df['Cleaned_Keywords'])

print("...Saving trained model...")
joblib.dump(vectorizer, 'keywords_vectorizer.pkl')


print("Training complete\n'keywords_vectorizer.pkl' saved.")

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

top1_acc = top1/totalTests
top5_acc = top5/totalTests
precision_scr = precision_scr/totalTests
print("Performance Metrics")
print(f"Total Test Samples Evluated: {totalTests}")
print(f" Top 1 Accuracy:{top1_acc}")
print(f" Top 5 Accuracy:{top5_acc}")
print(f"Precision:{precision_scr}\n\n")


#cleaning the result 
performance_result ={
    "Metric": ["Top-1 Accuracy","Top-5 Acurracy", "Precision"],
    "Score":[top1_acc,top5_acc,precision_scr],
    "Percentage": [f"{top1_acc*100:.2f}%",f"{top5_acc*100:.2f}%",f"{precision_scr*100:.2f}%"]
}

df_metrics = pd.DataFrame(performance_result)


plt.figure(figsize=(8,5))
plt.title("Performance TF-IDF")
bars = plt.bar(df_metrics["Metric"], df_metrics["Score"], color=['#FF0000', '#00B3FF', '#00FF11' ])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, yval+0.02, f"{yval*100:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.savefig("performance_metrics.png", bbox_inches='tight')
print("Performance Metrics Image Saved")