import pandas as pd
import re
data = pd.read_csv('job_dataset.csv')

print("\n\nFirst 5 rows of the dataset:")
print(data.head())

def clean_default(dataNorm):
    dataNorm = str(dataNorm).lower()
    dataNorm = dataNorm.split('-')[0].strip()
    return dataNorm

def clean_yearsEXP(dataNorm):
    dataNorm = str(dataNorm)
    dataNorm = re.findall(r'\d+',dataNorm)
    if dataNorm:
        return int(dataNorm[0])
    else:
        return 0

def clean_keywords(dataNorm):
    dataNorm = str(dataNorm).lower()
    return dataNorm

# Identifying null values in the variables
isnull = data.isnull()

# Counting the null values by Column
isnullCount = isnull.sum()
print("Number of Nulls:", isnullCount)

# Total Nulls in Dataset
Total_null_count = isnull.sum().sum()
print("Total Nulls:", Total_null_count)

# Identify and display all duplicate rows
duplicate_rows = data[data.duplicated()]
print('\n\nDuplicate Rows:', duplicate_rows)

# removes any duplicates and drops any records that have missing values
data = data.drop_duplicates(keep='first')
data = data.dropna(subset=['Keywords','Title'])

# normlization
data['Cleaned_Keywords'] = data['Keywords'].apply(clean_keywords)
data['Cleaned_YearsOfExperience'] = data['YearsOfExperience'].apply(clean_yearsEXP)
data ['Cleaned_Title'] = data['Title'].apply(clean_default)

#FORMATTING EXPERIENCE LEVEL: Unify the categories
unify_mapping ={
    'fresher': 'entry level',
    'entry level': 'entry level',
    'entry-level': 'entry level',
    'experienced': 'senior level',
    'senior level': 'senior level',
    'senior-level': 'senior level'
}

data['Cleaned_ExperienceLevel'] = data['ExperienceLevel'].str.lower().map(unify_mapping).fillna(data['ExperienceLevel'])
data_cleaned = data[['Cleaned_Title','Cleaned_ExperienceLevel','Cleaned_YearsOfExperience','Cleaned_Keywords']].copy()
#final verification data is cleaned
print(data[['Cleaned_Title', 'Cleaned_YearsOfExperience','Cleaned_ExperienceLevel','Cleaned_Keywords']].head(50))

#cleaned data is saved into a new file for trainig
data_cleaned['Company_Name'] = ["Neuxon"+str(i) for i in range(1, len(data_cleaned)+1)]
data_cleaned['Company_Email'] = ["hrNeuxon"+str(i)+ "@gmail.com"for i in range(1, len(data_cleaned)+1)]
data_cleaned.to_csv('cleaned_job_dataset.csv',index=False)
print('\n Data Cleaned and Saved Successfully')
