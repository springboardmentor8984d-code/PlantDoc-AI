#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas nltk')


# In[2]:


#importing the required libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[4]:


#load the dataset
df = pd.read_excel(r"C:\Users\paul\Downloads\Plant Symptoms Dataset(AutoRecovered).xlsx")


# In[5]:


#to display the headers of the dataset
df.head()


# In[6]:


#Text Pre-Processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # lowercase
    text = ''.join([ch for ch in text if ch not in string.punctuation])  # remove punctuation
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# In[7]:


#To clean the text 
text_columns = ['Plant Species (Host)','Disease/Condition (Identification)','Observable Symptoms', 'Likely Causes & Risk Factors', 'Management & Treatment Recommendations','Preventive Measures & Monitoring']
for col in text_columns:
    df[col] = df[col].astype(str).apply(clean_text)


# In[10]:


# Total species count
num_species = df['Plant Species (Host)'].nunique()

# Diseases per species
disease_count = df.groupby('Plant Species (Host)')['Disease/Condition (Identification)'].nunique().sort_values(ascending=False)

print(f"🌱 Total Plant Species: {num_species}\n")
print("🦠 Diseases per Species:\n")
print(disease_count)


# In[11]:


#To save the cleaned dataset to the local device
df.to_excel("Plant_Symptoms_Dataset_Cleaned.xlsx", index=False)


# In[ ]:




