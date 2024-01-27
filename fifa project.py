#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pip', 'install wordcloud')
from sklearn.metrics import accuracy_score


# In[2]:


import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


# In[3]:


df = pd.read_csv('fifa_world_cup_2022_tweets.csv')


# In[4]:


# here i am trying to see the first 5 rows of the dataframe 
df.head()


# In[5]:


# here i am trying to see the dataframe destribution and baisc properties of the numerical data in the dataframe
df.describe()


# In[6]:


# trying to see if there is missing data
df.isnull().sum()


# In[7]:


# just wants to see propetries of the dataframe
df.info()


# In[8]:


#Convert Date Created column to datetime
df['Date Created']= pd.to_datetime(df['Date Created'])
df['updated_date']= df['Date Created'].dt.date
df


# In[9]:


df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
df


# In[10]:


#counts the occurrences of each unique value in the Source_of_Tweet & creat a bar plot for it 
df.Source_of_Tweet.value_counts().head(10).plot.bar(figsize=(7,5))


# In[11]:


# twt_trend= df.groupby(['updated_date'])['Tweet'].count()

# fig= plt.figure(figsize= (8, 5))
# plt.plot(twt_trend.index, twt_trend.values, color='blue')
# plt.title('Tweet Trend')
# plt.xlabel('Dates')
# plt.ylabel('Tweets')
# plt.show()


# In[12]:


# trying to clean up our text,by removing various type of noise that may that may interfere with our text analysis

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '.text')
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                          text)
    text = (re.sub('[\W]+', '', text.lower())+
           ''.join(emoticons).replace('-', ''))
    return text


# In[13]:


def tokenizer(text):
    return text.split()
tokenizer('text')


# In[14]:


#transforming words into it's rooot form
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('text')


# In[15]:


#trying to remove every stop words from the dataset
from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('text')
if w not in stop]


# In[16]:


# Create a SentimentIntensityAnalyzer object and store it in the variable 'sia'
sia=SIA()
scores=[]
for i in range(len(df['Tweet'])):
    
    score = sia.polarity_scores(df['Tweet'][i])
    score=score['compound']
    scores.append(score)
sentiment=[]
for i in scores:
    if i>=0.05:
        sentiment.append('Positive')
    elif i<=(-0.05):
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
df['sentiment']=pd.Series(np.array(sentiment))


# In[17]:


# to see the top 10 head of the df 
df.head(10)


# In[18]:


temp = df.groupby('sentiment').count()['Tweet'].reset_index().sort_values(by='Tweet',ascending=False)
temp.style.background_gradient(cmap='coolwarm_r')


# In[19]:


# to see the popular worlds
from wordcloud import WordCloud, STOPWORDS
stopwords= set(STOPWORDS)

text_cloud= " ".join(df['Tweet'].tolist())

wordcloud= WordCloud(stopwords=stopwords, max_words=100, \
                    background_color="white").generate(text_cloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.title('General Statement WordCloud')
plt.show();


# In[20]:


# To see the percentage of each tweet sentiment breakdown
temp2=temp.set_index('sentiment')
plt.figure(figsize=(10,6))
plt.pie(temp2.Tweet, labels=temp2.index, autopct='%1.2f%%')
plt.title("df Tweets Sentiments Breakdown");


# In[21]:


# X_train = df.loc[:11262, 'Tweet'].values
# y_train = df.loc[:11262, 'sentiment'].values
# X_test = df.loc[11262:, 'Tweet'].values
# y_test = df.loc[11262:, 'sentiment'].values


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score



# # Split data into features and target
X = df['Tweet']  # Adjust 'text_column' to the actual column name containing text
y = df['sentiment']  # Adjust 'sentiment_column' to the actual column name containing labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression()

# Define hyperparameters for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear'],  # Solver for logistic regression
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

# Fit the model with the training data
grid_search.fit(X_train_vec, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)


# In[25]:


from sklearn.metrics import confusion_matrix


# In[27]:


cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[36]:


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix');


# In[42]:


# Assuming you have y_test (true labels) and y_pred (predicted labels)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Extract precision, recall, and F1-score for each class
class_labels = list(classification_rep.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
precision = [classification_rep[label]['precision'] for label in class_labels]
recall = [classification_rep[label]['recall'] for label in class_labels]
f1_score = [classification_rep[label]['f1-score'] for label in class_labels]

# Create a bar chart for precision, recall, and F1-score
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.bar(class_labels, precision, label='Precision')
plt.bar(class_labels, recall, label='Recall', alpha=0.7)
plt.bar(class_labels, f1_score, label='F1-Score', alpha=0.5)
plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.ylim(0, 1)  # Set the y-axis range from 0 to 1
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.legend()
plt.tight_layout()  # Ensure labels are not cut off
plt.show()


# In[50]:


# Create a function to annotate bars with their values
def annotate_bars(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')

# Create a separate bar chart for Precision
plt.figure(figsize=(8, 6))
precision_bars = plt.bar(class_labels, precision)
plt.xlabel('Classes')
plt.title('Precision for Each Class')
plt.ylim(0, 1)
plt.xticks(rotation=0)
annotate_bars(plt.gca(), precision_bars)


# In[52]:


# Create a separate bar chart for Recall
plt.figure(figsize=(8, 6))
recall_bars = plt.bar(class_labels, recall)
plt.xlabel('Classes')
plt.title('Recall for Each Class')
plt.ylim(0, 1)
plt.xticks(rotation=0)
annotate_bars(plt.gca(), recall_bars)


# In[53]:


# Create a separate bar chart for F1-Score
plt.figure(figsize=(8, 6))
f1_score_bars = plt.bar(class_labels, f1_score)
plt.xlabel('Classes')
plt.title('F1-Score for Each Class')
plt.ylim(0, 1)
plt.xticks(rotation=0)
annotate_bars(plt.gca(), f1_score_bars)


# In[55]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have y_test (true labels) and y_pred (predicted labels)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Create a bar chart for accuracy, precision, recall, and f1-score
plt.figure(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['blue', 'green', 'orange', 'red']

bars = plt.bar(metrics, values, color=colors)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Overall Model Evaluation Metrics')

# Add legends
for bar, metric in zip(bars, metrics):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{metric}: {values[metrics.index(metric)]:.2f}',
             ha='center', color='black', fontsize=12)

# Show the chart with legends
# plt.legend(metrics)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# In[58]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have y_test (true labels) and y_pred (predicted labels)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Create a bar chart for accuracy, precision, recall, and f1-score
plt.figure(figsize=(8, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['blue', 'green', 'orange', 'red']

bars = plt.bar(metrics, values, color=colors)
plt.xlabel('Metrics')
plt.title('Overall Model Evaluation Metrics')

# Add values on the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2f}',
             ha='center', color='black', fontsize=12)

plt.ylim(0, 1)
plt.tight_layout()
plt.show()

