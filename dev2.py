import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

nltk.download('stopwords')


data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'subject': [
        'Meeting at noon',
        'Win a free iPhone now',
        'Lunch plans',
        'Congratulations! You won a prize',
        'Project update',
        'Claim your prize now',
        'Can we reschedule?',
        'Earn money fast'
    ],
    'body': [
        'Let’s meet at noon to discuss the project.',
        'You have won a free iPhone. Click here to claim.',
        'Want to grab lunch today?',
        'You won a prize! Send your bank details.',
        'Project is on track for Friday deadline.',
        'Claim your reward by clicking the link.',
        'Can we reschedule the meeting to tomorrow?',
        'Make money fast with this one simple trick.'
    ]
}

df = pd.DataFrame(data)

print("Basic info:")
print(df.info(), "\n")

print("Label distribution:")
print(df['label'].value_counts(), "\n")

print("Missing values:")
print(df.isnull().sum(), "\n")

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label')
plt.title('Count of Ham and Spam Emails')
plt.show()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return filtered_tokens


ham_text = ' '.join(df[df['label']=='ham']['subject']) + ' ' + ' '.join(df[df['label']=='ham']['body'])
ham_words = preprocess_text(ham_text)
ham_word_counts = Counter(ham_words)
print("Most common words in ham emails:")
print(ham_word_counts.most_common(10))


spam_text = ' '.join(df[df['label']=='spam']['subject']) + ' ' + ' '.join(df[df['label']=='spam']['body'])
spam_words = preprocess_text(spam_text)
spam_word_counts = Counter(spam_words)
print("Most common words in spam emails:")
print(spam_word_counts.most_common(10))

plt.figure(figsize=(8,4))
WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(ham_word_counts).to_image().show()

plt.figure(figsize=(8,4))
WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(spam_word_counts).to_image().show()

