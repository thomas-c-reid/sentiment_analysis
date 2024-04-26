import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from keras.optimizers import AdamW
import seaborn as sns
import sys

# Process
# 1. Load the dataset
# 2. Data Cleaning - clean up the dataset so that it is only useful text that can be preprocessed in each review
# 3. EDA - investigate the data
# 4. Data pre-processing - preprocess the data
# 5. Building Model - build model architecture then compile
# 6. Cross-Validation Model Training - train model
# 7. Evaluation Model - evaluate model efficiency on test dataset and investigate metrics

# ===================================================================================================================================================

# 1. Loading in IMDB review dataset
filepath = 'IMDB_Dataset.csv'
df = pd.read_csv(filepath)
random_state = 42
num_of_epochs = 10
input_dim = 10000
view_data_cleaning_examples = False
view_eda_graphs = False
view_stemming_vs_lemmatization = False
print("INITIAL DATASET")
print(df.head())

# =========================================

# 2. Data-Cleaning
# 2.1. Convert 'sentiment' column numerical
# 2.2. Convert Reviews to lower case
# 2.3. Remove HTML tags
# 2.4. Remove URLs and links
# 2.5. Remove Punctuation
# 2.6. Spelling Correction
# 2.7. Remove stop words
# 2.8. Remove Emojis

# 2. Data-Cleaning
print("STEP: DATA CLEANING")
df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
df['review'] = df['review'].str.lower()


def remove_html_tags(reviews):
    return re.sub(r'<[^<]+?>', '', reviews)


def remove_url(text):
    return re.sub(r'http[s]?://\S+|www\.\S+', '', text)


def remove_punctuation(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text


def remove_stopwords(text):
    words_without_stopwords = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words_without_stopwords)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               # emoticons
                               u"\U0001F300-\U0001F5FF"
                               # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"
                               # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"
                               # flags (105)
                               u"\U00002702-\U00002780"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


if view_data_cleaning_examples:
    text_with_html_tags = "<p>This is a <strong>sample</strong> text with <a href='https://example.com'>HTML tags</a>.</p>"
    text_with_url = "Check out this website: https://example.com, it's really cool!"
    text_with_punctuation = "Wow!!! This is amazing, isn't it?"
    text_with_stopwords = "the quick brown fox jumps over the lazy dog"
    text_with_emojis = "I'm feeling 😊 very happy today! 🎉"

    text_without_html_tags = remove_html_tags(text_with_html_tags)
    text_without_url = remove_url(text_with_url)
    text_without_punctuation = remove_punctuation(text_with_punctuation)
    text_without_stopwords = remove_stopwords(text_with_stopwords)
    text_without_emojis = remove_emoji(text_with_emojis)

    print("*" * 25)
    print("Text with html tags")
    print(text_with_html_tags)
    print("Text without html tags (after data cleaning)")
    print(text_without_html_tags)
    print("*" * 25)

    print("Text with URL")
    print(text_with_url)
    print("Text without URL (after data cleaning)")
    print(text_without_url)
    print("*" * 25)

    print("Text with punctuation")
    print(text_with_punctuation)
    print("Text without punctuation (after data cleaning)")
    print(text_without_punctuation)
    print("*" * 25)

    print("Text with stopwords")
    print(text_with_stopwords)
    print("Text without stopwords (after data cleaning)")
    print(text_without_stopwords)
    print("*" * 25)

    print("Text with emojis")
    print(text_with_emojis)
    print("Text without emojis (after data cleaning)")
    print(text_without_emojis)
    print("*" * 25)

# Applying transformations to each column
df['review'] = df['review'].apply(remove_html_tags)
df['review'] = df['review'].apply(remove_url)
df['review'] = df['review'].apply(remove_punctuation)
df['review'] = df['review'].apply(remove_stopwords)
df['review'] = df['review'].apply(remove_emoji)

# =========================================

# 3. Exploratory Data Analysis
# 3.1. Creating word clouds to compare most frequent words
# 3.2. Viewing percentage of positive/negative reviews in dataset

if view_eda_graphs:
    print("STEP: EDA")
    # 3.1. Creating word clouds to compare most frequent words
    tokenized_review = df["review"].apply(word_tokenize)
    all_tokens = [token for sublist in tokenized_review for token in sublist]
    word_freq = Counter(all_tokens)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Whole Dataset Wordcloud")
    plt.axis("off")
    plt.show()

    positive_reviews_tokenized = df.loc[df['sentiment'] == 1, 'review'].apply(word_tokenize)
    all_tokens = [token for sublist in positive_reviews_tokenized for token in sublist]
    word_freq = Counter(all_tokens)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Positive Sentiment Dataset Wordcloud")
    plt.axis("off")
    plt.show()

    negative_reviews_tokenized = df.loc[df['sentiment'] == 0, 'review'].apply(word_tokenize)
    all_tokens = [token for sublist in negative_reviews_tokenized for token in sublist]
    word_freq = Counter(all_tokens)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Negative Sentiment Dataset Wordcloud")
    plt.axis("off")
    plt.show()

    # 3.2. Viewing percentage of positive/negative reviews in dataset
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.axis('equal')
    plt.show()
    # This is a balanced dataset - equal proportion of positive and negative examples within the dataset

# =========================================

# 4. Data PreProcessing
# 4.1. Stemming/Lemmatization
# 4.2. vectorisation

# 4.1.1 stemming (reducing words to their root/base form by removing suffixes and prefixes)
text = "The quick brown foxes are running and jumping over the lazy dogs"
ps = PorterStemmer()


def stem_words(text):
    return " ".join([ps.stem(word) for word in word_tokenize(text)])


stemmed_text = stem_words(text)

# 4.1.2 Lemmatization (Has the same function as stemming but usually produces better results)
lem = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lem.lemmatize(word) for word in word_tokenize(text)])


lemmatized_text = lemmatize_words(text)

if view_stemming_vs_lemmatization:
    print("*" * 25)
    print("Original Text: ", text)
    print("Stemmed Text: ", stemmed_text)
    print("lemmatized Text: ", lemmatized_text)
    print("*" * 25)

print("STEP: LEMMATIZATION")
df['review'] = df['review'].apply(lemmatize_words)
df['review'] = df['review'].apply(stem_words)

# 4.2 vectorisation

print("STEP: VECTORISATION")
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['review'])
pd.DataFrame.sparse.from_spmatrix(x,
                                  index=df['review'].index,
                                  columns=tfidf.get_feature_names_out())
y = df.iloc[:, 1]

# 4.3 RFE feature selection
print("STEP: FEATURE SELECTION")
# Initialize SelectKBest with chi-square test as the scoring function
selector = SelectKBest(score_func=chi2, k=10000)

x = selector.fit_transform(x, y)

# I also tried to do feature selection by using RFE with a randomForrestClassifier
# to select X (10000) best features as many of the columns after vectorisation will
# hold little importance in the overall goal of identifying the sentiment of a review
# However it is a very computationally expensive algorithm so i decided to use SelectKBest
# as a more efficient algorithm

# rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10000)
# rfe.fit(x, y)
# x = rfe.transform(x)

# =========================================

# 5. Building Models:
# 5.1. Simple 3 layer deep keras model

print("STEP: BUILDING MODEL")
input_shape = (input_dim,)

model = Sequential([
    Input(shape=input_shape),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model 1 Summary")
model.summary()

# =========================================

# 6. Cross-Validation Model Training
# 6.1. split data into test, train, validation split
# 6.2. Cross validtion
# 6.3. output simple accuracy score

# x - the vectorised reviews
# y - the sentiment of each review

# splitting the data into training dataset and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

histories = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
for train_idx, val_idx in CV.split(x_train, y_train):
    train_data, train_labels = x_train[train_idx], y_train.iloc[train_idx]
    val_data, val_labels = x_train[val_idx], y_train.iloc[val_idx]

    history = model.fit(train_data, train_labels, epochs=num_of_epochs, validation_data=(val_data, val_labels))
    histories.append(history)

    y_pred = model.predict(val_data)
    binary_predictions = (y_pred >= 0.5).astype(int)

    accuracies.append(accuracy_score(val_labels, binary_predictions))

print("AVERAGE ACCURACY: ", np.mean(accuracies))

# =========================================

# 7. Model Evaluation
# 7.1. run predictions on test data
# 7.2. return metrics ["accuracy", "precision", "recall", "f1 score", "roc_auc"]
# 7.3. confusion matrix
# 7.4. output graphs showing training acc and val acc per epoch
# 7.5. output graphs showing training loss and loss acc per epoch

# # 7.1. run predictions on test data
test_predictions = model.predict(x_test)
binary_test_predictions = (test_predictions >= 0.5).astype(int)

# # 7.2. return metrics ["accuracy", "precision", "recall", "f1 score", "roc_auc"]
test_accuracy = accuracy_score(y_test, binary_test_predictions)
test_precision = precision_score(y_test, binary_test_predictions)
test_recall = recall_score(y_test, binary_test_predictions)
test_f1 = f1_score(y_test, binary_test_predictions)
test_roc_auc = roc_auc_score(y_test, binary_test_predictions)

print("TEST ACCURACY: ", test_accuracy)
print("TEST PRECISION: ", test_precision)
print("TEST RECALL: ", test_recall)
print("TEST F1: ", test_f1)
print("TEST ROC_AUC: ", test_roc_auc)

# 7.3. confusion matrix
confusion_matrix = confusion_matrix(y_test, binary_test_predictions)
fx = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
fx.set_title('Confusion matrix\n\n')
fx.set_xlabel('\nValues model predicted')
fx.set_ylabel('True Values')
plt.show()

# # 7.4. output graphs showing training acc and val acc per epoch
history_accuracies = []
history_val_accuracies = []
history_losses = []
history_val_losses = []

for history in histories:
    history_accuracies.append(history.history['accuracy'])
    history_val_accuracies.append(history.history['val_accuracy'])
    history_losses.append(history.history['loss'])
    history_val_losses.append(history.history['val_loss'])

mean_accuracy = np.mean(history_accuracies, axis=0)
mean_val_accuracy = np.mean(history_val_accuracies, axis=0)
mean_loss = np.mean(history_losses, axis=0)
mean_val_loss = np.mean(history_val_losses, axis=0)

epochs = range(1, num_of_epochs + 1)

plt.plot(epochs, mean_accuracy, 'bo', label='Training acc')
plt.plot(epochs, mean_val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

# # 7.5. output graphs showing training loss and loss acc per epoch
plt.plot(epochs, mean_loss, 'bo', label='Training loss')
plt.plot(epochs, mean_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

sys.exit()
