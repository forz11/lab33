from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
import re

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

nltk.download('stopwords')
nltk.download('punkt')
# Инициализация Spark
conf = SparkConf().setAppName("TextAnalysis")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Загрузка текста из файла
file_path = r"Giperboloid_injenera_Garina.txt"
text_rdd = sc.textFile(file_path)

# 1. Text Cleaning
# Удаление знаков препинания и преобразование в нижний регистр
def clean_text(text):
    # Удаление цифр и знаков препинания
    text_no_punctuation = re.sub(r'[^A-Za-zА-Яа-я\s]', '', text)
    
    # Преобразование в нижний регистр
    return text_no_punctuation.lower()

cleaned_rdd = text_rdd.map(clean_text)

# Удаление стоп-слов
stop_words = nltk.corpus.stopwords.words("russian")
stop_words_broadcast = sc.broadcast(stop_words)

def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words_broadcast.value]
    return " ".join(filtered_words)

filtered_rdd = cleaned_rdd.map(remove_stopwords)

# 2. WordCount
word_count_rdd = filtered_rdd.flatMap(lambda line: line.split()).countByValue()

# 3. Print Top50 most and least common words
sorted_word_count = sorted(word_count_rdd.items(), key=lambda x: x[1], reverse=True)
top50_most_common = sorted_word_count[:50]
top50_least_common = sorted_word_count[-50:]

print("Top 50 most common words:")
for word, count in top50_most_common:
    print(f"{word}: {count}")

print("\nTop 50 least common words:")
for word, count in top50_least_common:
    print(f"{word}: {count}")

# 4. Stemming
stemmer = SnowballStemmer("russian")

def stem_text(text):
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

stemmed_rdd = filtered_rdd.map(stem_text)

# 5. WordCount after Stemming
stemmed_word_count_rdd = stemmed_rdd.flatMap(lambda line: line.split()).countByValue()

# 6. Print Top50 most and least common words after Stemming
sorted_stemmed_word_count = sorted(stemmed_word_count_rdd.items(), key=lambda x: x[1], reverse=True)
top50_most_common_stemmed = sorted_stemmed_word_count[:50]
top50_least_common_stemmed = sorted_stemmed_word_count[-50:]

print("\nTop 50 most common words after stemming:")
for word, count in top50_most_common_stemmed:
    print(f"{word}: {count}")

print("\nTop 50 least common words after stemming:")
for word, count in top50_least_common_stemmed:
    print(f"{word}: {count}")

# Завершение Spark
sc.stop()