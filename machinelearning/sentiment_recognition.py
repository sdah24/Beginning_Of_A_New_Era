import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open("C:\Users\Syed Alif Hasan\Desktop\sarcasm.json","r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


tokenizer = Tokenizer(oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)

widx = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded_seq = pad_sequences(sequences,padding='post')

