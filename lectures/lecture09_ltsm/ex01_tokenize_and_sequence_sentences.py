from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

sentence1 = "Shagun Shagun is prepared to succeed."
sentence2 = "Shagun Shagun sees opportunity in every challenge."
sentence3 = "Shagun Shagun is enjoying machine learning."
sentences = [sentence1, sentence2, sentence3]

# Restrict tokenizer to use top 2500 words.
tokenizer = Tokenizer(num_words=2500, lower=True,split=' ')
tokenizer.fit_on_texts(sentences)

# Convert to sequence of integers.
X = tokenizer.texts_to_sequences(sentences)
print(X)

# Showing padded sentences:
paddedX = pad_sequences(X)
print(paddedX)
