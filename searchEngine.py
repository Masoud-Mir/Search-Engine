import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

document_1 = "I love watching movies when it's cold outside"
document_2 = "Toy Story is the best animation movie ever, I love it!"
document_3 = "Watching horror movies alone at night is really scary"
document_4 = "He loves to watch films filled with suspense and unexpected plot twists"
document_5 = "My mom loves to watch movies. My dad hates movie theaters. My brothers like any kind of movie. And I haven't watched a single movie since I got into college"
documents = [document_1, document_2, document_3, document_4, document_5]

# get user input
query = input('please type your query: ')

# tokenization
tokenized_documents = []
def custom_tokenize(doc, keep_punct = False, keep_alnum = False, keep_stop = False):
  
    token_list = word_tokenize(doc)

    if not keep_punct:
        token_list = [token for token in token_list if token not in string.punctuation]

    if not keep_alnum:
        token_list = [token for token in token_list if token.isalpha()]

    if not keep_stop:
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')
        token_list = [token for token in token_list if not token in stop_words]

    return token_list

for doc in documents:
    tokenized_documents.append(custom_tokenize(doc)) 

# stemming
lancaster_stemmer = LancasterStemmer()
snoball_stemmer = SnowballStemmer('english')

def stem_tokens(tokens, stemmer):
    token_list = []
    for token in tokens:
        token_list.append(stemmer.stem(token))
    return token_list

stemmed_documents = []
for token in tokenized_documents:
    stemmed_documents.append(stem_tokens(token, snoball_stemmer)) 

# calculate tf-idf
def fit_tfidf(docs):
    tf_vect = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    tf_vect.fit(docs)
    return tf_vect

tf_vect = fit_tfidf(stemmed_documents)
tf_mtx = tf_vect.transform(stemmed_documents)

ft = tf_vect.get_feature_names_out()

dataframe = pd.DataFrame(tf_mtx.toarray(), columns=ft)
# print(dataframe)

# vectorize query string
query_tokenized = custom_tokenize(query)
query_stemmed = stem_tokens(query_tokenized, snoball_stemmer)
query_tf_mtx = tf_vect.transform([query_stemmed])

query_dataframe = pd.DataFrame(query_tf_mtx.toarray(), columns=ft)
# print(query_dataframe)

# find most similar data
cosine_distance = cosine_similarity(query_tf_mtx, tf_mtx)

def most_similar(documents, cosine_distance):
    	
    similarity = 0
    index = 0
    for i, dist in enumerate(cosine_distance[0]):
        if dist > similarity:
            similarity = dist
            index = i

    return documents[index]

output = most_similar(documents, cosine_distance)
print(output)