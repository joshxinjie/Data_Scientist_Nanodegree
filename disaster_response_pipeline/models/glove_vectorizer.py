import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

class GloveVectorizer(BaseEstimator, TransformerMixin):
    """
    Fit and transform text with pretrained Glove Embeddings. It will take 
    the simple average of word vectors to get the sentence vector. Vectors of
    100 dimensions will be used.    
    """
    
    def __init__(self):
        word2vec = {}
        embedding = []
        idx2word = []
        
        # get pretrained glove vectors
        with open('glove.6B/glove.6B.100d.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                # get word
                word = values[0]
                # get glove vector for word
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
                
        self.word2vec = word2vec
        self.embedding = embedding
        self.idx2word = idx2word
        
        # Get number of vocabulary and dimensions for word vector
        self.vocab_size, self.dim = len(embedding), len(embedding[0])
                
    def tokenize(self, text):
        """
        Clean and tokenize text for modeling. It will replace all non-
        numbers and non-alphabets with a blank space. Next, it will
        split the sentence into word tokens and remove all stopwords.
        The word tokens will then be lemmatized with Nltk's 
        WordNetLemmatizer(), first using noun as part of speech, then verb.
        
        INPUTS:
            text - the string representing the message
        RETURNs:
            clean_tokens - a list containing the cleaned word tokens of the
                            message
        """
        # replace all non-alphabets and non-numbers with blank space
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

        # Tokenize words
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words("english")]

        # instantiate lemmatizer
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            # lemmtize token using noun as part of speech
            clean_tok = lemmatizer.lemmatize(tok)
            # lemmtize token using verb as part of speech
            clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
            # strip whitespace and append clean token to array
            clean_tokens.append(clean_tok.strip())
         
        return clean_tokens
        
    
    def fit(self, x, y=None):
        pass
    
    def transform(self, X):
        """
        Transform all messages into sentence vectors using pretrained Glove
        Embeddings. It will take the simple average of word vectors to get
        the sentence vector.
        
        INPUTS:
            X - the messages to be transform
        RETURNS:
            new_X - the transformed messages
        """
        new_X = np.zeros((len(X), self.dim))
        
        # keep track of sentences without any glove vectors representation
        self.emptycount = 0
        
        n=0
        
        for message in X:
            clean_tokens = self.tokenize(message)
            vecs = []
            for word in clean_tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                # Get mean of all glove vectors of each message
                new_X[n] = vecs.mean(axis=0)
            else:
                self.emptycount += 1
            n += 1
        new_X = pd.DataFrame(new_X)
        return new_X
    
    def fit_transform(self, X, y=None):
        """
        Fits and transforms the messages.
        """
        self.fit(X)
        return self.transform(X)