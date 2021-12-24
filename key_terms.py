"""
https://hyperskill.org/projects/166/stages/863/implement#comment
https://imgur.com/a/ZwHUfDQ
"""
from lxml import etree
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string


class KeyTermExtraction:
    def __init__(self):
        self.xml_path = 'news.xml'
        self.stop_puncs = stopwords.words("english") + list(string.punctuation)
        self.tokens = []
        self.heads = []
        self.process_xml()

    def process_xml(self):
        text = ''
        head = ''
        tree = etree.parse(self.xml_path)
        lemmatizer = WordNetLemmatizer()
        for element in tree.iter('value'):  # to find tags with value.
            if element.get('name') == 'head':  # finding "name" attribute with value "head".
                self.heads.append(element.text)  # created headers list
            if element.get('name') == 'text':
                text = word_tokenize(element.text.lower())
                text = [lemmatizer.lemmatize(i, pos="n") for i in text]
                text = [i for i in text if i not in self.stop_puncs]
                text = [i for i in text if pos_tag([i])[0][1] == 'NN']
                value = ' '.join(text)
                self.tokens.append(value)  # created story list

    def tfidf_counter(self):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.tokens)  # created vector for news in all news
        terms = vectorizer.get_feature_names()  # created terms list
        document, word = tfidf_matrix.shape  # found document and terms length
        for d in range(document):
            word_score = {}
            for w in range(word):
                word_score.update({terms[w]: tfidf_matrix[(d, w)]})  # for each news in document created word score
            result = sorted(word_score.items(), key=lambda x: (x[1], x[0]), reverse=True)
            print(f"{self.heads[d]}:")
            print(*[k for k, v in result[:5]])


KeyTermExtraction().tfidf_counter()
