import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

#This is the class dealing with all preprocssing
#It includes key functions such as remove stopWords, lemmitazatino, bigrams, etc.
#This code is referenced and revised based on code from: Shantamvijayputra (2013) and Mbaye, M. (2020). 
class TextPreprocessor:
    def __init__(self):
		#Abbreviation dictionary
        self.abbreviations = {
            "$": "dollar",
            "€": "euro",
            "4ao": "for adults only",
            "a.m": "before midday",
            "a3": "anytime anywhere anyplace",
            "aamof": "as a matter of fact",
            "acct": "account",
            "adih": "another day in hell",
            "afaic": "as far as i am concerned",
            "afaict": "as far as i can tell",
            "afaik": "as far as i know",
            "afair": "as far as i remember",
            "afk": "away from keyboard",
            "app": "application",
            "approx": "approximately",
            "apps": "applications",
            "asap": "as soon as possible",
            "asl": "age, sex, location",
            "atk": "at the keyboard",
            "ave.": "avenue",
            "aymm": "are you my mother",
            "ayor": "at your own risk",
            "b&b": "bed and breakfast",
            "b+b": "bed and breakfast",
            "b.c": "before christ",
            "b2b": "business to business",
            "b2c": "business to customer",
            "b4": "before",
            "b4n": "bye for now",
            "b@u": "back at you",
            "bae": "before anyone else",
            "bak": "back at keyboard",
            "bbbg": "bye bye be good",
            "bbc": "british broadcasting corporation",
            "bbias": "be back in a second",
            "bbl": "be back later",
            "bbs": "be back soon",
            "be4": "before",
            "bfn": "bye for now",
            "blvd": "boulevard",
            "bout": "about",
            "brb": "be right back",
            "bros": "brothers",
            "brt": "be right there",
            "bsaaw": "big smile and a wink",
            "btw": "by the way",
            "bwl": "bursting with laughter",
            "c/o": "care of",
            "cet": "central european time",
            "cf": "compare",
            "cia": "central intelligence agency",
            "csl": "can not stop laughing",
            "cu": "see you",
            "cul8r": "see you later",
            "cv": "curriculum vitae",
            "cwot": "complete waste of time",
            "cya": "see you",
            "cyt": "see you tomorrow",
            "dae": "does anyone else",
            "dbmib": "do not bother me i am busy",
            "diy": "do it yourself",
            "dm": "direct message",
            "dwh": "during work hours",
            "e123": "easy as one two three",
            "eet": "eastern european time",
            "eg": "example",
            "embm": "early morning business meeting",
            "encl": "enclosed",
            "encl.": "enclosed",
            "etc": "and so on",
            "faq": "frequently asked questions",
            "fawc": "for anyone who cares",
            "fb": "facebook",
            "fc": "fingers crossed",
            "fig": "figure",
            "fimh": "forever in my heart",
            "ft.": "feet",
            "ft": "featuring",
            "ftl": "for the loss",
            "ftw": "for the win",
            "fwiw": "for what it is worth",
            "fyi": "for your information",
            "g9": "genius",
            "gahoy": "get a hold of yourself",
            "gal": "get a life",
            "gcse": "general certificate of secondary education",
            "gfn": "gone for now",
            "gg": "good game",
            "gl": "good luck",
            "glhf": "good luck have fun",
            "gmt": "greenwich mean time",
            "gmta": "great minds think alike",
            "gn": "good night",
            "g.o.a.t": "greatest of all time",
            "goat": "greatest of all time"
        }
	#Removel url
    def remove_URL(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'URL', str(text))

	#Remove html
    def remove_HTML(self, text):
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub(r'', text)

    #Find word abbrevation
    def word_abbrev(self, word):
        return self.abbreviations[word.lower()] if word.lower() in self.abbreviations.keys() else word

	#Replace word abbrevation with original one
    def replace_abbrev(self, text):
        string = ""
        for word in text.split():
            string += self.word_abbrev(word) + " "
        return string

	#Remove punctuations
    def remove_all_punct(self, text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

	#Remove stopwords
    def remove_stopwords(self, text):
        stop_words = set(stopwords.words("english"))
        return ' '.join(word for word in text.split() if word.lower() not in stop_words)

	#Remove stemming words
    def stemming(self, text):
        ps = PorterStemmer()
        return ' '.join(ps.stem(word) for word in text.split())

	#remove lemmatization
    def lemmatization(self, text):
        lm = WordNetLemmatizer()
        return ' '.join([lm.lemmatize(word, pos='v') for word in text.split()])

	#Main preprocssing procedure: use argument input to control which preprocessing function to implement
    def clean_text(self, text, remove_all_punct=False,remove_url=False, remove_html=False,replace_abbrev=False,
               lemmatization=False, stemming=False, remove_stopwords=False, lower_case=False):
        if remove_url:
            text = self.remove_URL(text)
        if remove_html:
            text = self.remove_HTML(text)
        if replace_abbrev:
            text = self.replace_abbrev(text)
        if remove_all_punct:
            text = self.remove_all_punct(text)
        if lemmatization:
            text = self.lemmatization(text)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        if stemming:
            text = self.stemming(text)
        if lower_case:
            text = text.lower()

        list = text.split(' ')
        return list
	
    def generate_bigrams(self, text):
        # Ensure that text is a string
        if isinstance(text, list):
            text = ' '.join(text)

        bigrams = ngrams(text.split(), 2)
        return [' '.join(bigram) for bigram in bigrams]
	
    def clean_text_with_bigrams(self, text, **kwargs):
        cleaned_text = self.clean_text(text, **kwargs)
        bigrams = self.generate_bigrams(cleaned_text)
        return bigrams


