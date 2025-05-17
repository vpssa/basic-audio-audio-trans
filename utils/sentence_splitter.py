import nltk

def split_into_sentences(text):
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        return nltk.sent_tokenize(text)