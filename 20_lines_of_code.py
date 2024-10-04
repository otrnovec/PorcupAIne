import matplotlib.pyplot as plt
import requests
from conllu import parse

def clean_text(text):
    punctuation = {',':'', '.':'', '!':'', '?':''}
    # too complicated, kick out too many: punctuation = {'‚': "", ",": "", "?": "", "!": "", ".": "", "(": "", ")": "", "-": "", ";": "", '“': "", '„': "", '‘':"", "–": "", ":": ""}  # nahradit určitý znaky v celým textu
    table = text.maketrans(punctuation)
    text_without_punctuation = text.translate(table)
    text = text_without_punctuation.lower()
    #text = text.split("\n")        # dávat si pozor, jestli je zapnuté nebo ne!
    return text

#spočítá průměrnou délku tokenu v daném textu
def average_token_length(text):
    text = text.split()
    all_length = 0
    for token in text:
        all_length += len(token)
    return round(all_length/len(text), 2)