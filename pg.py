

import urllib.request
# import json
import simplejson as json
from bs4 import BeautifulSoup

import pandas

import requests
import wget

adress = "https://raw.githubusercontent.com/rekiksab/Yelp/master/yelp_challenge/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json"
bdress = "https://api.github.com/repositories/858127/issues?per_page=5"

url = urllib.request.urlopen(adress)
content = url.read()
print("1")
soup = BeautifulSoup(content, "html.parser")
newDictionary = json.loads(str(soup))

print("done")

urllib.request.urlretrieve()


response = urllib.request.urlopen(adress)
str_response = response.read().decode('utf-8')
obj = json.loads(str_response)

data = response.read()
print(data)
info = json.loads(data.decode('utf-8'))


# parse line by line
dat = []

with urllib.request.urlopen(adress) as f:
    str_f = f.read().decode('utf-8')
    for line in str_f:
        dat.append(json.loads(line))

# requests
r = requests.get(url = adress)
r.json()

df = []
with requests.get(url = bdress) as f:
    for line in f:
        df.append(f.json())

r = requests.get(url = adress)
foo = r.json()

conten = foo

fs = wget.download(url=adress)
fs.
with open(fs, 'r') as f:
    content = f.read()
print(content)


# pandas

foo = pandas.read_json(adress)




## read data downloaded from yelp.com
# https://www.yelp.com/dataset/download

path = "/home/jgucci/Desktop/uni/text_mining/tm_data/review.json"

with open(path, "r") as f:
    data = json.loads(f.deco    )

data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))

print(data[2]["text"])
list(data[1].keys())


############# check encoding functionality


############ function to decode text


import projectlib as pl
import numpy as np
pl.check()
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"
alphabet = open(alphabetPath).read()
lenAlpha = len(alphabet)
maxChars = 1014

def generate_one_hot(text, alphabet, maxChars):

    #  initialize empty ndarray with zeros only and depth 1
    textRep = np.zeros(shape=(len(alphabet), maxChars, 1))
    print(textRep.shape)

    # cut text to maxChars
    if len(text) > maxChars:
        text = text[0:maxChars]
        print("too long")


    # loop over all chars in the text
    for char_index in range(0, len(text)):
        if text[char_index] in alphabet:
            alpha_index = alphabet.find(text[char_index])
            print(alpha_index)
            # rows = alphabet features, columns = characters
            textRep[alpha_index][char_index][0] = 1
        # in case of whitespace or unknown characters include 0-vector
        # do nothing

    return(textRep)



encoded = generate_one_hot(text = "af_fe", alphabet = alphabet, maxChars=maxChars)

def decoder(onehotText, alphabet, maxChars = 1014):
    # initialize string
    onehotText = encoded
    a = str()
    for colIndex in range(0, maxChars):
        # only store alphabet index, if 1 is in column
        if np.isin(1, onehotText[:, colIndex, 0]):
            alphaIndex = np.where(onehotText[:, colIndex, 0] == 1)[0][0]
            print(colIndex, alphaIndex)
            a = a + alphabet[alphaIndex]
    return(a)

foo = decoder(onehotText=encoded, alphabet = alphabet, maxChars=maxChars)






