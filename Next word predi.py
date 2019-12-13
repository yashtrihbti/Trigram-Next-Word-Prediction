

from flask import Flask,request, render_template
from collections import Counter
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

text=brown.words()
V=sorted(set(text))


length=len(text)


fdist=nltk.FreqDist(text)

bigram = list(nltk.bigrams(text))
cfd = nltk.ConditionalFreqDist(bigram)


trigram=list(nltk.trigrams(text))

a=len(list(trigram))


for i in range(0,a):
    trigram[i]=((trigram[i][0:2]),trigram[i][2])

cfd2 = nltk.ConditionalFreqDist(trigram)


def uniprob(word):
    return (fdist[word]/length)


def biprob(w1,w2): #prob(w2/w1)
    lamda1=0.6
    lamda2=0.4
    prob=(lamda1*cfd[w1][w2]/fdist[w1])+(lamda2*uniprob(w2))
    return prob


def triprob(w1,w2,w3):
    lamda1=0.70
    lamda2=0.20
    lamda3=0.10
    prob=(lamda1*cfd2[w1,w2][w3]/cfd[w1][w2])+(lamda2*biprob(w2,w3))+(lamda3*uniprob(w3))
    return prob


def topfive(input1):
    length_input1=len(input1)
    Dict={}
    if length_input1==1:
        for v in V:
            Dict[v]=biprob(input1[0],v)
        Dict=sorted(Dict.items(), key = lambda kv:(kv[1], kv[0]),reverse= True)
     
        return Dict[0:5]
    
    else:
        a=input1[length_input1-2]
        b=input1[length_input1-1]
        for v in V:
            Dict[v]=triprob(a,b,v)
        Dict=sorted(Dict.items(), key = lambda kv:(kv[1], kv[0]),reverse= True)
        
        return Dict[0:5]


app = Flask(__name__)
@app.route('/predict', methods=['GET','POST'])
def submit():
 x={}
 text = ""
 out=[]
 if request.method == 'POST':
  text =request.form['text']
  out=topfive(word_tokenize(text))
  for i in range(0,5):
    out[i]=out[i][0]
  #print(x)
 return render_template('form.html',x=dict(out),text=text)

if __name__ == '__main__':
   app.run(host = 'localhost', port = 3010)






