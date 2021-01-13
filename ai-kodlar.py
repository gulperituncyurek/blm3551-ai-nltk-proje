import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
train = pd.read_csv("train.csv", engine='python')
train.head()

train.tail()

train_original = train.copy()

%matplotlib inline
test = pd.read_csv("test.csv", engine='python')
test.head()

test.tail()

test_original = test.copy()

combine = train.append(test, ignore_index=True, sort=True)
combine.head()

combine.tail()

def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    
    for i in r:
        text = re.sub(i, "", text)
    
    return text

#Tweetlerdeki kullanıcı etiketklerini kaldırmak (@user gibi)
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")

combine.head()

#Noktalama İşaretlerini, Sayıları ve Özel Karakterleri Kaldırma
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

combine.head(10)

#Kısa Kelimeleri Kaldırmak (3 harf ve daha kısa kelimlerin kaldırılması)
#And, to, oh, hmm... gibi kelimelerin bir katkısı olmayacağı için kaldırıyoruz
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

combine.head(10)

#Tokenization, veri setimizdeki tüm temizlenmiş tweet'leri belirteceğiz.
tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())

tokenized_tweet.head()

#Ekleri kelime köklerinden ayırma
#son ekleri ("ing", "ly", "es", "s" vb.) ayırıyoruz
from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
tokenized_tweet.head()

#Tokenleri geri birleştirme
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combine['Tidy_Tweets'] = tokenized_tweet
combine.head()


#Metin verilerini içeren bir veri kümesini ön işlemeyi tamamladık.

#Bu adımdan sonra Veri Görselleştirme(Data Visualisation)'ye geçebiliriz.


from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import urllib
import requests

all_words_positive = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==0])

Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
image_colors = ImageColorGenerator(Mask)
wc = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_positive)

# Size of the image generated 
plt.figure(figsize=(10,20))
# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
plt.axis('off')
plt.show()

all_words_negative = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==1])

# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")

plt.axis('off')
plt.show()

#Understanding the impact of Hashtags on tweets sentiment
#Function to extract hashtags from tweets

def Hashtags_Extract(x):
    hashtags=[]
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags
    
ht_positive = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==0])
ht_positive

#Unnest the list
ht_positive_unnest = sum(ht_positive,[])

#A nested list of all the hashtags from the negative reviews from the dataset
ht_negative = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==1])
ht_negative

#Unnest the list
ht_negative_unnest = sum(ht_negative,[])

#Plotting Bar-plots
#Veri kümesindeki Olumlu Tweetler için Olumlu Duyguya sahip kelimelerin sıklığının sayılması
word_freq_positive = nltk.FreqDist(ht_positive_unnest)
word_freq_positive

#Hashtag'lerde en sık kullanılan kelimeler için bir veri çerçevesi (data frame) oluşturma
df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})
df_positive.head(10)

#Hashtag'ler için en sık kullanılan 20 kelime için barplotu çizme
df_positive_plot = df_positive.nlargest(20,columns='Count')
sns.barplot(data=df_positive_plot,y='Hashtags',x='Count')
sns.despine()


#Negatif Tweetler için Bar-plots
word_freq_negative = nltk.FreqDist(ht_negative_unnest)
word_freq_negative


df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})
df_negative.head(10)


df_negative_plot = df_negative.nlargest(20,columns='Count') 
sns.barplot(data=df_negative_plot,y='Hashtags',x='Count')
sns.despine()


