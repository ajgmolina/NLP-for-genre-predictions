{\rtf1\ansi\ansicpg1252\cocoartf2512
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 ![](RackMultipart20210305-4-xeyfa2_html_38ce1c067ab9b6c6.png)\
\
Andr\'e9s Garc\'eda Molina, PhD\
\
March 4, 2021\
\
**Using NLP to Predict Song Genres through their Lyrics**\
\
1. **Introduction**\
\
Natural Language Processing (NLP) techniques allow making sense of natural language, one of the most challenging types of data in machine learning given the unbounded, unstructured nature of language. In an era of digital circulation and streaming, music metadata is crucial in the process of accurately tagging and classifying music to aid its discoverability. How might we leverage NLP techniques in order to correctly assign genre labels to songs? This project combines NLP techniques with neural networks in order to predict genre labels on a corpus of song lyrics.\
\
A [2019 report](https://www.musicbusinessworldwide.com/nearly-40000-tracks-are-now-being-added-to-spotify-every-single-day/) published by Music Business Worldwide asserts that nearly 40,000 tracks are uploaded to Spotify every day. Given such a volume, manual classification is an expensive and slow task that is also error-prone. While in theory those uploading bear the burden of providing meaningful and accurate metadata, automating a process for sanity checks is well worth the investment. This notebook explores some ways to implement a genre-tagging model that is based on neural network training of labeled lyrics datasets.\
\
Stakeholders: Who might be interested in this report?\
\
A wide range of audiences are invested in understanding how to automate genre label tagging, including:\
\
- Artists who upload music to various online platforms and hope to maximize their searchability and findability\
- The platforms themselves, who hope to optimize music discovery and recommendation processes\
- Audiences who want to find music according to genre labels and also discover new acts that might be of interest\
\
Methods\
\
Several neural network machine learning models are trained and tested on a song lyrics dataset that has undergone preparation based on NLP techniques.\
\
1. **Data Wrangling**\
\
Perhaps the biggest challenge in this project concerns acquiring good, labeled data that pairs song lyrics with genres. For the task at hand, I analyzed several already-available datasets that have performed similar work, including that of[Bajwa et al](https://github.com/etarakci/music-genre-prediction),[Sianipar et al](https://medium.com/better-programming/predicting-a-songs-genre-using-natural-language-processing-7b354ed5bd80),[Kovachev et al](https://towardsdatascience.com/how-we-used-nltk-and-nlp-to-predict-a-songs-genre-from-its-lyrics-54e338ded537), and[Ram and Salz](http://cs229.stanford.edu/proj2017/final-reports/5241796.pdf). As an initial step, the following datasets were imported and examined:\
\
1. &quot;genre\\_lyrics\\_data.csv&quot;, by Bajwa et al, available on[GitHub](https://github.com/etarakci/music-genre-prediction/tree/master/data), including 6,733 lyrics and 90 genres.\
\
2. &quot;tcc\\_ceds\\_music.csv&quot;, by Moura et al, available via[Mendeley Data](https://data.mendeley.com/datasets/3t9vbwxgr5/3), including 28,372 lyrics and 7 genres.\
\
3. &quot;spotify\\_songs.csv&quot;, by Muhammad Nakhaee and available via[Kaggle](https://www.kaggle.com/imuhammad/audio-features-and-lyrics-of-spotify-songs?select=spotify_songs.csv), including 18,454 lyrics and 6 genres.\
\
4. &quot;original\\_cleaned\\_lyrics.csv&quot;, by Yalamanchili et al, available via[GitHub](https://github.com/hiteshyalamanchili/SongGenreClassification/tree/master/dataset), including 227,449 lyrics and 11 genres. This is a processed version of the no-longer available &quot;380,000+ lyrics from MetroLyrics&quot; dataset (likely removed due to copyright infringement) that served as the starting point for many related projects.\
\
Ultimately, #2, #3 and #4 are selected as the source datasets for this project, given their more restricted range in genres. The first dataset was discarded, given its small number of observations in relation to the vast spread of genres contained within.\
\
The different datasets are joined and only essential features are kept: the lyrics themselves and the corresponding genre label. Some other steps taken include:\
\
1. Dropping missing values: In this case, they refer to no lyrics available or no genre label available. In this particular business case, these values cannot be imputed in a reasonable way.\
2. Genre labels are standardized: For example, &quot;rock&quot;, &quot;rock &#39;n&#39; roll&quot;, &quot;rock and roll&quot; all become &quot;Rock.&quot;\
3. Duplicates are handled: Only one instance is kept for full duplicates (i.e. both lyrics and genre match) and lyric duplicates with different genres (e.g. a song by The Beatles labeled as both pop and rock).\
4. Removing other tokens: These include punctuation, typical lyrics identifiers such as &quot;[Chorus]&quot; or &quot;[Verse]&quot;, lyrics comprised of the word &quot;Instrumental&quot; as the only lyrical content, and corrupted/non-ASCII characters.\
5. Detecting lyrics languages and only selecting English lyrics.\
\
A decision is made early-on: Our dataset contains over 250,000 lyrics, but the classes are imbalanced. Rock comprises roughly 50% of the genres, and some genres (e.g. folk and reggae) represent less than 1% of the dataset. We thus focus on the top five genres\'97Rock, Pop, Hip-hop, Metal, and Country\'97and we use the minority class&#39;s size (n\\_Country = 18,580).\
\
We then tokenize lyrics in order to:\
\
1. Remove stopwords, and\
2. Lemmatize, which in turn requires applying part of speech tags.\
\
More details on these preprocessing steps can be found in the [Data Wrangling notebook](https://github.com/ajgmolina/NLP-for-genre-predictions/blob/master/1-Data%20Wrangling.ipynb) of the project.\
\
1. **Exploratory Data Analysis (EDA) and Feature Engineering**\
\
EDA was performed in order to get a general sense of the characteristics of the lyrics dataset. Initial exploration revealed that the process of lemmatizing had led to some instances of empty sets, as well as duplicates. These cases were addressed as an initial step.\
\
Two sentiment analysis toolkits are applied: Vader and TextBlob. As the images below demonstrates, the analyzer performances are remarkably different. Overall, Vader is quicker to polarize while TextBlob&#39;s output approaches a normal distribution:\
\
![](RackMultipart20210305-4-xeyfa2_html_8b197b1e1e290d2f.png) ![](RackMultipart20210305-4-xeyfa2_html_5215c629aa85feda.png)\
\
_Figure 1: TextBlob (left) and Vader (right) sentiment distributions._\
\
The figure below summarizes these differences per genre:\
\
![](RackMultipart20210305-4-xeyfa2_html_d02e5712c3608b0e.png)\
\
_Figure 2: TextBlob and Vader differences in sentiment analysis._\
\
We also explore how verbose each genre is, or put differently, how many unique words are employed in each of the genre labels:\
\
![](RackMultipart20210305-4-xeyfa2_html_80586375b77d3ca9.png)\
\
_Figure 3: Average unique words per genre. Unsurprisingly, hip-hop is a verbose outlier._\
\
Our EDA exploration also maps the 25 most frequent words per genre. Full details can be seen in the [EDA and Feature Engineering notebook of the project.](https://github.com/ajgmolina/NLP-for-genre-predictions/blob/master/2-Exploratory%20Data%20Analysis.ipynb)\
\
The following methods are employed in this section:\
\
1. LDA (Latent Dirichlet Allocation)\
2. NMF (Non-Negative Matrix Factorization)\
\
Both approaches yield different topic modeling, and below we explore the differences in their outputs. While there is significant overlap in the topics yielded, NMF is favored because the topics presented cover a broader range and are more distinct. In a problem like the one at hand, a more fine-tuned approach might be appropriate if topic modeling were the objective of the project.\
\
For instance, modeling could be genre specific in addition to global. In order to perform LDA modeling, the lyrics are first converted into a document term matrix using a count vectorizer. NMF modeling requires an input in the form of a TFIDF, or term frequency\'96inverse document frequency matrix, which reflects how important each word is in the entire corpus.\
\
It is important to note that each model yields different topics, and that the way they are presented are through word lists of settable predetermined length that require some subjective interpretation.\
\
LDA Topics\
\
Topic 0: Life, love, longing.\
 Topic 1: Money and hustling.\
 Topic 2: Love and sexuality.\
 Topic 3: Love and fun.\
 Topic 4: Affection, women.\
 Topic 5: Death, faith.\
 Topic 6: Dance.\
\
NMF Topics\
\
Topic 0: Life as a journey.\
 Topic 1: Money and hustling.\
 Topic 2: Home and longing.\
 Topic 3: Love and life.\
 Topic 4: Love and fun.\
 Topic 5: Love and sexuality.\
 Topic 6: Introspection and time\
\
The figure below presents topic distributions for the entire sample.\
\
![](RackMultipart20210305-4-xeyfa2_html_c4ae1d6cd4fbf1b7.png)\
\
_Figure 4: Global topic distributions. &quot;Home and longing&quot; is a clear dominator as a common theme across genres._\
\
Our EDA exploration also maps how topics are distributed across specific genres. Full details can be seen in the [EDA and Feature Engineering notebook of the project.](https://github.com/ajgmolina/NLP-for-genre-predictions/blob/master/2-Exploratory%20Data%20Analysis.ipynb)\
\
**IV. Modeling**\
\
Prior to training and testing models, we use a label encoder for our genres target variable. We also follow the conventional splitting of our data into training and test sets, and for the purposes of training a neural network we use a &quot;bag of words&quot; (BOW) representation of our lyrics, which is a vectorized count transformation. We also normalize the BOW representation so that all values in our matrix are between 0 and 1; scaled data allows neural networks to achieve better results.\
\
As a baseline model we employ a logistic regression model fit on the normalized BOW training data, achieving 62% accuracy.\
\
We start with a basic keras Sequential model with one 100-neuron layer that achieves 77% training accuracy and 63% testing accuracy after 60 epochs. The results below suggest that validation accuracy stops improving after around 20 epochs, and validation loss stagnates roughly around the same time.\
\
![](RackMultipart20210305-4-xeyfa2_html_b3a434b1851fe7a3.png)\
\
_Figure 5: Training and validation accuracy for a neural network with an inner layer of 100 neurons after 60 epochs._\
\
The model is re-run with a smaller epoch number and a smaller batch size. With the modifications made, the training accuracy improves, but the validation accuracy diminishes, suggesting overfitting. One final experiment is made, setting the number of nodes in the hidden layer to 200, achieving a training accuracy of 65% and a testing accuracy of 63%.\
\
![](RackMultipart20210305-4-xeyfa2_html_4f5482dc28fafd7c.png)_Figure 6: A simple Sequential model with a single 200-node layer after 7 epochs._\
\
We employ several other neural networks and their associated techniques in order to attempt higher accuracy scores in our validation data. As a general trend, we notice that it is indeed possible to obtain near-perfect accuracy in training, which in no way correlates to better accuracy in validation.\
\
We begin using word embeddings in order to train a neural network with an embedding layer that maps our BOW representation to dense vectors, thus reducing dimensionality. Details on the required pre-processing steps can be reviewed in the [Modeling notebook](https://github.com/ajgmolina/NLP-for-genre-predictions/blob/master/3-Feature%20Engineering%20and%20Modeling.ipynb) of the project.\
\
![](RackMultipart20210305-4-xeyfa2_html_3715b2da2030a531.png)\
\
_Figure 7: Structure of the sequential model with embedding._\
\
A verbose summary of the model as it trains alerts us to a general trend: epochs increase training accuracy but validation accuracy peaks at 62.27% around 25 epochs and then decays. Global max pooling is also employed as a technique to reduce dimensionality:\
\
![](RackMultipart20210305-4-xeyfa2_html_e0f59d4541fbebf8.png)\
\
_Figure 8: Global max pooling after embedding._\
\
This is the best performing model, which achieves 64% testing accuracy. The general trend, however, remains constant, where testing accuracy stagnates quickly.\
\
![](RackMultipart20210305-4-xeyfa2_html_8295f27267f20dc6.png)\
\
_Figure 9: Embedding + global max pooling performance after 15 epochs._\
\
The [Modeling notebook](https://github.com/ajgmolina/NLP-for-genre-predictions/blob/master/3-Feature%20Engineering%20and%20Modeling.ipynb) of the project details behavior across epochs and also using other approaches. A common approach is to use pre-existing, pre-trained word embeddings rather than creating our own based on our local corpus. The embeddings used by [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) (Global Vectors for Word Representation)--developed by the Stanford NLP Group--are employed. Specifically, the word embeddings from the Wikipedia 2014 + Gigaword 5 6 billion token model is used (the size of the vocabulary is 400k). The percentage of our vocabulary that is covered by the pretrained model is slightly over 50%. The results obtained stagnate around 48%; with such a low representation of our vocabulary (50%), a pre-trained word embeddings model cannot realistically obtain an accuracy higher than the percentage of representation.\
\
We then test if anything changes once we allow the embedding to be trainable, by setting the hyperparameter &quot;trainable&quot; to True. The trainable embedding and max pooling model appears to stagnate around 30 epochs, around a 62% of accuracy.\
\
Finally, we also apply Convolutional Neural Networks (CNN) as a way to generate and explore hidden patterns in our dataset. More specifically, a convolutional layer is added between embedding and pooling.\
\
![](RackMultipart20210305-4-xeyfa2_html_f2434dfd90f62c3b.png)\
\
![](RackMultipart20210305-4-xeyfa2_html_1e6ad7a43ceceda8.png)\
\
_Figure 10: A convolutional neural network and its performance after 30 epochs._\
\
As with other models, convolutional networks quickly start overfitting, and testing accuracy stagnates in a similar range as do all other models. We run two final sanity checks with simpler models. A Random Forest Classifier and a Multinomial Naive Bayes model confirm the general ceiling of this dataset: the random forest&#39;s testing accuracy clocks in at 62% while the NB model reaches 61% accuracy.\
\
**Conclusions and Recommendations**\
\
The results above show that a Random Forest Classifier trained on a Bag of Words representation, as with Logistic Regression, appear to produce similar results as neural networks. If this application were to be deployed, neural networks would not be favored as an appropriate solution, as the yielded results do not improve significantly on simpler models, but are much more computationally expensive.\
\
The next steps would require further testing with hyperparameter tuning using simple models and one of the less computationally expensive neural networks. Additional neural networks could be experimented with, such as a Long Short-Term Memory (LSTM) neural network, Recurrent Neural Networks (RNN), Gated Recurrent Unit (GRU), Hierarchical Attention Networks, Recurrent Convolutional Neural Networks (RCNN), Random Multimodel Deep Learning (RMDL), and Hierarchical Deep Learning for Text (HDLTex).\
\
Ultimately, the question might lie in the quality of the data, as the acquired datasets come from multiple sources and data quality is certainly an issue. This project ultimately demonstrates that one of the more crucial steps in the data science lifecycle concerns data acquisition and quality. In this case, generating a tidy lyrics dataset would produce high pre-processing costs.\
\
4\
}