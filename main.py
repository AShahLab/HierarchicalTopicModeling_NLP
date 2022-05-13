#Libraries
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary
import tomotopy as tp
import re
import spacy
import helperfunctions as hp
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_md")  # needs to be run in the terminal
#update the stopwords list with the spacy stopwords
en_stop.add("said")
en_stop.add("reuters")
en_stop.add("london")
en_stop.add("new york")
en_stop.add('reuters')
en_stop.add('say')
en_stop.add('like')
en_stop.add('thing')
en_stop.add('york')
en_stop.add('new')

model = LdaModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/lda/test-lda')#Gensim LDA model
mdl0 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda0.tmm')#TomoPy HLDAModel
mdl1 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda1.tmm')#TomoPy HLDAModel
mdl2 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda2.tmm')#TomoPy HLDAModel
mdl3 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda3.tmm')#TomoPy HLDAModel

cluster_names = {  # name the clusters as seems reasonable based on the LDA model outputs and the HLDA model outputs
    0: "Science and Technology",
    1: "Business/Tech/BioTech",
    2: "Health/Science/Drugs",
    3: "Entertainment/News"
}
# Test articles
# other_texts = {'txt':['The human genome is made up of about 3.1 billion DNA subunits, pairs of chemical bases known by the letters A, C, G and T. Genes are strings of these lettered pairs that contain instructions for making proteins, the building blocks of life. Humans have about 30,000 genes, organized in 23 groups called chromosomes that are found in the nucleus of every cell']}
other_texts = {'txt': [
    'Oliveira missed championship weight by half a pound at the official weigh-ins, using up almost the entirety of the initial two-hour window before making his first attempt and then taking another hour before stepping to the scale again. Both times he came in at 155.5 pounds. His UFC 274 main event challenger Justin Gaethje showed up minutes after the weigh-ins began, hitting 155 on the dot. It was later announced that Oliveira would be stripped of his title and that only Gaethje would be eligible to leave Footprint Center in Phoenix with the belt around his waist.']}
# other_texts = {'txt':['The Centre for Creative and Immersive XR (extended reality - an umbrella term for all types of digital reality, from the immersive VR viewed in headsets, to AR games like Pokemon Go, in which graphics are posted over the real world as seen through a smartphone camera) has received more than £5m in funding, including a £3.6m government grant']}
# other_texts = {'txt':['Rooftop solar has a “huge potential” to cut air pollution, create jobs, protect against outages and shrink utility bills, said Mark Jacobson, a professor of civil and environmental engineering at Stanford University. The technology will be vital for the U.S. to make the transition to all-renewable energy by 2050, he said.']}
# other_texts = {'txt':['The Kings finalized a deal with Brown on Sunday after meeting with him over two days late last week, sources said. Brown comes to the Kings with a clear organizational mandate: End the longest playoff drought in NBA history and return the Kings to the postseason for the first time in 16 years.']}
# other_texts = {'txt':['Patients with KRAS-mutant colorectal cancers do not respond to cetuximab, a monoclonal antibody against EGFR. A new proof-of-concept study presents a bispecific antibody with the ability to trigger EGFR degradation in LGR5+ cancer stem cells, and robust anti-tumor activity in KRAS-mutant and wild-type colorectal cancers.']}
# other_texts = {'txt':['Bitcoin continued to slide after a broader stock sell-off in the U.S. last week sent the cryptocurrency market into a frenzy and prompted bitcoin to plummet by roughly 10%. Bitcoin, the world’s largest digital currency by market value, was lower by about 3% at $33,438.03 late Sunday, according to data from Coin Metrics. This year, Bitcoin has been trading in a narrow range as it attempts to reclaim its highs of late 2021. The cryptocurrency is now down 50% from its peak price of $67,802.30 in November 2021. The drop comes after the blue-chip Dow Jones Industrial Average lost more than 1,000 points on Thursday and the Nasdaq plunged by 5%. Those losses marked the worst single-day drops since 2020. The Dow and Nasdaq fell again on Friday. \
# # \ Meanwhile, the Federal Reserve on Wednesday raised its benchmark interest rate by half a percentage point as it responds to inflation pressures.\
# # \ The stock market rallied after Fed chair Jerome Powell said a larger rate hike of 75 basis points isn’t being considered. But by Thursday, investors had erased the Fed rally’s gains.\
# # \ The global cryptocurrency market cap was at $1.68 trillion on Sunday, according to data from CoinGecko.com, and cryptocurrency trading volume in the last day was at $119 billion.']}
# --------------
other_texts = pd.DataFrame(other_texts)  # Convert dictionary to dataframe

nltk.download('wordnet')# Download wordnet
lemmatizer = WordNetLemmatizer()# Initialize lemmatizer

#------------------------set up test data for hlda---------------------------------
pat = re.compile('\w+')# Initialize pattern checker for tokenization
corpus_unseen = tp.utils.Corpus(# Initialize corpus for unseen texts
    tokenizer=tp.utils.SimpleTokenizer(stemmer=None, lowercase=True),
    stopwords=lambda x: len(x) <= 2 or x in en_stop or x.isnumeric() or not pat.match(x) or not lemmatizer.lemmatize(x)# Initialize tokenizer with lemmatizer and stopwords and pattern checker and length checker
)
unseen = other_texts['txt']# Add unseen texts to corpus
corpus_unseen.process(d.lower() for d in unseen)# Process unseen texts
#----------------------------------------------------------------------------------
#------------------------set up test data for gensim lda---------------------------------
dict_gensim, corpus_gensim = hp.preprocess_text(nlp(other_texts['txt'][0].lower()), 1,
                                                False)  # Preprocess the dataframe
unseen_doc = corpus_gensim[0]
#----------------------------------------------------------------------------------
#Inference Ideas

#Idea 1: Run LDA on unseen texts and see if it can predict the topic of the unseen text and then use that to run hierarchical LDA on the unseen text
vector = model[unseen_doc]  # get topic probability distribution for a document
topic_number, proba = sorted(vector, key=lambda item: item[1])[-1]# get the most probable topic
topic_number2, proba2 = sorted(vector, key=lambda item: item[1])[-2]  # thinking of using the second topic only, since first doesnt change--decide later


if proba > 0.49:  # increased to 0.49 for selecting first topic so that we see it only if the model is atleast 50% positive, else not worth it
    tn = topic_number
else:
    tn = topic_number2
if proba < 0.2:
    print(-1, -1)
else:
    print(cluster_names.get(topic_number), proba)
    print(cluster_names.get(topic_number2), proba2)
hldamdl = tp.HLDAModel()
if tn == 0:
    hldamdl = mdl0
elif tn == 1:
    hldamdl = mdl1
elif tn == 2:
    hldamdl = mdl2
elif tn == 3:
    hldamdl = mdl3
cps,ll=hldamdl.infer(corpus_unseen)# get the inferred topics and log likelihood for the unseen text using the hierarchical LDA model
for doc in cps:#loop through the inferred topics
    for path in doc.path:#loop through each level of the the path provided e.g. [0,8,120], where 0 is always the root topic
        if path==0:
            print('Root Topic is {}'.format(cluster_names.get(tn)))#print the root topic as manually selected in the cluster_names dictionary
            print('Subtopics Level {}:\n{}'.format(path,[i[0] for i in hldamdl.get_topic_words(path)]))#print the subtopics provided by the model for root topic
        else:
            print('Subtopics Level {}:\n{}'.format(path,[i[0] for i in hldamdl.get_topic_words(path)]))#print the subtopics for rest of the levels
    print('Original Unseen Word|Probability pairs:\n{}'.format(doc.get_words(top_n=10)))#print the top 10 words from the unseen text and their probabilities

#Idea 2: Run all the HLDA models and get the log likelihood for each model and then compare them and see which one is the best
mdls=[mdl0,mdl1,mdl2,mdl3]#list of all the models
lst=[]
for i in mdls:#loop through all the models
  cps2, ll2=i.infer(corpus_unseen)#get the inferred topics and log likelihood for the unseen text using the hierarchical LDA model
  lst.append((cps2,abs(ll2)))#append the inferred topics and log likelihood to the list
maxll=max(lst,key=lambda item:item[1])[1]#get the maximum log likelihood
maxcps=max(lst,key=lambda item:item[1])[0]#get the inferred topics for the unseen text using the hierarchical LDA model
max_index = lst.index((maxcps,maxll))#get the index of the model with the maximum log likelihood
md=mdls[max_index]#get the model with the maximum log likelihood using the max_index

for doc in maxcps:
    for path in doc.path:
      if path==0:
        print('Root Topic is {}'.format(cluster_names.get(max_index)))
        print('Subtopics Level {}:\n{}'.format(path,[i[0] for i in md.get_topic_words(path)]))
      else:
        print('Subtopics Level {}:\n{}'.format(path,[i[0] for i in md.get_topic_words(path)]))
    print('Original Unseen Word|Probability pairs:\n{}'.format(doc.get_words(top_n=10)))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Formattiing the data for the visualization
#Tree structure with one random topic selected from level 1
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('Printing the tree structure for hlda model with one random topic selected from level 1')
print('Root Topic 0:   \t\t{}'.format(cluster_names.get(max_index)))
for doc in maxcps:
    for i,path in enumerate(doc.path):
      if i==1:
        offbranchtopiclevel1=[j for j in md.children_topics(0) if j!=path]
        print('\n{}:{} \t\tAND\t\t {}:{} '.format(path,[i[0] for i in md.get_topic_words(path)][:2],offbranchtopiclevel1[0],[i[0] for i in md.get_topic_words(offbranchtopiclevel1[0])][:2]))
      elif i==2:
        offbranchtopiclevel2=[j for j in md.children_topics(offbranchtopiclevel1[0]) if j!=path]
        print('\n{}:{} \t\tAND\t\t {}:{} '.format(path,[i[0] for i in md.get_topic_words(path)][:2],offbranchtopiclevel2[0],[i[0] for i in md.get_topic_words(offbranchtopiclevel2[0])][:2]))
        # print('\n{}:{}'.format(path,[i[0] for i in md.get_topic_words(path)]))
    print('\n\n\n\nOriginal Unseen Word|Probability pairs:\n{}'.format(doc.get_words(top_n=10)))









