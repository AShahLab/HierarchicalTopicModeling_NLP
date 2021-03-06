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

#Notes:
# 1. Keep a look out for the 3 checkpoints. Those are areas that may need some looking into for improvement
# 2. First checkpoint is meant for converting nltk useage  to spacy-->>only relevant to lemmatization, if there would be a performance advantage
# 3. Second checkpoint is for looking into the log-likelihood mess-up that I slackd about, we can continue with it or change it to the logically correct way. May need more research. If needed, we can also try out the other idea in ideas.py i.e. LDA-->>HLDA
# 4. Third checkpoint is for the visualization printing. The level 1 topic currently is being picked up as the first number that is non unique under the root topics (0) children. Maybe we can look into ways of identifying better ways to select topics e.g. any similarity measure, not sure if that would affect performance though
#Folder Structure Notes:
#1. test_hlda*.tmm<--Files are the individual HLDA models where the number(*) is the Topic Number
#2. dict<-- is the dictionary that is used to convert the corpus to a bag of words for the LDA model
#3. corp.json<-- is the corpus that is used for the LDA model
#4. df-dom.csv<-- is the dataframe that is a product of the LDA model and is used for the splitting and further training using HLDA model
#5. test-corpus*.cached.cps<-- is the corpus that is used for the HLDA model where the number(*) is the Topic Number
#6. lda<-- folder is the folder that contains the LDA model and its associated files



nlp = spacy.load("en_core_web_md")  # Loading the spacy model (medium sized)
#update the stopwords list with some more words that seem to be common in the data
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
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda0.tmm')#TomotoPy HLDAModel
mdl1 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda1.tmm')#TomotoPy HLDAModel
mdl2 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda2.tmm')#TomotoPy HLDAModel
mdl3 = tp.HLDAModel.load(
    '/Users/awaisshah/Desktop/Python/Code/HierarchicalTopicModeling_NLP/HierarchicalTopicModeling_NLP/subsample/test-hlda3.tmm')#TomotoPy HLDAModel

cluster_names = {  # name the clusters as seems reasonable based on the LDA model outputs and the HLDA model outputs
    0: "Science and Technology",
    1: "Health/Science and Business",
    2: "Business/Tech/BioTech",
    3: "Entertainment/News"
}
# Sample Test articles
#----------------------------------------------------------------------------------------------------------------------
# other_texts = {'txt':['The human genome is made up of about 3.1 billion DNA subunits, pairs of chemical bases known by the letters A, C, G and T. Genes are strings of these lettered pairs that contain instructions for making proteins, the building blocks of life. Humans have about 30,000 genes, organized in 23 groups called chromosomes that are found in the nucleus of every cell']}
other_texts = {'txt': [
    'Oliveira missed championship weight by half a pound at the official weigh-ins, using up almost the entirety of the initial two-hour window before making his first attempt and then taking another hour before stepping to the scale again. Both times he came in at 155.5 pounds. His UFC 274 main event challenger Justin Gaethje showed up minutes after the weigh-ins began, hitting 155 on the dot. It was later announced that Oliveira would be stripped of his title and that only Gaethje would be eligible to leave Footprint Center in Phoenix with the belt around his waist.']}
# other_texts = {'txt':['The Centre for Creative and Immersive XR (extended reality - an umbrella term for all types of digital reality, from the immersive VR viewed in headsets, to AR games like Pokemon Go, in which graphics are posted over the real world as seen through a smartphone camera) has received more than ??5m in funding, including a ??3.6m government grant']}
# other_texts = {'txt':['Rooftop solar has a ???huge potential??? to cut air pollution, create jobs, protect against outages and shrink utility bills, said Mark Jacobson, a professor of civil and environmental engineering at Stanford University. The technology will be vital for the U.S. to make the transition to all-renewable energy by 2050, he said.']}
# other_texts = {'txt':['The Kings finalized a deal with Brown on Sunday after meeting with him over two days late last week, sources said. Brown comes to the Kings with a clear organizational mandate: End the longest playoff drought in NBA history and return the Kings to the postseason for the first time in 16 years.']}
# other_texts = {'txt':['Patients with KRAS-mutant colorectal cancers do not respond to cetuximab, a monoclonal antibody against EGFR. A new proof-of-concept study presents a bispecific antibody with the ability to trigger EGFR degradation in LGR5+ cancer stem cells, and robust anti-tumor activity in KRAS-mutant and wild-type colorectal cancers.']}
# other_texts = {'txt':['Bitcoin continued to slide after a broader stock sell-off in the U.S. last week sent the cryptocurrency market into a frenzy and prompted bitcoin to plummet by roughly 10%. Bitcoin, the world???s largest digital currency by market value, was lower by about 3% at $33,438.03 late Sunday, according to data from Coin Metrics. This year, Bitcoin has been trading in a narrow range as it attempts to reclaim its highs of late 2021. The cryptocurrency is now down 50% from its peak price of $67,802.30 in November 2021. The drop comes after the blue-chip Dow Jones Industrial Average lost more than 1,000 points on Thursday and the Nasdaq plunged by 5%. Those losses marked the worst single-day drops since 2020. The Dow and Nasdaq fell again on Friday. \
# # \ Meanwhile, the Federal Reserve on Wednesday raised its benchmark interest rate by half a percentage point as it responds to inflation pressures.\
# # \ The stock market rallied after Fed chair Jerome Powell said a larger rate hike of 75 basis points isn???t being considered. But by Thursday, investors had erased the Fed rally???s gains.\
# # \ The global cryptocurrency market cap was at $1.68 trillion on Sunday, according to data from CoinGecko.com, and cryptocurrency trading volume in the last day was at $119 billion.']}
# ---------------------------------------------------------------------------------------------------------------------
other_texts = pd.DataFrame(other_texts)  # Convert dictionary to dataframe

#Checkpoint! Try to update code for spacy instead of nltk
nltk.download('wordnet')# Download wordnet. It would be better to use the Spacy model instead of NLTK. This is only relevant to lemmatization.
lemmatizer = WordNetLemmatizer()# Initialize lemmatizer

#------------------------set up test data for hlda---------------------------------
pat = re.compile('\w+')# Initialize pattern checker for tokenization. This just ensures that we extract only words from the document
corpus_unseen = tp.utils.Corpus(# Initialize corpus for unseen texts
    tokenizer=tp.utils.SimpleTokenizer(stemmer=None, lowercase=True),#convert to lowercase and tokenize (split into words)
    stopwords=lambda x: len(x) <= 2 or x in en_stop or x.isnumeric() or not pat.match(x) or not lemmatizer.lemmatize(x)# Initialize tokenizer with lemmatizer and stopwords and pattern checker and length checker
)
unseen = other_texts['txt']# Add unseen texts to corpus
corpus_unseen.process(d.lower() for d in unseen)# Process unseen texts by sending them through the tokenizer and stopwords
#----------------------------------------------------------------------------------

#Inference

#Idea 2: Run all the HLDA models and get the log likelihood for each model and then compare them and see which one is the best
mdls=[mdl0,mdl1,mdl2,mdl3]#list of all the models
lst=[]
for i in mdls:#loop through all the models
  cps2, ll2=i.infer(corpus_unseen)#get the inferred topics and log likelihood for the unseen text using the hierarchical LDA model
  lst.append((cps2,abs(ll2)))#Checkpoint!: append the inferred topics and log likelihood to the list. This is where I am converting the Log-Liklihood to a positive value, leading to the mess-up (or the opposite). I am not sure why this is happening.
maxll=max(lst,key=lambda item:item[1])[1]#get the maximum log likelihood
maxcps=max(lst,key=lambda item:item[1])[0]#get the inferred topics for the unseen text using the hierarchical LDA model
max_index = lst.index((maxcps,maxll))#get the index of the model with the maximum log likelihood
md=mdls[max_index]#get the model with the maximum log likelihood using the max_index


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Formattiing the data for the visualization
#Tree structure with one random topic selected from level 1
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('Printing the tree structure for hlda model with one random topic selected from level 1')
print('Root Topic 0:   \t\t{}'.format(cluster_names.get(max_index))) # Root topic from cluster_names
for doc in maxcps:
    for i,path in enumerate(doc.path):
      if i==1:
        #offbranch variables are the variables that are not on the path defined by our HLDA model.
        offbranchtopiclevel1=[j for j in md.children_topics(0) if j!=path]#Checkpoint: Make sure that the level 1 path topic is not duplicated and that we get to pick the first that is unique. We could get more than one topic from the level 1 if needed. Another thing worth trying is to see another way to selecting child topics based on the tokens. Maybe cosine similarity or some other similarity measure.
        #path is the topic number, md.get_topic_words(path) gets the words for the topic, offbrancktopiclevel1 gives the offbranch topic number for level 1, md.get_topic_words(offbrancktopiclevel1) gives the words for the offbranch topic number for level 1
        print('\n{}:{} \t\tAND\t\t {}:{} '.format(path,[i[0] for i in md.get_topic_words(path)][:2],offbranchtopiclevel1[0],[i[0] for i in md.get_topic_words(offbranchtopiclevel1[0])][:2]))
      elif i==2:#working on level 2. Skip if not needed for visualization i.e. comment out or get rid of it. I would say in most cases the lower levels are closer in relevance to the test document.
        offbranchtopiclevel2=[j for j in md.children_topics(offbranchtopiclevel1[0]) if j!=path]
        print('\n{}:{} \t\tAND\t\t {}:{} '.format(path,[i[0] for i in md.get_topic_words(path)][:2],offbranchtopiclevel2[0],[i[0] for i in md.get_topic_words(offbranchtopiclevel2[0])][:2]))
        # print('\n{}:{}'.format(path,[i[0] for i in md.get_topic_words(path)]))
    print('\n\n\n\nOriginal Unseen Word|Probability pairs:\n{}'.format(doc.get_words(top_n=10)))# This just prints the top 10 words for the unseen text with the probabilities.









