
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('20k_test2.csv')
df.head()


# In[3]:


decoder_list = df['Encoder'].tolist()
#tokenize the decoder data based on spaces
decoder_list = [sent.split() for sent in decoder_list]
#for each sentence, remove all tokens after EOS token
for i in range(len(decoder_list)):
    if '<EOS>' in decoder_list[i]:
        decoder_list[i] = decoder_list[i][:decoder_list[i].index('<EOS>')+1]
    else:
        decoder_list[i] = decoder_list[i][:decoder_list[i].index('.')+1]
        
decoder_list = [' '.join(sent) for sent in decoder_list]


# In[4]:


from rouge_score import rouge_scorer
# make a RougeScorer object with rouge_types=['rouge1']
summary_list = df['Summary'].tolist()

scorer = rouge_scorer.RougeScorer(['rougeL'])

# a dictionary that will contain the results
results = {'precision': [], 'recall': [], 'fmeasure': []}

# for each of the hypothesis and reference documents pair
for (h, r) in zip(decoder_list, summary_list):
    # computing the ROUGE
    score = scorer.score(h, r)
    # separating the measurements
    precision, recall, fmeasure = score['rougeL']
    # add them to the proper list in the dictionary
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['fmeasure'].append(fmeasure)


# In[5]:


#calculating the average
avg_precision = sum(results['precision']) / len(results['precision'])
avg_recall = sum(results['recall']) / len(results['recall'])
avg_fmeasure = sum(results['fmeasure']) / len(results['fmeasure'])
print("Average Precision: ", avg_precision)
print("Average Recall: ", avg_recall)
print("Average F-Measure: ", avg_fmeasure)


# In[6]:


index = int(input("Enter index: "))
print("Decoder: ", decoder_list[index])
print("Summary: ", df['Summary'][index])

