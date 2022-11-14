
# coding: utf-8

# In[1]:


#Initializing torch and cuda device
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


# In[2]:


import pandas as pd 
import numpy as np
import re


# In[3]:


import torch
import torchtext

# print("Torch Text Version : {}".format(torchtext.__version__))


# In[4]:


df_train = pd.read_csv('train_50k.csv')
# df_train.head()
# df_train = df_train[:40000]


# In[5]:


#Importing tokenizer from torch.data
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer('basic_english')


# In[6]:


train_articles = df_train['article'].values
tokenized_train_articles = [tokenizer(sent) for sent in train_articles]
# print(tokenized_train_articles[0])
train_highlights = df_train['highlights'].values
tokenized_train_highlights = [tokenizer(sent) for sent in train_highlights]
tokenized_train_highlights = [['<SOS>'] + sent + ['<EOS>'] for sent in tokenized_train_highlights]
# print(tokenized_train_highlights[0])


# In[7]:


split_ratio = 0.8
split = int(split_ratio * len(tokenized_train_articles))
train_articles = tokenized_train_articles[:split]
test_articles = tokenized_train_articles[split:]
train_highlights = tokenized_train_highlights[:split]
test_highlights = tokenized_train_highlights[split:]
print("Number of training examples : {}".format(len(train_articles)))
print("Number of test examples : {}".format(len(test_articles)))
tokenized_train_articles = train_articles
tokenized_test_articles = test_articles
tokenized_train_highlights = train_highlights
tokenized_test_highlights = test_highlights


# In[8]:


avg_article_length = np.mean([len(x) for x in tokenized_train_articles])
avg_highlight_length = np.mean([len(x) for x in tokenized_train_highlights])
print("Average article length : {}".format(avg_article_length))
print("Average highlight length : {}".format(avg_highlight_length))


# In[9]:


#Importing Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader

#Creatin vocabulary
def create_vocab(tokenized):
    vocab = {}
    freq = {}
    #add  and  tokens
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<SOS>'] = 2
    vocab['<EOS>'] = 3
    freq['<PAD>'] = 0
    freq['<UNK>'] = 0
    freq['<SOS>'] = 0
    freq['<EOS>'] = 0
    #add tokens from tokenized sentences to vocab and freq
    for sent in tokenized:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
                freq[word] = 1
            else:
                freq[word] += 1
    #words with freq less than 5 are replaced with  token
    vocab_final = {}
    vocab_final['<PAD>'] = 0
    vocab_final['<UNK>'] = 1
    vocab_final['<SOS>'] = 2
    vocab_final['<EOS>'] = 3
    #add tokens from tokenized sentences to vocab_final if freq is greater than 5
    for word in vocab:
        if freq[word] >= 3:
            vocab_final[word] = len(vocab_final)
    return vocab_final

#build vocab from tokenized sentences
vocab = create_vocab(tokenized_train_articles)
print(list(vocab.items())[:10])
#print length of vocab
print(len(vocab))


# In[10]:


#Changing tokens in tokenized sentences to indices
def token2index_dataset(tokenized):
    indices = []
    for sent in tokenized:
        index = []
        for word in sent:
            if word in vocab:
                index.append(vocab[word])
            else:
                index.append(vocab['<UNK>'])
        indices.append(index)
    return indices
train_articles = token2index_dataset(tokenized_train_articles)
train_highlights = token2index_dataset(tokenized_train_highlights)
test_articles = token2index_dataset(tokenized_test_articles)
test_highlights = token2index_dataset(tokenized_test_highlights)


# In[11]:


#Function to pad sentences to max length
def pad_sents(sents, pad_token, max_len):
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len:
            padded_sents.append(sent + [pad_token] * (max_len - len(sent)))
        else:
            padded_sents.append(sent[:max_len])
    return padded_sents

def pad_sents2(sents, pad_token, max_len):
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len:
            padded_sents.append(sent + [pad_token] * (max_len - len(sent)))
        else:
            padded_sents.append(sent[:max_len])
            #replace last token with <EOS> token
            padded_sents[-1][-1] = vocab['<EOS>']
    return padded_sents


# In[12]:


class PointerGenDataset(Dataset):
    def __init__(self, articles, highlights, vocab):
        self.articles = articles
        self.highlights = highlights
        self.vocab = vocab
        self.encoder_data = pad_sents(self.articles, self.vocab['<PAD>'], 300)
        self.decoder_data = pad_sents2(self.highlights, self.vocab['<PAD>'], 75)

    def __len__(self):
        return len(self.encoder_data), len(self.decoder_data)

    def __getitem__(self, idx):
        encoder_data = torch.tensor(self.encoder_data[idx])
        decoder_data = torch.tensor(self.decoder_data[idx])
        return encoder_data, decoder_data

#Creating dataset
train_dataset = list(PointerGenDataset(train_articles, train_highlights, vocab))
# print(train_dataset[1000])
# print(len(train_dataset))

#Creating dataset
test_dataset = list(PointerGenDataset(test_articles, test_highlights, vocab))
# print(test_dataset[12])
# print(len(test_dataset))



# In[13]:


#Create dataloaders and organize datasets based on batch size
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
encoder_train, decoder_train = next(iter(train_loader))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
encoder_test, decoder_test = next(iter(test_loader))
# print('Train Dataset:')
# print(encoder_train.shape) #batch_size,seq_len
# print(decoder_train.shape)
# print(len(train_loader))
# print(encoder_test.shape) #batch_size,seq_len
# print(decoder_test.shape)
# print(len(test_loader))


# In[14]:


#Loading GloVe vectors
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=100)


# In[15]:


def create_embedding_matrix(vocab, embedding_dim):
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    for word, index in vocab.items():
        if word in glove.stoi:
            embedding_matrix[index] = glove.vectors[glove.stoi[word]]
        elif word == '<UNK>':
            embedding_matrix[index] = torch.mean(embedding_matrix[:index], dim=0) #dim = 100
    return embedding_matrix.detach().clone()

#initialize embedding matrix
embedding_matrix = create_embedding_matrix(vocab, 100)
# print(embedding_matrix.shape)
# print(embedding_matrix[0])


# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[17]:


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, embedding_matrix):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    def forward(self, encoder_data):
        embeddings = self.embedding(encoder_data)
        #dimensions of embeddings: (batch_size, seq_len, embedding_dim)
        enc_output, (hidden,cell) = self.bi_lstm(embeddings)
        #dimensions of enc_output: (batch_size, seq_len, hidden_dim*2)
        #dimensions of hidden: (2, batch_size, hidden_dim)
        #dimensions of cell: (2, seq_len, hidden_dim)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        #dimensions of hidden: (batch_size, hidden_dim*2)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        #dimensions of cell: (batch_size, hidden_dim*2)
        hidden = self.fc(hidden)
        cell = self.fc(cell)
        #reduce enc_output to hidden_dim
        enc_output = self.fc(enc_output)
        return enc_output, hidden, cell
#declare encoder
encoder = Encoder(len(vocab), 100, 200, batch_size, embedding_matrix)
print(encoder)

#display dimensions
enc_output, enc_hidden, enc_cell = encoder(encoder_train)


# In[18]:


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # Defining the layers/weights required depending on alignment scoring method
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.embed_fc = nn.Linear(100, hidden_size, bias=False)
        self.hiddento1 = nn.Linear(hidden_size, 1, bias=False)
        self.embedsto1 = nn.Linear(100, 1, bias=False)
        #create a learnable parameter of (batch_size, 1)
        self.v = nn.Parameter(torch.rand(hidden_size, 1))
  
    def forward(self, decoder_hidden, encoder_outputs, embeddings):
        #encoder_outputs: (batch_size, seq_len, hidden_dim)
        #decoder_hidden: (batch_size, hidden_dim)
        alignmt_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        #alignmt_scores: (batch_size, seq_len)
        alignmt_weights = F.softmax(alignmt_scores, dim=1)
        #alignmt_weights: (batch_size, seq_len)
        context_vector = torch.bmm(encoder_outputs.transpose(1,2), alignmt_weights.unsqueeze(2)).squeeze(2)
        #context_vector: (batch_size, hidden_dim)
        p_gen = torch.sigmoid(self.hiddento1(decoder_hidden) + self.embedsto1(embeddings) + self.hiddento1(context_vector))
        return alignmt_weights, context_vector, p_gen

#declare attention
attention = Attention(200)
print(attention)

# alignmt_weights, context_vector, p_gen = attention(enc_hidden, enc_output)
# print(alignmt_weights.shape) # (batch_size, seq_len)
# print(context_vector.shape) # (batch_size, hidden_dim)
# print(p_gen.shape) # (batch_size, 1)


# In[19]:


class AttentionDecoder(nn.Module):
    def __init__(self,embed_dim, hidden_dim, batch_size, vocab_size, embedding_matrix, encoder, attention):
        super(AttentionDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.dec_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = attention
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.output_layer = nn.Linear(hidden_dim*2, vocab_size)
    def forward(self, encoder_data, decoder_data):
        #pass encoder data through encoder
        enc_output, enc_hidden, enc_cell = self.encoder(encoder_data)
        enc_seq_len = enc_output.shape[1]
        dec_seq_len = decoder_data.shape[1]
        dec_hidden = enc_hidden.unsqueeze(0) #dimensions: (1, batch_size, hidden_dim)
        dec_cell = enc_cell.unsqueeze(0)
        vocab_dist = torch.zeros((self.batch_size, dec_seq_len, self.vocab_size))
        for t in range(dec_seq_len):
            #pass decoder data through embedding layer
            embeddings = self.embedding(decoder_data[:,t])
            #dimensions of embeddings: (batch_size, embedding_dim)
            #pass embeddings through decoder
            dec_output, (dec_hidden, dec_cell) = self.dec_lstm(embeddings.unsqueeze(1), (dec_hidden, dec_cell))
            #dimensions of dec_output: (batch_size, hidden_dim) #dec_hidden: (1, batch_size, hidden_dim) #dec_cell: (1, batch_size, hidden_dim)
            #apply attention
            dec_attn_hidden = dec_hidden.squeeze(0)
            #dimensions of dec_attn_hidden: (batch_size, hidden_dim)
#             alignmt_weights, context_vector, p_gen = self.attention(dec_attn_hidden, enc_output)
            alignmt_weights, context_vector, p_gen = self.attention(dec_attn_hidden, enc_output, embeddings)
            #dimensions of alignmt_weights: (batch_size, seq_len)
            #dimensions of context_vector: (batch_size, hidden_dim)
            copy_vocab = torch.zeros((self.batch_size, self.vocab_size)).to(device)
            copy_vocab = copy_vocab.scatter_add_(1, encoder_data, alignmt_weights)
            #concatenate context_vector and dec_output
        
            dec_output = dec_output.squeeze(1)
            dec_output = torch.cat((dec_output, context_vector), dim=1)
            #dimensions of dec_output: (batch_size, hidden_dim*2)
            #pass dec_output through output layer
            dec_output = self.output_layer(dec_output)
            #dimensions of dec_output: (batch_size, vocab_size)
            p_final = p_gen* dec_output + copy_vocab * (1-p_gen)#dimensions: (batch_size, vocab_size)
            vocab_dist[:,t,:] = p_final
#             vocab_dist[:,t,:] = dec_output
        return vocab_dist

#declare decoder
decoder = AttentionDecoder(100, 200, batch_size, len(vocab), embedding_matrix, encoder, attention)
print(decoder)


# In[20]:


# voca = decoder(encoder_train, decoder_train)
# print(voca.shape) # (batch_size, seq_len, hidden_dim)


# In[21]:


model = decoder.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# In[22]:


from tqdm import tqdm
epochs = 1
for epoch in range(epochs):
    model.train()
    for i, (encoder_data, decoder_data) in enumerate(tqdm(train_loader)):
        encoder_data = encoder_data.to(device)
        decoder_data = decoder_data.to(device)
        optimizer.zero_grad()
        vocab_dist = model(encoder_data, decoder_data)
        vocab_dist = vocab_dist[:,1:,:] #dimensions: (batch_size, seq_len, vocab_size)
        vocab_dist = vocab_dist.permute(0,2,1).to(device)
        decoder_data = decoder_data[:,1:] #dimensions: (batch_size, seq_len)
        #apply cross entropy loss
        loss = criterion(vocab_dist, decoder_data)
        loss.backward()
        optimizer.step()
        if i % 10000 == 0:
            print("Epoch: {} | Batch: {} | Loss: {}".format(epoch, i, loss.item()))


# In[23]:


model.eval()
decoder_list = []
summary_list = []
for i, (encoder_data, decoder_data) in enumerate(test_loader):
    encoder_data = encoder_data.to(device) #dimensions: (batch_size, seq_len)
    batch_len = encoder_data.shape[0]
    decoder_data = decoder_data.to(device) #dimensions: (batch_size, seq_len)
    vocab_dist = model(encoder_data, decoder_data) #dimensions: (batch_size, seq_len, vocab_size) 
    vocab_dist = vocab_dist[:,1:,:] #dimensions: (batch_size, seq_len, vocab_size)
    #take the index of the word with the highest probability
    vocab_dist = torch.argmax(vocab_dist, dim=2) #dimensions: (batch_size, seq_len)
    vocab_dist = vocab_dist.tolist()
    decoder_data = decoder_data.tolist()
    for i in range(batch_len):
        decoder_list.append(decoder_data[i])
        summary_list.append(vocab_dist[i])


# In[24]:


vocab_inv = {v: k for k, v in vocab.items()}


# In[25]:


def index2token_dataset(indices):
    tokenized = []
    for index in indices:
        sent = []
        for word in index:
            sent.append(vocab_inv[word])
        tokenized.append(sent)
    return tokenized

decoder_list = index2token_dataset(decoder_list)
summary_list = index2token_dataset(summary_list)


# In[26]:


decoder_list = [' '.join(sent) for sent in decoder_list]
summary_list = [' '.join(sent) for sent in summary_list]
#preview 5 examples


# In[27]:


from rouge_score import rouge_scorer
# make a RougeScorer object with rouge_types=['rouge1']
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


# In[28]:


#calculating the average
avg_precision = sum(results['precision']) / len(results['precision'])
avg_recall = sum(results['recall']) / len(results['recall'])
avg_fmeasure = sum(results['fmeasure']) / len(results['fmeasure'])
print("Average Precision: ", avg_precision)
print("Average Recall: ", avg_recall)
print("Average F-Measure: ", avg_fmeasure)


# In[30]:


df_outty = pd.DataFrame({'Encoder': decoder_list, 'Summary': summary_list})
df_outty.head()
df_outty.to_csv('40k_train2.csv', index=False)

