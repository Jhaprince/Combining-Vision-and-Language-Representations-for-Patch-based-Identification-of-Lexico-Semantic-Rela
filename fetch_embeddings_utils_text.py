import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity

# Get the cosine similarity matrix
def similarityMatrix1(words_list, glove_model):
    sim = cosine_similarity(glove_model[words_list])		
    return sim

# Get the cosine distance (sim+1) matrix for PageRank on global patch
def similarityMatrix(words_list, glove_model):
    sim = cosine_similarity(glove_model[words_list])
    for i in range(0, sim.shape[0]):
        for j in range(0, sim.shape[0]):
            sim[i][j] += 1
    return sim

# Get list of words and their neighbors
def get_words_neighbors_list(wN1, wN2, k) :
    # Get list of word1 + its neighbors
    list_words1 = []
    cnt=0
    for item in wN1:
        l1 = []
        l1.append(item[0])
        for i in range(0,k):
            l1.append(item[1][i][0])
        list_words1.append(l1)
        cnt=cnt+1
        
        if(cnt==min(len(wN1),len(wN2))):
            break

    # Get list of word2 + its neighbors
    list_words2 = []
    cnt=0
    for item in wN2:
        l2 = []
        l2.append(item[0])
        for i in range(0,k):
            l2.append(item[1][i][0])
        list_words2.append(l2)
        cnt=cnt+1
        if(cnt==min(len(wN1),len(wN2))):
            break

    return list_words1, list_words2

# CUBOID FUNCTION
def cuboid_fn(l1, l2, embeddings_voca, k, dimlen):  #k= no. of neighbours (k=10, not 11)
    z = np.zeros((dimlen, k + 1, k + 1))
    for i in range(k + 1):
        for j in range(k + 1):
            z[:, i, j] = np.multiply(embeddings_voca[l1[i]], embeddings_voca[l2[j]])
    z = z.reshape(1, -1)
    return z


