'''

'''
# Importing from libraries
import numpy as np
import networkx as nx

# Importing functions from other files of this program
from fetch_embeddings_utils_image import get_words_neighbors_list, similarityMatrix, similarityMatrix1, cuboid_fn
from get_image_feature import get_image_clip_feature,unpickle_global_feature


# Get embedding of each element in the patches, weighted if attention and attNeigh == True
# Cosine cuboid input added if cuboid == True
def get_vector_and_neighbors_embeddings(wN1, wN2, k, hps):
    list_words1, list_words2 = get_words_neighbors_list(wN1, wN2, k)

    # Get embeddings
    embw1N = []
    for elem in list_words1:
        emb1 = []
        pp1 = similarityMatrix1(elem)

        if hps['attNeigh']:  # This is INTRA patch attention.
            # print("Intra-patch on.")
            g = nx.from_numpy_array(pp1)
            pr = nx.pagerank(g)
            for e, r in zip(elem, pr):
                emb1.append(get_image_clip_feature("/home/prince/clip_feat",e)* pr[r])  # Re-weighted embeddings according to pagerank values
        else:
            for e in elem:
                emb1.append(get_image_clip_feature("/home/prince/clip_feat",e))  # Concatenating embeddings of neighbours with each other.
        # Appending adjusted embeddings to a new list.
        embw1N.append(np.array(emb1).reshape(1, -1))  # shape of embw1N -> (no.of patches, no. of neighbours, embedding dim.)
    vecW1 = np.vstack(embw1N)

    embw2N = []
    for elem in list_words2:
        emb2 = []
        pp2 = similarityMatrix1(elem)

        if hps['attNeigh']:
            g = nx.from_numpy_array(pp2)
            pr = nx.pagerank(g)
            for e, r in zip(elem, pr):
                emb2.append(get_image_clip_feature("/home/prince/clip_feat",e)* pr[r])
        else:
            for e in elem:
                emb2.append(get_image_clip_feature("/home/prince/clip_feat",e))
        embw2N.append(np.array(emb2).reshape(1, -1))
    vecW2 = np.vstack(embw2N)
    vecW12 = np.hstack((vecW1, vecW2))

    if hps['cuboid']:
        cubW12 = []

        for elem1, elem2 in zip(list_words1, list_words2):
            cube = cuboid_fn(elem1, elem2, embeddings_voca, k, hps['glove_dim'])
            cubW12.append(cube)
        vecCube = np.array(cubW12).squeeze()
        vecW12C = np.hstack((vecW12, vecCube))

    if hps['cuboid']:
        if hps['diffPatchAtt']:
            return vecW12C, list_words1, list_words2
        else:
            return vecW12C
    else:
        if hps['diffPatchAtt']:
            return vecW12, list_words1, list_words2
        else:
            return vecW12

# Get the second attention vector (attention between patches) on the global graph.
def diffPatchAttention(list_words1, list_words2,hps, k):
    cosW = []
    for elem1, elem2 in zip(list_words1, list_words2):
        cosi = []
        s = similarityMatrix(elem1+elem2)
        if hps['setZeros']: # This is always kept as True
            s[:k+1, :k+1] = 0
            s[k+1:, k+1:] = 0
        g = nx.from_numpy_array(s)
        pr = nx.pagerank(g)
        for e, r in zip(elem1+elem2, pr):
            cosi.append(pr[r])
        cosW.append(np.array(cosi).reshape(1, -1))

    dP = np.vstack(cosW)
    print('diffPatchAttention done!')
    return dP

# Multiply a vector with its attention value 
def productCos(x, pr):
    print("productCos started")
    prodF = []
    for e, r in zip(x, pr):
        e = e.reshape(len(r), 512)
        prod = []
        for elem, ra in zip(e, r):
            prod.append(elem*ra)
        prodF.append(np.array(prod).reshape(1, -1))
    out = np.vstack(prodF)
    print("productCos finished")
    return out

# For the parallel architcture, multiply the concatenation of input vectors by the attention value
def productCos1(list_words1, list_words2, pr):
    try :
        embw1N = []
        for elem in list_words1:
            emb1 = []
            for e in elem:
                emb1.append(get_image_clip_feature("/home/prince/clip_feat",e))
            embw1N.append(np.array(emb1).reshape(1, -1))
        vecW1 = np.vstack(embw1N)

        embw2N = []
        for elem in list_words2:
            emb2 = []
            for e in elem:
                emb2.append(get_image_clip_feature("/home/prince/clip_feat",e))
            embw2N.append(np.array(emb2).reshape(1, -1))
        vecW2 = np.vstack(embw2N)
        vecW12 = np.hstack((vecW1, vecW2))
        output = productCos(vecW12, pr)
    except Exception as error:
        print(error)

    return output

