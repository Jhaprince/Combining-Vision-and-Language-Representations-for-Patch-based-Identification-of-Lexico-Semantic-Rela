'''
This is the file containing the func_train_text function.
The function prepares the data from the wnVectors, to be fed into a NN later on.
'''
# import from libraries
import numpy as np
from sklearn.utils import shuffle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from sklearn.model_selection import train_test_split
from gensim.scripts.glove2word2vec import glove2word2vec
# Importing functions from other files of this program
from fetch_embeddings_text import get_vector_and_neighbors_embeddings, diffPatchAttention, productCos, productCos1
from fetch_embeddings_utils_text import get_words_neighbors_list, similarityMatrix1

# Convert Glove to Word2Vec

def gloveToWord(glovePath, tmpfile):
    glove_file = datapath(glovePath)
    tmp_file = get_tmpfile(tmpfile)

    _ = glove2word2vec(glove_file, tmp_file)
    glove_model = KeyedVectors.load_word2vec_format(tmp_file)
    return glove_model

def func_train_test_text(vectors, name_data, embeddings, k, hps, file_paths):
    '''
    Params-
    vectors: WnVectors, this is our input dataset stored in a specific format. 
             This is a variable storing at a depth of 4, including 
             1) dataset selectior, 
             2) column selection (i.e. word1 or word2, along with task specificity) 
             3) row no corresponding to the word pair at hand
             4) N nearest neighbors, along with the similarity scores
    name_data:  name of the dataset currently working on
    embeddings: the GloVe embeddings
    k: number of neighbors in the patch size
    hps: a dictionary containing all the flag values
    '''
    # dictionary mapping the dataSelectior value:
    data_selector = {'Cogalex': 3, 'Weeds': 4, 'Rumen': 0, 'Root9': 1, 'Bless': 2, 'Root9_Bal': 5, \
                     'Bless_Bal': 6, 'Cogalex_Bal': 7, 'Weeds_Bal': 8, 'Root9+Bless+Weeds': 9, \
                     'Root9+Bless+Rumen': 10}
    i = data_selector[str(name_data)]
    print('selected data indicator is -', i)
    try:
        glove_model = gloveToWord(file_paths['glove_file'], file_paths['temp_file'])
    except Exception as err:
        print(err)

    print('xyz')
    #print(glove_model)
    
    wN1trainCoord = [[vectors[i][0][j][0], vectors[i][0][j][1][0:k]] for j in range(0, len(vectors[i][0]))]
    wN2trainCoord = [[vectors[i][1][j][0], vectors[i][1][j][1][0:k]] for j in range(0, len(vectors[i][1]))]

    wN1testCoord =  [[vectors[i][2][j][0], vectors[i][2][j][1][0:k]] for j in range(0, len(vectors[i][2]))]
    wN2testCoord =  [[vectors[i][3][j][0], vectors[i][3][j][1][0:k]] for j in range(0, len(vectors[i][3]))]

    wN1trainHyper = [[vectors[i][4][j][0], vectors[i][4][j][1][0:k]] for j in range(0, len(vectors[i][4]))]
    wN2trainHyper = [[vectors[i][5][j][0], vectors[i][5][j][1][0:k]] for j in range(0, len(vectors[i][5]))]

    wN1testHyper =  [[vectors[i][6][j][0], vectors[i][6][j][1][0:k]] for j in range(0, len(vectors[i][6]))]
    wN2testHyper =  [[vectors[i][7][j][0], vectors[i][7][j][1][0:k]] for j in range(0, len(vectors[i][7]))]

    wN1trainRando = [[vectors[i][8][j][0], vectors[i][8][j][1][0:k]] for j in range(0, len(vectors[i][8]))]
    wN2trainRando = [[vectors[i][9][j][0], vectors[i][9][j][1][0:k]] for j in range(0, len(vectors[i][9]))]

    wN1testRando =  [[vectors[i][10][j][0], vectors[i][10][j][1][0:k]] for j in range(0, len(vectors[i][10]))]
    wN2testRando =  [[vectors[i][11][j][0], vectors[i][11][j][1][0:k]] for j in range(0, len(vectors[i][11]))]

    print('wN1testRando', wN1testRando)
    print('wN1trainCoord', len(wN1trainCoord))
    print('wN1testCoord', len(wN1testCoord))
    print('wN1trainHyper', len(wN1trainHyper))
    print('wN1testHyper', len(wN1testHyper))
    print('wN1trainRando', len(wN1trainRando))
    print('wN1testRando', len(wN1testRando))


    if hps['concatNeigh']:
        if hps['diffPatchAtt']:
            print('diffPatchAtt')
            xtrainCoord1, ltrainCoord1, ltrainCoord2 = get_vector_and_neighbors_embeddings(embeddings, wN1trainCoord, wN2trainCoord, k, glove_model, hps)
            # print('xtrainCoord1', xtrainCoord1)
            print('xtrainCoord1', xtrainCoord1.shape)
            print('ltrainCoord1', ltrainCoord1)
            print('ltrainCoord2', ltrainCoord2)
            costrainCoord = diffPatchAttention(ltrainCoord1, ltrainCoord2, glove_model, hps, k)
            #if sequential architecture
            xtrainCoord = productCos(xtrainCoord1, costrainCoord)
            print('xtrainCoord', xtrainCoord.shape)
            #else if parallel architecture
            #prodw1w2trainCoord = productCos1(embeddings, ltrainCoord1, ltrainCoord2, costrainCoord)
            #xtrainCoord = np.hstack((xtrainCoord1, prodw1w2trainCoord))


            xtestCoord1, ltestCoord1, ltestCoord2 = get_vector_and_neighbors_embeddings(embeddings, wN1testCoord, wN2testCoord, k, glove_model, hps)
            costestCoord = diffPatchAttention(ltestCoord1, ltestCoord2, glove_model, hps, k)
            #if sequential architecture
            xtestCoord = productCos(xtestCoord1, costestCoord)
            #if parallel architecture
            #prodw1w2testCoord = productCos1(embeddings, ltestCoord1, ltestCoord2, costestCoord)
            #xtestCoord = np.hstack((xtestCoord1, prodw1w2testCoord))

            xtrainHyper1, ltrainHyper1, ltrainHyper2 = get_vector_and_neighbors_embeddings(embeddings, wN1trainHyper, wN2trainHyper, k, glove_model, hps)
            costrainHyper = diffPatchAttention(ltrainHyper1, ltrainHyper2, glove_model, hps, k)
            #if sequential architecture
            xtrainHyper = productCos(xtrainHyper1, costrainHyper)
            #if parallel architecture
            #prodw1w2trainHyper = productCos1(embeddings, ltrainHyper1, ltrainHyper2, costrainHyper)
            #xtrainHyper = np.hstack((xtrainHyper1, prodw1w2trainHyper))


            xtestHyper1, ltestHyper1, ltestHyper2 = get_vector_and_neighbors_embeddings(embeddings, wN1testHyper, wN2testHyper, k, glove_model, hps)
            costestHyper = diffPatchAttention(ltestHyper1, ltestHyper2, glove_model, hps, k)
            #if sequential architecture
            xtestHyper = productCos(xtestHyper1, costestHyper)
            #if parallel architecture
            #prodw1w2testHyper = productCos1(embeddings, ltestHyper1, ltestHyper2, costestHyper)
            #xtestHyper = np.hstack((xtestHyper1, prodw1w2testHyper))

            xtrainRando1, ltrainRando1, ltrainRando2 = get_vector_and_neighbors_embeddings(embeddings, wN1trainRando, wN2trainRando, k, glove_model, hps)
            costrainRando = diffPatchAttention(ltrainRando1, ltrainRando2, glove_model, hps, k)
            #if sequential architecture
            xtrainRando = productCos(xtrainRando1, costrainRando)
            #if parallel architecture
            #prodw1w2trainRando = productCos1(embeddings, ltrainRando1, ltrainRando2, costrainRando)
            #xtrainRando = np.hstack((xtrainRando1, prodw1w2trainRando))


            xtestRando1, ltestRando1, ltestRando2 = get_vector_and_neighbors_embeddings(embeddings, wN1testRando, wN2testRando, k, glove_model, hps)
            costestRando = diffPatchAttention(ltestRando1, ltestRando2, glove_model, hps, k)
            #if sequential architecture
            xtestRando = productCos(xtestRando1, costestRando)
            #if parallel architecture
            #prodw1w2testRando = productCos1(embeddings, ltestRando1, ltestRando2, costestRando)
            #xtestRando = np.hstack((xtestRando1, prodw1w2testRando))

        else :
            xtrainCoord = get_vector_and_neighbors_embeddings(embeddings, wN1trainCoord, wN2trainCoord, k, glove_model)
            xtestCoord= get_vector_and_neighbors_embeddings(embeddings, wN1testCoord, wN2testCoord, k, glove_model)

            xtrainHyper= get_vector_and_neighbors_embeddings(embeddings, wN1trainHyper, wN2trainHyper, k, glove_model)
            xtestHyper = get_vector_and_neighbors_embeddings(embeddings, wN1testHyper, wN2testHyper, k, glove_model)

            xtrainRando = get_vector_and_neighbors_embeddings(embeddings, wN1trainRando, wN2trainRando, k, glove_model)
            xtestRando = get_vector_and_neighbors_embeddings(embeddings, wN1testRando, wN2testRando, k, glove_model)

    if hps['diffPatch'] :
        print('diffPatch')
        ltrainCoord1, ltrainCoord2 = get_words_neighbors_list(wN1trainCoord, wN2trainCoord, k)
        ltestCoord1, ltestCoord2  = get_words_neighbors_list(wN1testCoord, wN2testCoord, k)

        ltrainHyper1, ltrainHyper2 = get_words_neighbors_list(wN1trainHyper, wN2trainHyper, k)
        ltestHyper1, ltestHyper2  = get_words_neighbors_list(wN1testHyper, wN2testHyper, k)

        ltrainRando1, ltrainRando2 = get_words_neighbors_list(wN1trainRando, wN2trainRando, k)
        ltestRando1, ltestRando2 = get_words_neighbors_list(wN1testRando, wN2testRando, k)

        costrainCoord = []
        for elem1, elem2 in zip(ltrainCoord1, ltrainCoord2):
            costrainCoord.append(similarityMatrix1(elem1+elem2, glove_model)[:k+1, k+1:].reshape(1, -1)) # to avoid duplicate the data
        CtrainCoord = np.vstack((costrainCoord))
        print(CtrainCoord.shape) 

        costestCoord = []
        for elem1, elem2 in zip(ltestCoord1, ltestCoord2):
            costestCoord.append(similarityMatrix1(elem1+elem2, glove_model)[:k+1, k+1:].reshape(1, -1))
        CtestCoord = np.vstack((costestCoord))
        print(CtestCoord.shape)

        costrainHyper = []
        for elem1, elem2 in zip(ltrainHyper1, ltrainHyper2):
            costrainHyper.append(similarityMatrix1(elem1+elem2, glove_model)[:k+1, k+1:].reshape(1, -1))
        CtrainHyper = np.vstack((costrainHyper))
        print(CtrainHyper.shape)


        costestHyper = []
        for elem1, elem2 in zip(ltestHyper1, ltestHyper2):
            s1 = similarityMatrix1(elem1+elem2, glove_model)[:k+1, k+1:].reshape(1, -1)
            costestHyper.append(s1)
        CtestHyper = np.vstack((costestHyper))
        print(CtestHyper.shape)

        costrainRando = []
        for elem1, elem2 in zip(ltrainRando1, ltrainRando2):
            costrainRando.append(similarityMatrix1(elem1+elem2, glove_model)[:k+1, k+1:].reshape(1, -1))
        CtrainRando = np.vstack((costrainRando))
        print(CtrainRando.shape)

        costestRando = []
        for elem1, elem2 in zip(ltestRando1, ltestRando2):
            costestRando.append(similarityMatrix1(elem1+elem2, glove_model)[:k+1, k+1:].reshape(1, -1))
        CtestRando = np.vstack((costestRando))
        print(CtestRando.shape)

        try :
            ptrainCoord = np.hstack((ptrainCoord,CtrainCoord))
            ptestCoord = np.hstack((ptestCoord, CtestCoord))

            ptrainHyper = np.hstack((ptrainHyper,CtrainHyper))
            ptestHyper = np.hstack((ptestHyper, CtestHyper))

            ptrainRando = np.hstack((ptrainRando,CtrainRando))
            ptestRando = np.hstack((ptestRando, CtestRando))

        except NameError:
            ptrainCoord = CtrainCoord
            ptestCoord = CtestCoord

            ptrainHyper = CtrainHyper
            ptestHyper = CtestHyper
            
            ptrainRando = CtrainRando
            ptestRando = CtestRando

    try :
        xtrainCoord = np.hstack((xtrainCoord, ptrainCoord))
        print('here4', xtrainCoord.shape)
        xtestCoord = np.hstack((xtestCoord, ptestCoord))

        xtrainHyper = np.hstack((xtrainHyper, ptrainHyper))
        xtestHyper = np.hstack((xtestHyper, ptestHyper))

        xtrainRando = np.hstack((xtrainRando, ptrainRando))
        xtestRando = np.hstack((xtestRando, ptestRando))
    except :
        try :
            xtrainCoord = ptrainCoord
            print('here4Except', xtrainCoord.shape)
            xtestCoord = ptestCoord

            xtrainHyper = ptrainHyper
            xtestHyper = ptestHyper

            xtrainRando = ptrainRando
            xtestRando = ptestRando
        except :
            pass

    x_train_1, x_train_2= np.vstack((xtrainCoord, xtrainRando)), np.vstack((xtrainHyper, xtrainRando))
    print('here5', x_train_1.shape)
    y_train_1, y_train_2 = [1]*len(xtrainCoord) + [0]*len(xtrainRando), [1]*len(xtrainHyper) + [0]*len(xtrainRando)

    x_test_1, x_test_2= np.vstack((xtestCoord, xtestRando)), np.vstack((xtestHyper, xtestRando))
    y_test_1, y_test_2 = [1]*len(xtestCoord) + [0]*len(xtestRando), [1]*len(xtestHyper) + [0]*len(xtestRando)

    x_train_1, y_train_1 = shuffle(x_train_1, y_train_1, random_state=1234)
    x_train_2, y_train_2 = shuffle(x_train_2, y_train_2, random_state=1234)
    x_test_1, y_test_1 = shuffle(x_test_1, y_test_1, random_state=1234)
    x_test_2, y_test_2 = shuffle(x_test_2, y_test_2, random_state=1234)
    assert len(x_train_1) == len(y_train_1)
    assert len(x_train_2) == len(y_train_2)
    assert len(x_test_1) == len(y_test_1)
    assert len(x_test_2) == len(y_test_2)
    data = {}
    for name, x_train, y_train, x_test, y_test in zip(["Coord-Random", "Hyper-Random"], [x_train_1, x_train_2], [y_train_1, y_train_2], [x_test_1, x_test_2], [y_test_1, y_test_2]):
        # Perform the splits in train, validation, unlabeled
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, stratify=y_train,  test_size=0.10, random_state=1234,)

        # keep the train/validation/test splits so that hey can be used with multitask learning and the results are comparable between them
        #print('name '+name)
        data[name]={"x_train": x_train, "y_train":y_train, "x_valid":x_valid,"y_valid":y_valid, "x_test":x_test,  "y_test":y_test} 
        

    sum_task1Vsrandom=len(data["Coord-Random"]["y_train"])+len(data["Coord-Random"]["y_test"])+len(data["Coord-Random"]["y_valid"])
    sum_task2Vsrandom=len(data["Hyper-Random"]["y_train"])+len(data["Hyper-Random"]["y_test"])+len(data["Hyper-Random"]["y_valid"])
    return data
