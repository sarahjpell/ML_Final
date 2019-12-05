#Tom Hanlon
#Sarah Pell

# main method
# load data
# train
# test

# Import Statements
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from scipy.stats import randint as sp_randint

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

import time

# METHODS:

# Takes string of ingredients
# Returns list of ingredient strings
def processString(s):
    toReturn = s.replace('[','').replace(']','').replace('"','').replace("'",'').split(',')
    for i in toReturn:
        if i[0] == ' ':
            toReturn = toReturn[1:]
    return toReturn

# Takes list of strings, number of total ingredients, label encoder and OHE encoder
# Returns ONEHOT encoding as an array
def encode_ingred(l, total_n, encoder, ohencoder):
    # l is a list of strings
    l_int = []
    # print(l)
    # print(le.transform(l))
    for i in range(len(l)):
        # print(l[i])
        l_int.append(encoder.transform([l[i]]))
    # Now l_int has the integer encodings of the stuff
    toRet = np.zeros((1, total_n))
    arrayInt = np.array(l_int)
    # print(arrayInt.shape)
    arrayOhe = ohencoder.transform(arrayInt)
    #toRet = np.zeros((1, arrayOhe.shape[1]))
    for i in range(len(arrayOhe)):
        toRet += arrayOhe[i]

    return toRet

def main2():
    total_clock = time.clock()
    ### Import Data
    # path to dataset
    url = "RAW_recipes.csv"
    url2 = "train.json"
    # import the file
    data = pd.read_csv(url)

    # Trim dataset for testing
    data_trim = data[:5000]
    # data_trim = data.sample(1000)

    ### Preprocess Data

    ingredients_set = set()
    ### Turn strings of ingredients into lists of strings ***
    for i in range(data_trim.shape[0]):
        temp = data_trim['ingredients'][i]
        newEntry = temp.replace('[','').replace(']','').replace('"','').replace("'",'')
        newEntry = newEntry.split(',')
        for j in range(len(newEntry)):
            if newEntry[j][0] == ' ':
                newEntry[j] = newEntry[j][1:]
            ingredients_set.add(newEntry[j])
        data_trim['ingredients'][i] = newEntry


    ### Fit Label Encoder to those strings
    le = preprocessing.LabelEncoder()
    le.fit(np.array(list(ingredients_set)))
    encoded = le.transform(np.array(list(ingredients_set)))
    print("Encoded Shape: ", encoded.shape)

    ### Fit OHE Encoder to Label Encoder
    ohe = preprocessing.OneHotEncoder()
    ohe.fit(encoded.reshape(len(encoded),1))

    ### Create new DataFrame of Ingredient Binaries
    enc_time = time.clock()
    ingredientMatrix = np.zeros((data_trim.shape[0],len(ingredients_set)))
    num_ingred = len(ingredients_set)
    print("Ingredient Matrix Shape: ",ingredientMatrix.shape)
    for i in range(data_trim.shape[0]):
        if i%1000 == 0:
            print("encoded ",i," entries")
        ingred_list = data_trim['ingredients'][i]
        ingred_encoded = list()
        for ingred in ingred_list:
            ingred_encoded.append(le.transform([ingred]))

        temp_array = np.zeros((1,num_ingred))

        ingred_encoded = np.array(ingred_encoded).reshape(1,len(ingred_list))
        ingred_ohe = ohe.transform(ingred_encoded.T)

        for ohe_array in ingred_ohe:
            temp_array += ohe_array

        ingredientMatrix[i] = temp_array
    print("Encoding took ", time.clock()-enc_time, " seconds.")

    pca_clock = time.clock()
    ### Calculate PCA for that DataFrame
    print("Fitting PCA -- n_components = min(ingredientMatrix.shape) = ",min(ingredientMatrix.shape) )
    pca = PCA(n_components=min([20,21]), whiten=True).fit(ingredientMatrix)
    print("Fitting PCA took ", time.clock()-pca_clock, " seconds")
    #print("Fitting PCA -- n_components = 100")
    #pca = PCA(n_components=100, whiten=True).fit(ingredientMatrix)

    transformed_matrix = pca.transform(ingredientMatrix)

    k_clock = time.clock()
    ### Run KMeans on the PCA
    print("Running K-means -- k = 6")
    ### Maybe try fir_transform?
    kmeans = KMeans(n_clusters=20, random_state=0, verbose=1).fit(transformed_matrix)
    pred_y = kmeans.predict(transformed_matrix)
    print("Fitting kmeans took ", time.clock()-k_clock, " seconds")

    ### Fit & Plot TSNE for that data
    # Plot TSNE of data w/ k-means coloring
    tsne_clock =  time.clock()
    print("Plotting TSNE")
    test_embedded = TSNE(n_components=2,verbose=1).fit_transform(transformed_matrix)

    plt.subplot(2,2,1)
    plt.scatter(test_embedded[:, 0], test_embedded[:, 1], c=pred_y, s=50, cmap='viridis') #,learning_rate=300,perplexity=30

    test_embedded2 = TSNE(n_components=2, verbose=1,learning_rate=300,perplexity=30).fit_transform(transformed_matrix)
    plt.subplot(2,2,2)
    plt.scatter(test_embedded2[:, 0], test_embedded2[:, 1], c=pred_y, s=50, cmap='viridis')

    test_embedded3 = TSNE(n_components=2, verbose=1, learning_rate=100, perplexity=40).fit_transform(transformed_matrix)
    plt.subplot(2, 2, 3)
    plt.scatter(test_embedded3[:, 0], test_embedded3[:, 1], c=pred_y, s=50, cmap='viridis')

    test_embedded4 = TSNE(n_components=2, verbose=1, learning_rate=500, perplexity=20).fit_transform(transformed_matrix)
    plt.subplot(2, 2, 4)
    plt.scatter(test_embedded4[:, 0], test_embedded4[:, 1], c=pred_y, s=50, cmap='viridis')

    print("Fitting TSNE took ", time.clock() - tsne_clock, " seconds")

    print("Total runtime: ", time.clock() - total_clock, "seconds = ", (time.clock() - total_clock) / 60, " minutes")

    plt.show()

main2()