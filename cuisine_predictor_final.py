#Import Statements
# General
import numpy as np
import pandas as pd
import json
from pprint import pprint

# Scikit
from sklearn import metrics

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier as RFC

def  main():
    debug = True

    # Import Data
    filename = 'train.json'
    data = pd.read_json(filename)

    # Print info to console
    if debug:
        print(" DATA INFO: ")
        print("-------------")
        print("Data Stored as Pandas Dataframe")
        print("DataFrame shape: ", data.shape)
        print("Columns: ", data.columns)
        print("-------------")
        print(" CUISINE HEAD: ")
        print("-------------")
        print(data['cuisine'].head())
        print("-------------")
        print(" ID HEAD: ")
        print("-------------")
        print(data['id'].head())
        print("-------------")
        print(" INGREDIENTS HEAD: ")
        print("-------------")
        print(data['ingredients'].head())
        print("-------------")
        print(" INGREDIENTS LIST INFO: ")
        print("-------------")

    # Summary stats
    run_total = 0
    size_list = []
    for i in range(data.shape[0]):
        run_total += len(data['ingredients'][i])
        size_list.append(len(data['ingredients'][i]))

    if debug:
        print(" Min ingredients: ", min(size_list))
        print(" Max ingredients: ", max(size_list))
        print(" Mean ingredients: ", run_total / data.shape[0])
        print("-------------")
        print("Encoding Labels")
        print("-------------")

    # Fit label encoder
    le = LabelEncoder()
    le.fit(data['cuisine'])
    data['cuisine_encoded'] = le.transform(data['cuisine'])

    if debug:
        print(" Classes: ", le.classes_)
        print(" Transform irish: ", le.transform(['irish']))
        print(" Inv Transform 8: ", le.inverse_transform([8]))
        print("-------------")
        print(" Cuisine & Cuisine Encoded Head: ")
        print(data['cuisine'].head())
        print(data['cuisine_encoded'].head())
        print("-------------")
        print("-------------")
        print(" Testing sub-dataset Info: ")
        print("-------------")

    # Use smaller dataset for testing
    test = data.iloc[:10000]

    if debug:
        print(" test shape: ", test.shape)
        print(" test type: ", type(test))
        print("-------------")

    # Preprocess data
    # NOTE:
    # Changing lists of strings into long strings
    # -> Spaces within ingredients changed to '_'
    # -> Spaces put between ingredients w/in each recipe

    # Clean lists into strings:

    # New column in dataframe for strings of ingredients
    test["ingred_string"] = np.zeros((test.shape[0]))

    for i in range(test.shape[0]):
        ingred_list = test['ingredients'][i]

        # Create string of ingredients
        temp_str = ""
        for l in ingred_list:
            temp_str += l.replace(" ", "_")
            temp_str += " "
        test["ingred_string"][i] = temp_str

    # Fit Count Vectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(test["ingred_string"])

    # Fit tf Transformer
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    # Train Classifier
    # Here, using Random Forest, should Cross-Validate w/ multiple models
    rfc_parameters = {'n_estimators': np.arange(100, 250, 50), 'criterion': ('gini', 'entropy'),
                      'max_features': ('sqrt', 'log2')}

    # Get Feature Vecs & Target Vec
    X = np.matrix(test["tf"])
    y = np.matrix(test["cuisine_encoded"]).T

    # Split data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_tf, y, test_size=0.2, random_state=0)

    # Cross-validate RFC
    rfc = RFC()
    rfc_clf = GridSearchCV(rfc, rfc_parameters, cv=3, verbose=1, n_jobs=4).fit(X_train, y_train)

    # Report accuracy
    print("Training Accuracy: ")
    print(rfc_clf.score(X_train, y_train))
    print("Testing Accuracy: ")
    print(rfc_clf.score(X_test, y_test))

    # Get optimal parameters
    print(rfc_clf.get_params())

    # Test the Classifier on some recipes
    new_recipe = ['pasta olive_oil black_pepper salt garlic parsley red_pepper_flakes parmesan ',
                  'onion bell_pepper salt paprika veggie_oil shredded_cheese flour_tortillas smoked_turkey',
                  'potatoes cream_cheese sour_cream butter salt pepper corned_beef',
                  'oyster_sauce bok_choy tofu carrots',
                  'sugar corn_syrup confectioners_glaze salt cocoa_powder palm_kernel_oil gelatin sesame_oil honey dextrose soy_lecithin food_coloring',
                  'tomato_sauce mozarella olive_oil pepper basil']
    X_new_counts = count_vect.transform(new_recipe)
    X_new_tfidf = tf_transformer.transform(X_new_counts)

    predicted = rfc_clf.predict(X_new_tfidf)

    for rec, cuis in zip(new_recipe, predicted):
        print('%r => %s' % (rec, le.inverse_transform([cuis])))


main()
