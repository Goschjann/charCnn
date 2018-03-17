

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap as tw
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics
import projectlib as pl
import csv

# 0 (= positive) and 1 (= negative)

modelName = "charCnn8Small"



pl.check()

# read Test data
# dataPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTest.csv"
# Small Yelp polarity data set by Le Cunn et al
dataPath = '/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_polarity/testPrep.csv'
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"
pd.options.display.max_colwidth = 1000

alphabet = open(alphabetPath).read()
lenAlpha = len(alphabet)
maxChars = 1014

# process data:


x_test, y_test = pl.buildSetTest(path_data=dataPath, path_alphabet=alphabetPath, maxChars=1014, skiprows = 0,
                                 amountData=38000)
print("ytest shape", "\n", y_test.shape)
print("xtest shape", "\n", x_test.shape)

"""

"""


# load model
model = keras.models.load_model((modelName + ".h5"))


# direct cnn output
predScore = model.predict(x = x_test, batch_size=1000)
predTrue = np.asarray(y_test).reshape(-1, 1)
hardPred = np.argmax(predScore, axis = 1).reshape(-1, 1)

both = np.hstack((hardPred, predTrue))
predCorrect = np.where(hardPred == predTrue, 1, 0)


# confusion matrix via sklearn
confMat = sklearn.metrics.confusion_matrix(y_pred=hardPred, y_true=y_test).astype("int")
# report = sklearn.metrics.classification_report(y_true=y_test, y_pred=hardPred, target_names=["positive", "negative"]).astype("int")

np.savetxt(("confMat4_model_" + modelName + ".csv"), confMat, delimiter=",", fmt = "%i")
# np.savetxt(("report_model_" + modelName + ".csv"), report, delimiter=",")

fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, predScore[:, 0], pos_label=0)

results = {"AUC": sklearn.metrics.auc(x = fpr, y = tpr, reorder=False),
           "F1_Score": sklearn.metrics.accuracy_score(y_true=y_test, y_pred=hardPred),
           "Accuracy": sklearn.metrics.accuracy_score(y_true=y_test, y_pred=hardPred)}

w = csv.writer(open("metrics_" + modelName + ".csv", "w"))
for key, val in results.items():
    w.writerow([key, val])



print(sklearn.metrics.roc_auc_score(y_true=y_test, y_score=predScore[:, 0]))
print(sklearn.metrics.auc(x = fpr, y = tpr, reorder=False))
print(sklearn.metrics.accuracy_score(y_true=y_test, y_pred=hardPred))
print(sklearn.metrics.f1_score(y_true=y_test, y_pred=hardPred))


keras.backend.clear_session()




"""
## examples
nExamples = 1000
example = pd.read_csv(filepath_or_buffer=dataPath, nrows= nExamples)

# 0 = positive
# 1 = negative
#TODO: fix plot error with dollar signs at i = 92

for i in range(0, nExamples):
    plt.bar(x=[1, 2], height=predVal[i, :])
    plt.ylim((0, 1))
    plt.xticks([1, 2], ("positive", "negative"))
    plt.title("Prediction Probability Test Review " + str(i))
    plt.subplots_adjust(bottom = 0.5)

    text = tw.fill(tw.dedent(text=str(example.iloc[i, 0])[0:700]), width = 80)
    plt.figtext(0.5, 0.05, text, horizontalalignment="center", fontsize = 10,
                multialignment='left', bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                                                 ec="0.5", pad=0.5, alpha=1), fontweight='normal')

    plt.savefig('/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_polarity/results/test_' + str(i) + ".png")
    plt.close()


Proof: 
x_test.shape
a = x_test[0, :, :].reshape(maxChars * lenAlpha)
can be checked in VViewer
first = x_test[0, :, :]
firstRes = np.transpose(first)

dec = pl.decoder2dim(firstRes, alphabet = alphabet, maxChars= maxChars)
print(dec)
"""
