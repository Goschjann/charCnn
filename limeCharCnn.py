import lime.lime_text as lt
import keras
# my self written library
import projectlib as pl
import numpy as np


# read alphabet for character-vectorization of the input text
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"
alphabet = open(alphabetPath).read()
maxChars = 1014

# This is the text that I want to explain
inputText2 = str("I did not like the food and the drinks were expensive. Still I like the atmosphere and the people in this nice spot.")
inputText = str("The food was fantastic. My wife loves their fresh pineapple pie and the nice atmosphere")
listTexts = [inputText, inputText2]

#
recodeText = pl.generate_one_hot(text=inputText, alphabet=alphabet, maxChars=maxChars)

model = keras.models.load_model("charCnn_7_polarity.h5")

# pipeline-like function
# takes raw text as input, converts it to one-hot character vectors via the alphabet
# feeds it into keras' model.predict() function to receive predictions
# works for lists of text as well as for single strings
# IMPORTANT: output numpy array in dimension d x k
def predictFromText(textInputList):

    # catch single string inputs and convert them to list
    if textInputList.__class__ != list:
        textInputList = [textInputList]
        print("caught single string")
    # list for predictions
    predStorage = []
    # loop through input list and predict
    for textInput in textInputList:

        recodeText = pl.generate_one_hot(text=textInput, alphabet=alphabet, maxChars=maxChars)
        pred = model.predict(recodeText.transpose())
        # control output of function
        # print(str(textInput), "\n", pred)
        predStorage.append(pred)

    # convert to dxk ndarray
    return(   np.hstack(predStorage).reshape(-1, 2))

# this works, yields an array with probabilities for both classes
#print(predictFromText(textInputList = listTexts))
#print(predictFromText(textInputList=inputText))


# Lime Explainer
# bow controls if words are perturbed or overwritten with UNKWORDZ
# False makes sense, if location of words is important as in this classifier
explainer = lt.LimeTextExplainer(kernel_width=25, verbose=True, class_names=["positive", "negative"],
                           feature_selection="highest_weights", split_expression=" ", bow=False)

print("yo")
exp = explainer.explain_instance(text_instance=inputText, labels=[0, 1],
                     classifier_fn=predictFromText, num_features=8, num_samples=5000)

print(exp)

html = exp.as_html(labels=[0, 1], predict_proba=True, show_predicted_value=True)

foo = exp.as_list(label = 1)
print(foo)

# print(exp.__class__)
fig = exp.as_pyplot_figure(label = 0)
fig.savefig("limePositive.pdf")
fig = exp.as_pyplot_figure(label = 1)
fig.savefig("limeNegative.pdf")


#exp.show_in_notebook()
exp.save_to_file("limeResult_hw_nobow5.html", [1])



# close keras
keras.backend.clear_session()


