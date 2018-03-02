import lime.lime_text as lt
import keras
# my self written library
import projectlib as pl
import numpy as np
import pickle


# read alphabet for character-vectorization of the input text
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"
alphabet = open(alphabetPath).read()
maxChars = 1014

# This is the text that I want to explain
inputText2 = str("I did not like the food and the drinks were expensive. Still I love the atmosphere and the people in this nice spot")
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
explainer = lt.LimeTextExplainer(kernel_width=25, verbose=False, class_names=["positive", "negative"],
                           feature_selection="forward_selection", split_expression=" ", bow=True)

exp = explainer.explain_instance(text_instance=inputText, labels=[0, 1],
                     classifier_fn=predictFromText, num_features=11, num_samples=100)

# print(exp.__class__)
fig = exp.as_pyplot_figure(label = 0)
fig.savefig("limePositive.pdf")
fig = exp.as_pyplot_figure(label = 1)
fig.savefig("limeNegative.pdf")


#exp.show_in_notebook()
exp.save_to_file("limeResult.html", [1])



# close keras
keras.backend.clear_session()


