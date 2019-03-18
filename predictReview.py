import keras
import projectlib as pl
import matplotlib.pyplot as plt
import textwrap as tw

pl.check()

model_folder = "/home/jgucci/Desktop/uni/text_mining/tm_code_collection/"

modelFolder = '/home/jgucci/Desktop/uni/text_mining/tm_data/charCnn19.h5'

# read Test data
alphabetPath = "alphabet.txt"
alphabet = open(alphabetPath).read()
lenAlpha = len(alphabet)
maxChars = 1014

# Sample reviews

# input_text = input("Enter your review:")
input_text = "I did not like the restaurant at all. The people were angry, the dinner was not good. I would never ever go there again"
# input_text = "We enjoyed the food and beverages a lot. It was an awesome atmosphere being there and I will definitevly come back to this wonderful spot. This was the nicest spot I have ever been to in my whole life. "
# input_text = "it was ok, just a very normal restaurant with average prices and ok-ish dinner."
recodeText = pl.generate_one_hot(text=input_text, alphabet=alphabet, maxChars=maxChars)

# check if encoding was correct
dec = pl.decoder(onehotText=recodeText, alphabet= alphabet, maxChars=maxChars)
print(dec)

# load model
model = keras.models.load_model(modelFolder)

# only transpose AND NOT RESHAPE the inputs!!!
pred = model.predict(recodeText.transpose())
print(pred)
keras.backend.clear_session()

# plot and save the results
plt.bar(x=[1, 2], height=[pred[0][0], pred[0][1]])
plt.ylim((0, 1))
plt.xticks([1, 2], ("positive", "negative"))
plt.title("Prediction Probability Selfmade Test Review")
plt.subplots_adjust(bottom = 0.5)
text = tw.fill(tw.dedent(text=input_text), width = 80)
plt.figtext(0.5, 0.10, s = text, horizontalalignment="center", fontsize = 10,
            multialignment='left', bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                                             ec="0.5", pad=0.5, alpha=1), fontweight='normal')
plt.savefig('{}InputTest.png'.format(model_folder))
# plt.show()
plt.close()



