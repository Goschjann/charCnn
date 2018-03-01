import keras
import projectlib as pl
import matplotlib.pyplot as plt
import textwrap as tw

pl.check()

# read Test data
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"
alphabet = open(alphabetPath).read()
lenAlpha = len(alphabet)
maxChars = 1014

# Sample reviews

#mytext = "I did not like the restaurant at all. The people were angry, the dinner was not good. I would never ever go there again"
#mytext = "We enjoyed the food and beverages a lot. It was an awesome atmosphere being there and I will definitevly come back to this wonderful spot. This was the nicest spot I have ever been to in my whole life. "
#mytext = "are you kidding me???!! the lady at the drive through here didn't want to hand me my drink until i handed her my card! haha really??!! my girlfriend was fumbling through her purse to find the debit card and i tried to reach for the drink she was handing me out of the drive through window and she immediately pulled the drink back and said, do i look that untrustworthy? talk about making a customer feel low! i guess she thought i was going to drive off with a fountain drink and leave the rest of the order... what a joke!"
# I enjoyed my stay here a lot. My daugter was crazy about your superb tacos and we will definitvely come back soon!
# The dinner was very nice and the servants quite professional. Still, the drinks were too expensive and not even chilled as served. There was also an issue with my wife's meat. Though, Me and my son enjoyed the dinner and the nice atmosphere


input = input("Enter your review:")
recodeText = pl.generate_one_hot(text=input, alphabet=alphabet, maxChars=maxChars)

# check if encoding was correct
# dec = pl.decoder(onehotText=recodeText, alphabet= alphabet, maxChars=maxChars)
# print(dec)


# load model
model = keras.models.load_model("charCnn_6_polarity.h5")

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
text = tw.fill(tw.dedent(text=input), width = 80)
plt.figtext(0.5, 0.10, s = text, horizontalalignment="center", fontsize = 10,
            multialignment='left', bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                                             ec="0.5", pad=0.5, alpha=1), fontweight='normal')
plt.savefig('/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_polarity/results/artificalTestNeutral.png')
plt.close()



