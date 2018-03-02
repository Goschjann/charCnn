import pandas as pd
import matplotlib.pyplot as plt

########################
#
#       Train Progress
#
########################

dat = pd.read_csv(filepath_or_buffer="/home/jgucci/Desktop/uni/text_mining/tm_code_collection/charnn_keras/charCnn_7_polarity.csv", sep = ",")

# accuracy

plt.plot(dat['epoch'].tolist(), dat['acc'].tolist(), color = "skyblue", label = "train acc")
plt.plot(dat['epoch'].tolist(), dat['val_acc'].tolist(), color = "orange", label = "test acc")
# plt.ylim([0, 1])
plt.xlim([0, 9])
plt.title(s = "Accuracy in Training")
plt.xlabel(s = "Epochs")
plt.ylabel(s = "Accuracy")
plt.legend()
# plt.show()
plt.savefig('/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_polarity/results/trainAcc.png')
plt.close()

# loss

plt.plot(dat['epoch'].tolist(), dat['loss'].tolist(), color = "skyblue", label = "train loss")
plt.plot(dat['epoch'].tolist(), dat['val_loss'].tolist(), color = "orange", label = "test loss")
# plt.ylim([0, 1])
plt.xlim([0, 9])
plt.title(s = "Loss in Training")
plt.xlabel(s = "Epochs")
plt.ylabel(s = "Accuracy")
plt.legend()
# plt.show()
plt.savefig('/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_polarity/results/trainLoss.png')




