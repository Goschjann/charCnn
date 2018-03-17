# explainable charCnn for sentiment classification 
Implementation of character CNN by LeCunn et. al. for an university talk. Pimped with LIME to make the predictions of the black box charCNN more accessible and localy interpretable. Also, a small flask App is included for demonstration purposes. 


This readme is structured as follows:

1. Basic idea of CNNs for text classification
2. Implementation 
3. Basic idea of LIME
4. How to: handle those scripts

Work in progress

## 1. CNNs for text classification

1D convolution over character-encoded text. Example with the text "text mining":

![animation 1D convolution](https://github.com/Goschjann/charCnn/blob/master/images/cnnFilter.gif)
