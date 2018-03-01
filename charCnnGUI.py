"""
GUI that allows to access the trained CharacterCNN
and predict the sentiment of any input given to it
"""


import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
from  matplotlib.backends.backend_tkagg  import  FigureCanvasTkAgg
from  matplotlib.figure  import  Figure

# cnn
import keras
import projectlib as pl



class  Fitting(tk.Frame):
    def  __init__(self , master=tk.Tk()):
        tk.Frame.__init__(self , master)
        self.master=master
        self.grid()
        self.createVariables()
        self.createWidgets()

    def createVariables(self):
        self.e_input_text = tk.StringVar()
        self.e_input_text.set('I did like the food very much and the atmosphere is awesome! Still I think the drinks are a little bit too expensive.')

    def createWidgets(self):

        # matplotlib  FigureCanvasTkAgg
        # initialize plot
        f = Figure(figsize =(8, 8), dpi =100)
        self.sp = f.add_subplot(111)
        self.x = [1, 2]
        x = self.x
        y = [0, 0]
        self.sp.bar(x, y)
        self.sp.set_ylim([0, 1])
        self.sp.set_xticks([1, 2])
        self.sp.set_xticklabels(["positive", "negative"])
        self.sp.set_title("Predicted Sentiment Probability")
        self.canvas = FigureCanvasTkAgg(f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row = 2, column =10, columnspan =25,  sticky=tk.NSEW)

        # entry
        self.e_input = tk.Entry(self , textvariable=self.e_input_text)
        self.e_input.grid(row=5, column =10,  columnspan =25,  sticky=tk.EW)


        # buttons
        self.b_plot = tk.Button(self , text="Predict the sentiment", command=self.on_plot)
        self.b_plot.grid(row=10,  column =10,  sticky=tk.EW)
        self.b_quit = tk.Button(self , text="Quit", command=self.master.destroy)
        self.b_quit.grid(row=10,  column =20,  sticky=tk.EW)

    def refreshPlot(self, x, y):
        # clear old plot data
        self.sp.clear()
        self.sp.bar(x, y)
        self.sp.set_ylim([0, 1])
        self.sp.set_xlim([1, 2])
        self.sp.set_xticks([1, 2])
        self.sp.set_xticklabels(["positive", "negative"])
        self.sp.set_title("Predicted Sentiment Probability")
        self.canvas.draw()

    def predictCnn(self, textInput):

        # basically same code as in detect_review
        alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"
        alphabet = open(alphabetPath).read()
        maxChars = 1014

        input = textInput
        recodeText = pl.generate_one_hot(text=input, alphabet=alphabet, maxChars=maxChars)
        # load model
        model = keras.models.load_model("charCnn_6_polarity.h5")

        # only transpose AND NOT RESHAPE the inputs!!!
        pred = model.predict(recodeText.transpose())
        return([pred[0][0], pred[0][1]])
        # keras.backend.clear_session()

        # gets called by the Predict Button
    def on_plot(self):
        x = self.x
        y = self.predictCnn(textInput = self.e_input_text.get())
        self.refreshPlot(x = x, y = y)

app = Fitting()
app.mainloop()