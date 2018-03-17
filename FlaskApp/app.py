"""
Link collection

nltk text mining etc
https://realpython.com/blog/python/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/

basic flask html steps
https://www.tutorialspoint.com/flask/flask_templates.htm

url text requests
https://scotch.io/bar-talk/processing-incoming-request-data-in-flask

add input fields to flask app
http://hplgit.github.io/web4sciapps/doc/pub/._web4sa_flask006.html
in general super neat intro to flask etc.
http://hplgit.github.io/web4sciapps/doc/pub/._web4sa_flask000.html

"""


from flask import Flask, render_template, request
from io import BytesIO
import keras
import numpy as np
import matplotlib.pyplot as plt
import projectlib as pl
import base64
import lime.lime_text as lt


########################
#
#   FUNCTIONS
#
########################

modelPath = '../charCnn8Huge.h5'

# helper function for Prediction Plot
def predictCnn(textInput):

    # basically same code as in detect_review
    alphabetPath = "../alphabet.txt"
    alphabet = open(alphabetPath).read()
    maxChars = 1014

    input = textInput
    recodeText = pl.generate_one_hot(text=input, alphabet=alphabet, maxChars=maxChars)
    # load model
    model = keras.models.load_model(modelPath)

    # only transpose AND NOT RESHAPE the inputs!!!
    pred = model.predict(recodeText.transpose())
    return([pred[0][0], pred[0][1]])

# Helper to build the shutdown route
# Option to shutdown server from url
# http://flask.pocoo.org/snippets/67/
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()



##################
#
#       APP
#
###################

app = Flask(__name__)

# index offers input to user
@app.route('/')
def index():
    return render_template('form.html')

@app.route('/home')
def home():
    return render_template('form.html')

#simple prediction
@app.route("/predictInput", methods=['POST'])
def predictInput():
    # get text 1 from first form and set to lower case
    text = request.form['text1'].lower()
    # to delete previous plot
    plt.close()
    # create plot
    x = [1, 2]
    y = predictCnn(textInput=text)
    plt.bar(x, y)
    plt.ylim([0, 1])
    plt.xlim([1, 2])
    plt.xticks([1, 2],("positive", "negative"))
    #plt.ticklabels(["positive", "negative"])
    plt.title("Predicted Sentiment Probability")

    # render in html
    figfile = BytesIO()
    plt.savefig(figfile, format = 'png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result = str(figdata_png)[2:-1]
    return render_template('plot.html', result = result)

@app.route("/query")
def query():
    value = request.args.get('value')
    return '<h1> entered value is: {}</h1>'.format(value)


# LIME explanation
@app.route("/explainInput", methods=['POST'])
def explainInput():
    # get text 2 from first form and set to lower case
    text = request.form['text2'].lower()

    explainer = lt.LimeTextExplainer(kernel_width=25, verbose=True, class_names=["positive", "negative"],
                                     feature_selection="lasso_path", split_expression=" ", bow=False)
    
    # still super hacky implementation in projectlib, yet running
    exp = explainer.explain_instance(text_instance=text, labels=[0, 1],
                                     classifier_fn=pl.predictFromText, num_features=5, num_samples=1000)

    htmlResult = exp.as_html(labels=[1], predict_proba=True, show_predicted_value=True)

    # add home button to end of file
    htmlResult = htmlResult.replace("</body></html>",
                                 "<button type=\"button\" onclick=\"window.location.href=\'/home\';\">Home</button> \n </body></html>")

    return htmlResult


@app.route('/shut')
def shut():
   shutdown_server()
   return 'Server shutting down...'

if __name__ == "__main__":
    app.run()


