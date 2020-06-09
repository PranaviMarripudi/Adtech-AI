from base import Model
import flask
from flask import request, jsonify
# loaded tells the model if we have precalculated model or not
# so loaded = False means it will calculate the model again
# tosave is for saving a model, it will save the model only when loaded = False

model_instance = Model(loaded = True,tosave = False)
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/cat',methods = ['POST'])
def findCat():
    title = []
    tit = request.form['title']
    title.append(tit)
    return jsonify(model_instance.predict(title))

# to predict some category we need to pass an array of strings for it
# currently it prints the output , we can make it so it can return that

app.run()
# inp = ["Donald trump is in the news again"]

# model_instance.predict(inp)
