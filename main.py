from base import Model
# loaded tells the model if we have precalculated model or not
# so loaded = False means it will calculate the model again
# tosave is for saving a model, it will save the model only when loaded = False

model_instance = Model(loaded = False,tosave = False)

# to predict some category we need to pass an array of strings for it
# currently it prints the output , we can make it so it can return that

inp = ["Donald trump is in the news again"]

model_instance.predict(inp)
