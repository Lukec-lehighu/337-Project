import random

def predict(inputValues, predRange=5, epsilon=0):
    if random.random() < epsilon:
        return (random.uniform(-predRange, predRange), random.uniform(-predRange, predRange))
    else:
        return (0,0) # TODO: prediction from model here
    