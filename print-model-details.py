import json
from keras.models import model_from_json
import os

def load_model():
    """ Load existing trained model.
    """
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))
    print('File open')
    model.load_weights('model.h5')
    return model


curr_dir_contents = os.listdir()

if 'model.json' in curr_dir_contents:
    print('Loading model for continued training')
    model = load_model()
    model.summary()

else:
    print('Model does not exist.')
