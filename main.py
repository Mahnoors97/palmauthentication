import  json
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
import pickle
import os
import numpy as np
from globals  import compute_balance_after

dir_path="cpas/"

with open(dir_path+'conf.json') as f:    
    config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
labels_path   = config["labels_path"]
test_size     = config["test_size"]
results     = config["results"]
model_path    = config["model_path"]
classifier_path = config["classifier_path"]
test_path= config["test_path"]
image_size = (224, 224)


def create_model():
   
    if model_name == "mobilenet":
      base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
      model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
    else:
      base_model = None

    return model

def predict_label(x):
    model_new=create_model()
    model_new.load_weights(dir_path+model_path+"0.1.h5")
    classifier = pickle.load(open(classifier_path, 'rb'))
    train_labels = os.listdir(train_path)
    x= np.expand_dims(x, axis=0)
    x= preprocess_input(x)
    feature = model_new.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier.predict(flat)
    #prediction 	= train_labels[preds[0]]
    return train_labels[preds[0]]


from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    
    return "Welcome to Contact Less PALM Authentication"

@app.route('/authenticate',methods = ['POST', 'GET'])
def authenticate():
    #image_name = request.args.get('image_name')
    print("Entered in authenticate method")
    json_string=request.get_json()
    loaded_json = json.loads(json.dumps(json_string))
    img=loaded_json['image_Array']
    #print("JSON String "+str(img))
    
    #path = test_path + "/"+image_name
    #img= image.load_img(path, target_size=image_size)
    x = image.img_to_array(img)
    
    #return "it is "+ predict_label(x)
    return "JSON String "+ predict_label(x)


@app.route('/authenticateURL')
def authenticateURL():
    image_name = request.args.get('image_name')
    print("Input image name "+image_name)
    path = test_path + "/"+image_name
    img= image.load_img(path, target_size=image_size)
    x = image.img_to_array(img)
    
    return "I think it is a " + predict_label(x)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
