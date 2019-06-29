import  json
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
import pickle
import tensorflow as tf
import numpy as np


dir_path=""

with open(dir_path+'conf/conf.json') as f:    
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

user_labels=np.load("user_labels.npy")

print(user_labels.shape)

def create_model():
   
    if model_name == "mobilenet":
      base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
      #base_model.summary()
      model = Model(input=base_model.input, output=base_model.layers[-1].output)
    else:
      base_model = None

    return model

def predict_label(x):
    
    x= np.expand_dims(x, axis=0)
    x= preprocess_input(x)
    feature = app.model.predict(x)
    #print(feature.shape)
    flat = feature.flatten()
    #print(flat.shape)
    flat = np.expand_dims(flat, axis=0)
    #print(flat.shape)
    preds = app.classifier.predict(flat)
    #print(preds)
    print(preds[0])
    prediction 	= user_labels[preds[0]]
    return prediction


from flask import Flask
from flask import request

app = Flask(__name__)


@app.before_first_request
def load_model_to_app():
    # Load the model
    
    model_new=create_model()
    model_new.load_weights(dir_path+model_path+"0.1.h5")
    app.classifier = pickle.load(open(classifier_path, 'rb'))
    app.model = model_new
        
    # Save the graph to the app framework.
    app.graph = tf.get_default_graph()

@app.route('/')
def index():
    
    return "Welcome to Contact Less PALM Authentication"

@app.route('/authenticate',methods = ['POST', 'GET'])
def authenticate():
    #image_name = request.args.get('image_name')
    print("Entered in authenticate method")
    json_string=request.get_json()
    loaded_json = json.loads(json.dumps(json_string))
    print(loaded_json)
    img=loaded_json['image_Array']
    
    graph = app.graph    
    with graph.as_default():
        return "I think it is a " + str(predict_label(img))
    


@app.route('/authenticateURL')
def authenticateURL():
    image_name = request.args.get('image_name')
    print("Input image name "+image_name)
    path = test_path + "/"+image_name
    img= image.load_img(path, target_size=image_size)
    x = image.img_to_array(img)
    
    graph = app.graph    
    with graph.as_default():
        return "I think it is a " + str(predict_label(x))


if __name__ == '__main__':     
    app.run(host='0.0.0.0')
