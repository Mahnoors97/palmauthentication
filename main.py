from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    
    return "Welcome to Contact Less PALM Authentication"

if __name__ == '__main__':
    app.run(host='0.0.0.0')