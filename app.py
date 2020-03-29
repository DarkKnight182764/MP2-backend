from flask import Flask

app = Flask(__name__)
from flask import request
import json
from flask_cors import CORS, cross_origin
from model_run import predict
import os

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/lip-read', methods=['POST'])
@cross_origin()
def lip_read():
    print("request received")
    vid = request.files["video"]
    filepath = os.path.join(os.getcwd(), "video.mpg")
    vid.save(filepath)
    print("video saved at %s" % filepath)
    print("Processing video now--------")
    return {"pred": predict(filepath)}


if __name__ == '__main__':
    app.run(threaded=False)
