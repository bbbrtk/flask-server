from flask import Flask, request, Response
from pprint import pprint
import jsonpickle, json
import requests
import numpy as np
import cv2
import base64 
from src.BarkNetModel import BarkNet

app = Flask(__name__)
IMG_PATH = 'data/test.jpeg'
API_KEY = "2b10xnBrTMZN7ri4Vl8kMx424u"
api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={API_KEY}"


def encode_and_save(request):
    imgdata = base64.b64decode(request.data[23:])
    with open(IMG_PATH, 'wb') as f:
        f.write(imgdata)

    # nparr = np.fromstring(request.data, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imwrite(IMG_PATH, img)


def plantnet(organ='leaf'):
    img = open(IMG_PATH, 'rb')
    print('image opened')

    data = {
        'organs': ['leaf']
    }

    files = [
        ('images', (IMG_PATH, img))
    ]

    req = requests.Request('POST', url=api_endpoint, files=files, data=data)
    prepared = req.prepare()

    s = requests.Session()
    response = s.send(prepared)
    json_result = json.loads(response.text)

    pprint(response.status_code)
    
    return json_result

def barknet():
    f = 'data/699_epo_1.jpg'   
    barkNet1 = BarkNet()
    out = barkNet1.return_classification_array(f)
    print(out)
    return out


@app.route('/api/test', methods=['POST'])
def test():
    r = request
    print('r', r)

    encode_and_save(r)

    # fromapi = plantnet()
    # pprint(fromapi)
    # response = {'message': fromapi
    #             }

    frombark = barknet()

    response = {'message': frombark
                }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")



app.run(host="0.0.0.0", port=5000)