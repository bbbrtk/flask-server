from flask import Flask, request, Response
from PIL import Image
from pprint import pprint

import jsonpickle, json, requests, base64, cv2
import numpy as np

from BarkNetModel import BarkNet

app = Flask(__name__)

IMG_PATH = '../data/test2.jpeg'

API_KEY = "2b10xnBrTMZN7ri4Vl8kMx424u"
api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={API_KEY}"


def encode_and_save(request):
    organ = request.data.decode()[:4]
    print(organ)
    imgdata = base64.b64decode(request.data[28:])
    with open(IMG_PATH, 'wb') as f:
        f.write(imgdata)

    return organ
    # nparr = np.fromstring(request.data, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imwrite(IMG_PATH, img)

def convert_image(self, img):
    img = np.array(img)
    if len(img.shape) > 2 and img.shape[2] == 4:
        return Image.fromarray(img[...,:3])
    else: 
        return img

def plantnet(organ='leaf'):
    img = open(IMG_PATH, 'rb')
    print('image opened')

    data = {
        'organs': [organ]
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
    
    if response.status_code == 200:
        results_arr = json_result['results']
        return results_arr

    else:
        return []

def barknet():
    barkNet = BarkNet()
    out = barkNet.return_classification_array(IMG_PATH)
    print(out)
    return out


@app.route('/api/test', methods=['POST'])
def test():
    r = request
    organ = encode_and_save(r)

    frombark = []
    fromplant = []

    if organ == 'leaf':
        fromplant = plantnet(organ)
    elif organ == 'bark':
        fromplant = plantnet(organ)
        frombark = barknet()

    response = {
        'bark': frombark,
        'plant': fromplant
    }

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

app.run(host="0.0.0.0", port=5000)