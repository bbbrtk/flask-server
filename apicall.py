import requests
import json
from pprint import pprint
import cv2

API_KEY = "2b10xnBrTMZN7ri4Vl8kMx424u"
api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={API_KEY}"


# image_path_1 = "../data/swierk.jpg"
# image_data_1 = open(image_path_1, 'rb')

image_path_2 = "data/im_04.png"
# image_path_2 = "/home/bartek/Documents/mgr/flask-server/data/test.jpeg"
# image_path_2 = "/home/bartek/Pictures/Screenshot from 2021-02-26 09-11-15.png"
img = open(image_path_2, 'rb')

# img = cv2.imread(image_path_2)

data = {
    'organs': ['leaf']
}

files = [
    ('images', (image_path_2, img))
]

req = requests.Request('POST', url=api_endpoint, files=files, data=data)
prepared = req.prepare()

s = requests.Session()
response = s.send(prepared)
json_result = json.loads(response.text)

pprint(response.status_code)
pprint(json_result)
