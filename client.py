from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('data/699_epo_1.jpeg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpeg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print('finished')
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}