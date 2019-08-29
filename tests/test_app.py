import requests
SERVER_URL = "http://127.0.0.1:5002/"
IMAGE_PATH = "./test_images/six.png"

def test_index():
    response = requests.request('GET', SERVER_URL)
    # check response
    assert response.status_code == 200
    assert response.data == b"Digiti Classifier App"

def test_api():
    # server REST API endpoint url and example image path

    # create payload with image for requests
    image = open(IMAGE_PATH, "rb")
    payload = {"file": image}
    response = requests.request('POST', SERVER_URL, files=payload)

if __name__ == '__main__':
    test_api()