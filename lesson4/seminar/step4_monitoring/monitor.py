import requests

def health(api_url, timeout):
    '''
    Мониторинг HEALTH.
    '''

    try:
        response = requests.get(f'{api_url}/health', timeout=timeout)
        status = response.status_code

        data = response.json() if response.ok else response.text
        return data

    except Exception as e:
        print(f'Ошибка запроса: {e}')

def predict(api_url, img_path, timeout):
    '''
    Мониторинг PREDICT.
    '''

    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path, f, "image/jpeg")}
            response = requests.post(f'{api_url}/predict', files=files)
            data = response.json() if response.ok else response.text
            return data

    except Exception as e:
        print(f'Ошибка инференса: {e}')