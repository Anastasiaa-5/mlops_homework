import yaml

def load_cfg(path):
    try:
        with open(path, 'r') as file:
            # Сохраняем данные в переменную
            data = yaml.safe_load(file)
        return data

    except FileNotFoundError:
        print(f"Ошибка: файл '{path}' не найден.")
    except yaml.YAMLError as e:
        print(f"Ошибка извлечения данных из YAML: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")