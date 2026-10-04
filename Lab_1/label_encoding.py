import numpy as np
from sklearn import preprocessing

# 1. Определяем метки
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# 2. Создаем кодировщик и обучаем его
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# 3. Выводим отображение слов на числа
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# 4. Преобразуем новый набор меток
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# 5. Декодируем случайный набор чисел
encoded_values_to_decode = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values_to_decode)
print("\nEncoded values =", encoded_values_to_decode)
print("Decoded labels =", list(decoded_list))