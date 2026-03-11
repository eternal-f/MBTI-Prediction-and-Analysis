import pandas as pd
import re
import random

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def random_select(data):
    result = []
    random_numbers = random.sample(range(0, len(data)), 2000)
    for i in random_numbers:
        result.append(data[i])
    return result

if __name__ == '__main__':
    path = ('../Data/data_cleaned.csv')
    text, label = load_data(path)
    data_list = []
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []
    data_6 = []
    data_7 = []
    data_8 = []
    data_9 = []
    data_10 = []
    data_11 = []
    data_12 = []
    data_13 = []
    data_14 = []
    data_15 = []
    data_16 = []
    for idx, value in enumerate(label):
        if value == 'infp':
            data_1.append(text[idx])
        if value == 'infj':
            data_2.append(text[idx])
        if value == 'intj':
            data_3.append(text[idx])
        if value == 'intp':
            data_4.append(text[idx])
        if value == 'enfp':
            data_5.append(text[idx])
        if value == 'enfj':
            data_6.append(text[idx])
        if value == 'entj':
            data_7.append(text[idx])
        if value == 'entp':
            data_8.append(text[idx])
        if value == 'isfp':
            data_9.append(text[idx])
        if value == 'isfj':
            data_10.append(text[idx])
        if value == 'istj':
            data_11.append(text[idx])
        if value == 'istp':
            data_12.append(text[idx])
        if value == 'esfp':
            data_13.append(text[idx])
        if value == 'esfj':
            data_14.append(text[idx])
        if value == 'estj':
            data_15.append(text[idx])
        if value == 'estp':
            data_16.append(text[idx])
    data_1n = random_select(data_1)
    data_2n = random_select(data_2)
    data_3n = random_select(data_3)
    data_4n = random_select(data_4)
    data_5n = random_select(data_5)
    data_6n = random_select(data_6)
    data_7n = random_select(data_7)
    data_8n = random_select(data_8)
    data_9n = random_select(data_9)
    data_10n = random_select(data_10)
    data_11n = random_select(data_11)
    data_12n = random_select(data_12)
    data_13n = random_select(data_13)
    data_14n = random_select(data_14)
    data_15n = random_select(data_15)
    data_16n = random_select(data_16)
    for value in data_1n:
        data_list.append({'text': value, 'label': 'infp'})
    for value in data_2n:
        data_list.append({'text': value, 'label': 'infj'})
    for value in data_3n:
        data_list.append({'text': value, 'label': 'intj'})
    for value in data_4n:
        data_list.append({'text': value, 'label': 'intp'})
    for value in data_5n:
        data_list.append({'text': value, 'label': 'enfp'})
    for value in data_6n:
        data_list.append({'text': value, 'label': 'enfj'})
    for value in data_7n:
        data_list.append({'text': value, 'label': 'entj'})
    for value in data_8n:
        data_list.append({'text': value, 'label': 'entp'})
    for value in data_9n:
        data_list.append({'text': value, 'label': 'isfp'})
    for value in data_10n:
        data_list.append({'text': value, 'label': 'isfj'})
    for value in data_11n:
        data_list.append({'text': value, 'label': 'istj'})
    for value in data_12n:
        data_list.append({'text': value, 'label': 'istp'})
    for value in data_13n:
        data_list.append({'text': value, 'label': 'esfp'})
    for value in data_14n:
        data_list.append({'text': value, 'label': 'esfj'})
    for value in data_15n:
        data_list.append({'text': value, 'label': 'estj'})
    for value in data_16n:
        data_list.append({'text': value, 'label': 'estp'})
    df = pd.DataFrame(data_list)
    df.to_csv('../Data/data_train.csv', index=False, encoding='utf-8-sig')
    df.to_csv('../Data/data_predict.csv', index=False, encoding='utf-8-sig')
