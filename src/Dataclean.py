import re
import pandas as pd
import emoji

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def remove_at(text):
    pattern = r'@\S+'
    return re.sub(pattern, '', text)

def remove_http_links(text):
    pattern = r'https?://\S+'
    return re.sub(pattern, '', text)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F1E0-\U0001F1FF"  # 国旗符号
        "\U00002702-\U000027B0"  # 装饰符号
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642" 
        "\u2600-\u2B55"
        "\u200d"  # 零宽连接符
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # 变体选择器
        "\u3030" # 封闭字符
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_sharp(text):
    pattern = r'#'
    return re.sub(pattern, '', text)

def remove_alb(text):
    return re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)

def remove_check(text):
    pattern = r'// automatically checked by'
    return re.sub(pattern, '', text)

def remove_emoji_using_lib(text):
    return emoji.replace_emoji(text, replace='')

def blank_norm(text):
    pattern = r' +'
    return re.sub(pattern, ' ', text)

def enter_norm(text):
    pattern = r'\s+'
    return re.sub(pattern, ' ', text)

if __name__ == '__main__':
    path = '../Data/twitter_MBTI.csv'
    text, label = load_data(path)
    data_list = []
    count = 0
    print(text[0])
    for idx, value in enumerate(text):
        value = value.split('|||')
        for i in range(len(value)):
            value[i] = remove_at(value[i])
            value[i] = remove_http_links(value[i])
            value[i] = remove_emojis(value[i])
            value[i] = blank_norm(value[i])
            value[i] = enter_norm(value[i])
            value[i] = remove_emoji_using_lib(value[i])
            value[i] = remove_sharp(value[i])
            value[i] = remove_alb(value[i])
            value[i] = remove_check(value[i])
            if len(value[i]) != 0:
                if value[i][0] == ' ':
                    value[i] = value[i][1:]
            if len(value[i]) != 0 and len(value[i]) >= 30:
                data_list.append({'text': value[i], 'label': label[idx]})
                print(count)
                count += 1
    print('编制结束')
    df = pd.DataFrame(data_list)
    df.to_csv('../Data/data_cleaned.csv', index=False, encoding='utf-8-sig')
