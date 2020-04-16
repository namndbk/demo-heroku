import numpy as np
import string
import unidecode


# accented_chars_vietnamese = [
#     'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
#     'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
#     'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
#     'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
#     'í', 'ì', 'ỉ', 'ĩ', 'ị',
#     'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
#     'đ',
# ]
# accented_chars_vietnamese.extend([c.upper() for c in accented_chars_vietnamese])
# alphabet = list(('\x00 _' + string.ascii_letters + string.digits + ''.join(accented_chars_vietnamese)))

import pickle

alphabet = pickle.load(open("a.pkl", "rb"))

MAXLEN = 30

def encode(text, maxlen=MAXLEN):
    text = "\x00" + text
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        x[i, alphabet.index(c)] = 1
    if i < maxlen - 1:
        for j in range(i+1, maxlen):
            x[j, 0] = 1
    return x


def decode(x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)


def generate_data(data): 
    x = []
    y = []
    for i, c in enumerate(data):
        try:
            y.append(encode(data[i]))
            x.append(encode(unidecode.unidecode(data[i])))
        except:
            break
    return np.array(x), np.array(y)

