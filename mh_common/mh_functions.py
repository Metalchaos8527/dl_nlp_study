import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim >= 2:
        x = x - x.max(axis = 1, keepdims = True)
        x = np.exp(x)
        x /= x.sum(axis = 1, keepdims = True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y, t):
    #y는 은닉층들의 계산을 통해 생성된 어떤 값이고
    #t는 실제 정답 데이터를 말한다
    if y.ndim == 1: 
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size: #t가 원핫벡터 형식일 때
        t = t.argamx(axis = 1) #argmax는 axis기준 가장 큰 값의 인덱스를 반환하는 메서드


    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size




