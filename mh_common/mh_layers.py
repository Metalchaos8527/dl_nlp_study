#시그모이드 계층
import sys
sys.path.append('C:/Users/myunghoon_k/OneDrive - 서울시립대학교/bitamin/dl_nlp_study')

import numpy as np
from mh_common.mh_functions import softmax, cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx       


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout): #dout은 dl / dy
        dx = dout * (1.0 - self.out) * self.out
        #dout은 dL / dy 값을 의미하고
        #self.out * (1 - self.out)은 dy / dx, 즉 시그모이드 미분 값을 말한다
        return dx

#선형변환 계층
#입력값들과 가중치들의 선형변환을 담당하는 계층
class Affine:
    def __init__(self, W, b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)] #가중치 초기값들을 모두 0으로 초기화
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b #Matmult + repeat node가 발생
        self.x = x
        return out

    def backward(self, dout): #dout은 dl / dy
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis = 0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size: #원핫 인코딩이 되어 있다면
            self.t = self.t.argmax(axis = 1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout = 1):
        #dout이 1인 이유는 맨 마지막에서 출발하는 흐름이기 때문에
        #맨 마지막 층의 미분값은 미분할 대상이 존재하지 않기 때문에 1로 출발
        #따라서 dout이 1로 주어진다
        #다른 층에서 사용 되는 dout은 모두 이 SoftmaxWithLoss의 dout을 받아서 사용되는 것
        batch_size = self.t.shape[0]

        dx = self.y.copy() #softmax 출력값을 받기
        dx[np.arange(batch_size), self.t] -= 1 
            #정답값 yi가 1이므로 예측값과 정답값 라벨이 동일한 경우에 -1을 빼줌으로써 backpropagation 수행
        dx *= dout #최종 backpropagation은 이전 미분값을 곱해야하므로 dout을 곱함
        dx = dx / batch_size #minibatch 방식이므로 평균적인 backpropagation 값을 구하기 위해 batch_szie로 나눔

        return dx

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        #for i, word_id in enumerate(self.idx):
        #    dW[word_id] += dout[i]

        np.add.at(dW, self.idx, dout)
        #첫번째 인자의 행렬 idx 위치에 세번째 인자의 값을 더하라
        #즉 dw[idx] + dout과 동일한 연산을 수행
        #다만 idx에 중복된 idx가 존재하는 경우는 평균을 취하여 원래의 인덱스의 값을 대체함
        return None

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx
              