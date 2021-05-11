import sys

sys.path.append('C:/Users/myunghoon_k/OneDrive - 서울시립대학교/bitamin/dl_nlp_study')

from mh_common.mh_layers import Affine, Sigmoid, SoftmaxWithLoss
import numpy as np

class TwoLayerNet: #가중치를 2개 가지는 DNN 생성 (은닉층이 1개)
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size #입력, 은닉, 출력층의 차원 결정

        #W, b 초기화
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        #계층 생성
        self.layers = [
            Affine(W1, b1), #입력층 -> 은닉층으로의 순전파
            Sigmoid(), #은닉층 내 활성함수
            Affine(W2, b2) #은닉층 0> 출력층으로의 순전파
        ]
        self.loss_layer = SoftmaxWithLoss()
        
        #학습 과정에서 생성되는 W(가중치), d(기울기)를 self.params, self.grads에 담는다
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x): #순전파를 수행
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x) #순전파로 선형변환된 행렬
        loss = self.loss_layer.forward(score, t) #선형변환된 행렬을 바탕으로 손실값 생성
        return loss

    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout