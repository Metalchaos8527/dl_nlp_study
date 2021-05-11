import numpy as np
import sys
try:
    sys.path.append('C:/Users/kang_lp/OneDrive - UOS/bitamin/dl_nlp_study/mh_common')
except:
    sys.path.append('C:/Users/myunghoon_k/OneDrive - UOS/bitamin/dl_nlp_study/mh_common')

from mh_common.mh_layers import *
from mh_common.mh_functions import sigmoid

#RNN 계층 구현
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None #역전파 계산시 사용할 중간 데이터를 담을 객체

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next**2) 
        #여기서 (1 - h_next**2)는 tanh에 해당하는 h_next를 미분한 식
        db = np.sum(dt, axis = 0) #b가 repeat노드를 통해서 브로드캐스팅 되었으므로 합치는게 곧 역전파
        dWh = np.dot(h_prev.T, dt) #덧셈노드들을 통해서 흘러들어온 dt가 matmul계층의 역전파 h_prev의 전치행렬과 행렬곱
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx #덮어씌우는 작업을 통해 기울기 갱신
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

#T개의 RNN계층을 TimeRNN으로 구현
#TimeRNN의 입력은 x1,x2.. 같이 여러 벡터들의 배열로 이루어져있음
#output에 해당하는 h도 마찬가지

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful = False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful 
        #다음 TimeRNN계층에 현 TimeRNN의 마지막 output h를 넘길지 말지에 관한 옵션

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape 
        #텐서 형태, N은 미니배치 크기, T는 시계열 데이터 개수, D는 벡터차원
        #즉 입력 벡터 자체는 T*D 형태 
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype= 'f') #값이 없는 경우 0배열로 초기화
        
        for t in range(T):
            layer = RNN(*self.params) 
            #여기서 *self.params는 self.params를 unpacking하겠다는 의미
            #즉 Wx, Wh, b를 인자로 사용하여 RNN계층을 생성
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)): #역전파는 뒤의 RNN계층부터 해야하므로 reversed메서드 사용
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            #현재 계층의 역전파 시작은 dh_t와 다음노드의 RNN계층으로 가는 dh_next_t의 합으로 구성되므로
            #이 두 인자의 합이 parameter로 들어가야함
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad #개별 계층 내 grad 업데이트

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad #TimeRNN계층의 grad 업데이트
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

#Embedding 계층을 Time 형식으로 구현
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape #N은 배치 사이즈, T는 단어(시점)의 개수
        #t시점에 N개 만큼의 단어ID가 값으로 존재하는 행렬
        V, D = self.W.shape #D는 은닉층의 길이

        out = np.empty((N, T, D), dtype= 'f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout): #dout은 RNN 계층에서 받는 것
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad #업데이트
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape 
        #입력은 RNN을 거쳐서 생성된 행렬
        #D를 h라고 생각하면 이해가 쉬움, 즉 은닉층의 길이
        W, b = self.params

        rx = x.reshape(N*T, -1)
        #기존의 x는 텐서구조, 각 n배치, t시점에 존재하는 데이터들에 대한 행렬곱을 수행해야함
        #이를 효율화 하기 위해 텐서를 행렬 형태로 변환하는 작업 수행, reshape로 구현
        #따라서 (n*t) * d 형태의 행렬이 생성 
        out = np.dot(rx, W) + b 
        #여기서 W는 d * v 형태를 가짐, 여기서 v는 voc_size
        #각 단어들마다 softmax score를 softmax층에서 매겨야하므로 v가 되어야 함
        self.x = x
        return out.reshape(N, T, -1) #3차원 텐서로 재변환하여 출력

    def backward(self, dout): #SoftMax 계층에서 받은 dout
        x = self.x
        N, T, D = x.shape 
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis = 0) #repeat노드를 거쳐 나온것으므로 모두 더함
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1 #특정 라벨을 무시할 경우 사용자가 바꿔서 선언하면 됨

    def forward(self, xs, ts): #ts는 정답데이터
        N, T, V = xs.shape

        if ts.ndim == 3: #정답데이터 ts가 원 핫 인코딩 형태로 되어 있는 경우
            ts = ts.argmax(axis = 2) 
            #axis = 2는 voc_size에 해당, 즉 1인 인덱스만 추출
            #즉 데이터를 원핫인코딩이 안된 형태로 변환

        mask = (ts != self.ignore_label) #무시할 라벨이 아니면 1, 맞으면 0

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts]) #각 케이스의 1에 해당되는 ts 인덱스와 대응하여 log값을 취함, crossentropy를 구하기 위한 첫 작업
        ls *= mask #무시할 라벨만 0으로 곱해지고 나머지는 1로 곱해지므로 수치의 변화가 없음
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout = 1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1 #정답에 해당하는 인덱스에만 1을 빼주기
        dx *= dout
        dx /= mask.sum() #시점의 개수 T로 나누는 것과 동일한 작업
        dx *= mask[:, np.newaxis]

        dx = dx.reshape((N, T, V)) #Affine 게층의 출려과 size를 맞추기

        return dx

class LSTM:
    def __init__(self, Wx, Wh, b):

        '''
        Wx: f,i,g,o의 Xt에 종속되는 4가지 가중치들이 담김
        Wh: f,i,g,o의 ht에 종속되는 4가지 가중치들이 담김
        b : f,i,g,o의 각 케이스별 편향, 총 4가지가 담김
        '''

        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b #f,i,g,o의 모든케이스에 대해서 담은 결과

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(f)
        o = sigmoid(o)

        c_next = f * c_prev + g * i #elemnt-wise product를 하므로 *를 사용함
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        #ds에는 repeat노드를 통과한 c_t가 역전파에 관여하므로 dc_next와 그 아래의 흐름이 타고 와야함
        #여기서 (dh_next * o)는 마찬가지로 repeat노드를 통과한 h_t의 역전파로 생성된 dh_next가 *의 역전파로 o와 곱해지는 작업
        #(1 - tanh_c_next ** 2)는 tanh의 역전파 값 1- tanh ** 2

        dc_prev = ds * f #덧셈 노드는 역전파를 그냥 흘리므로 ds를 그대로 받음
        
        di = ds * g #곱셈 노드의 역전파는 반대편의 흘러들어온 미분과 반대편의 순전파를 곱함
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i) #시그모이드의 미분은 y(1-y)이므로
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g**2) #tanh의 미분은 (1 - y**2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis = 0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful = False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape #N 배치사이즈, T 차원 수(시점 개수), D wrod2vec 차원 수
        H = Wh.shape[0] #H는 은닉층 차원 수

        self.layers = []
        hs = np.empty((N, T, H), dtype = 'f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype= 'f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype= 'f')

        for t in range(T):
            layer =  LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype = 'f')
        dh, dc = 0, 0 #초기 역전파시 이전 시점의 미분이 들어오지 않는다는 의미

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc) #ht는 copy층을 지나기 때문에 역전파시 합치는 작업 필요
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

class TimeDropout:
    def __init__(self, dropout_ratio = 0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask

        else:
            return xs

    def backward(self, dout):
        return dout * self.mask

