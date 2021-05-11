import sys
sys.path.append('C:/Users/myunghoon_k/OneDrive - 서울시립대학교/bitamin/dl_nlp_study')
import numpy as np
import time
import matplotlib.pyplot as plt

from mh_common.mh_utils import clip_grads

#중복되는 가중치를 효율적으로 처리하기 위한 함수
#0928 기준 완벽히 이해하지 못했으므로 나중에 다시 들쳐볼것
def remove_duplicate(params, grads):

    params, grads = params[:], grads[:]

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i+1, L):
                # i번째 매개변수가 i + 1부터 마지막 인덱스 데이터들을 모두 찾아서
                # 동일한 값이 존재한다면 다음과 같이 처리를 하여라
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j) #중복되는 매개변수를 삭제
                    grads.pop(j)

                #가중치를 전치행렬로 공유하는 경우에는 다음과 같이 처리하여라
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].T.shape and np.all(params[i].T == params[j].T):
                    grads[i] += grads[j].T 
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                
                if find_flg: break
            if find_flg: break
        
        if not find_flg: break

    return params, grads
                    


#모델 학습 및 결과 출력 전 과정을 해주는 클래스 
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch = 10, batch_size = 32, max_grad = None, eval_interval = 20):
        data_size = len(x) #케이스 수
        max_iters = data_size // batch_size #1 epoch 당 미니배치의 학습횟수
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            #데이터를 랜덤하게 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size] #batch_size로 지정된 수만큼 데이터를 뽑아서 미니 배치 생성
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                
                #순전파, 역전파 
                loss = model.forward(batch_x, batch_t)
                model.backward()

                #매개변수의 중복값 및 정규화 작업 추가 수행
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad) #max_grad가 입력된 경우에만 기울기 정규화 작업을 수행

                #매개변수 업데이트
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| epoch %d | mini-batch %d / %d | 시간 %d[s] |avg_loss %.2f'
                    % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0 #다음 epoch로 넘어가기 전에 loss를 초기화

            self.current_epoch += 1

    def plot(self, ylim = None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label = 'train')
        plt.xlabel('epoch (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()




