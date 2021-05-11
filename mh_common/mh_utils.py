import numpy as np

#기울기 폭발을 막기 위한 제약장치
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2) #음의 기울기 양수처리를 위해 제곱값을 취함
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


#텍스트 전처리 + 단어별 정수인덱싱 생성 + 단어id 리스트 반환 함수
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(id_to_word)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


#동시발생 행렬 생성
def create_co_matrix(corpus, vocab_size, window_size = 1):
    '''
    인자1 단어들이 모두 정수인덱싱으로 표현된 corpus를 사용
    인자2 word_to_id 혹은 id_to_word의 길이, 즉 유일하게 상용되는 단어들의 개수
    인자3 윈도우 크기, 기본값은1
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype= np.int32)

    for idx, word_id in enumerate(corpus): #corpus의 정수인덱스 출현 순서 및 정수인덱스 값을 사용
        for i in range(1, window_size + 1):
            #corpus 정수 인덱스 출현 순서 idx를 바탕으로
            #윈도우 크기에 따른 맥락 인덱스를 먼저 생성
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0: #좌측 맥락의 인덱스가 0이상인 경우에만 시행
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size: #우측 맥락의 인덱스가 corpus 길이보다 작은 경우에만 시행
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

#코사인 유사도
def cos_similarity(x,y, eps = 1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

#가장 비슷한 단어를 검색하여 출력
def most_similar(query, word_to_id, id_to_word, word_matrix, top = 5):
    '''
    query: 검색하고자 하는 단어
    word_to_id: preprocess 함수를 통해 생성된 단어를 키로 갖는 딕셔너리 객체
    id_to_word: preprocess 함수를 통해 생성된 정수인덱싱을 키로 갖는 딕셔너리 객체
    word_matirx: create_co_matrix를 통해 생성된 동시발생행렬
    top: 상위 몇 개까지 출력할지 지정하는 수
    '''

    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    #코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size) #query단어와 
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        #np.argsort()는 배열 내 오름차순 순서로 인덱스를 반환
        #본래 배열에 -를 취하면 배열을 내림차순 순서로 반환함
        if id_to_word[i] == query: 
            continue #검색단어와 동일한 경우에는 무시
        print(' %s: %s' % (id_to_word[i], similarity[i])) #query와 대응되는 단어들의 유사도를 출력

        count += 1
        if count >= top:
            return

#양의 상호정보량
def ppmi(C, verbose = False, eps = 1e-8):
    '''
    C: 동시발생 행렬
    verbose: 진행 상황 출력여부
    '''

    M = np.zeros_like(C, dtype = np.float32)
    #np.zeros는 사용자가 직접 행, 열의 개수를 입력하여서 0으로만 구성된 행렬을 생성하는 것이고
    #np.zeros_like는 인자로 np.array를 입력하여 해당 array의 동일한 shape로 구성된 0으로 구성된 행렬 반환
    N = np.sum(C) #동시발생행렬 내 원소들의 합 = 등장횟수 N
    S = np.sum(C, axis= 0) #동시발생행렬 내 각 단어들의 등장횟수 S
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i]*S[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))

    return M

def create_contexts_target(corpus, window_size = 1):

    target = corpus[window_size:-window_size] 
    #정답 데이터들은 윈도우 크기를 이용하여 인덱싱
    #윈도우 크기가 1인 경우에 1번째 인덱싱부터 정답데이터가 될 수 있고 
    #최종적으로 -1이 마지막 정답 데이터가 됨
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = [] #매번 맥락을 담을 리스트를 초기화
        for t in range(-window_size, window_size + 1):
            if t == 0: #target위치에 올 경우에는 건너뛰기
                continue
            cs.append(corpus[idx + t]) #맥락에 해당하는 단어들을 하나씩 cs에 쌓기
        contexts.append(cs) #cs에 담아진 맥락들을 하나씩 contexts에 쌓기

    return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size):

    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype = np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype = np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def to_cpu(x):
    import numpy as np
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)

def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)





