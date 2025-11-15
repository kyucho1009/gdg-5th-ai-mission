# Mission Course Week1

안녕하세요! AI 프로젝트 미션 트랙에 오신 것을 환영합니다!

저희는 앞으로 **4주간의 미션 코스**를 통해, 미니 프로젝트 **“AI 감정 분석 프로젝트”** 를 함께 완성해 나갈 예정이에요.

이번 **Week 1**에서는 인공지능이 텍스트 데이터를 이해할 수 있도록 만드는 과정, 즉 **데이터 전처리(Preprocessing)** 를 배웁니다.

AI 모델은 사람이 쓰는 문장을 그대로 이해하지 못합니다. 그래서 문장을 숫자로 바꾸는 “언어의 번역기” 역할을 하는 **전처리 과정**을 거쳐야 해요.

이번 주차에서는 이 과정을 단계별 코드 실습을 통해 직접 경험해보아요!

### 학습 목표

- **텍스트 데이터의 구조 이해:**
  네이버 영화 리뷰(NSMC) 데이터를 분석해 `id`, `document`, `label`의 의미를 파악합니다.
- **데이터 분리:**
  전체 데이터에서 **입력(X)** 과 **정답(Y)** 을 분리하여, **지도학습(Supervised Learning)** 의 기본 구조를 익힙니다.
- **전처리 파이프라인 구축:**
  `TextVectorization` 레이어를 사용하여, 문장을 **숫자 시퀀스(벡터)** 로 변환하고, 모델이 처리할 수 있도록 **패딩(Padding)** 과 **어휘 사전(Vocabulary)** 을 구성합니다.
- **데이터 시각화:**
  `Matplotlib`을 활용해 문장 길이 분포를 시각화하고, 전처리 설정(`MAX_LEN`)이 데이터 특성과 잘 맞는지 확인합니다.

### 미션 내용

```python
# 라이브러리 import 및 TensorFlow 버전 확인

import tensorflow as tf
import numpy as np
import pandas as pd

print("TensorFlow version:", tf.__version__)
```

실습을 위한 기본 라이브러리(**Numpy, Pandas, TensorFlow**)를 불러오고 현재 TensorFlow 버전을 출력하는 과정입니다. Colab 환경에서 실습을 준비하기 위한 첫 단계입니다.

[ ] 기본 라이브러리(**Numpy, Pandas, TensorFlow**) 개념에 대해서 정리해주세요.

```python
# 데이터 다운로드
path_to_train_file = tf.keras.utils.get_file(
    'train.txt',
    'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt'
)
path_to_test_file = tf.keras.utils.get_file(
    'test.txt',
    'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt'
)
```

TensorFlow의 `keras.utils` 모듈을 이용해 네이버 영화 리뷰(NSMC) 데이터를 자동으로 다운로드하는 단계입니다.

[ ] **Keras**가 무엇인지 알아보고, **TensorFlow 와 Keras의 역할 관계**에 대해서 정리해주세요.

```python
# IN - 텍스트로 로드
path_to_train_file = tf.keras.utils.get_file(
    'train.txt',
    'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt'
)
path_to_test_file = tf.keras.utils.get_file(
    'test.txt',
    'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt'
)

train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
test_text = open(path_to_test_file, 'rb').read().decode(encoding='utf-8')

print('Length of train text: {} characters'.format(len(train_text)))
print('Length of test text: {} characters'.format(len(test_text)))
print(train_text[:300])
```

```python
# OUT
Length of train text: 6937271 characters
Length of test text: 2318260 characters
id	document	label
9976970	아 더빙.. 진짜 짜증나네요 목소리	0
3819312	흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나	1
10265843	너무재밓었다그래서보는것을추천한다	0
9045019	교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정	0
6483659	사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다	1
5403919	막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.	0
7797314	원작의
```

다운로드한 데이터를 실제 텍스트로 읽어오는 과정입니다. `rb`는 파일을 바이너리 모드로 읽겠다는 뜻이고, `.decode('utf-8')` 은 한국어가 깨지지 않도록 UTF-8 인코딩으로 변환하는 단계입니다. 처음 300자만 출력하여 결과를 보면, **데이터가 `id`, `document`, `label`형태** (9976970, 아 더빙.. 진짜 짜증나네요 목소리, 0)로 구성되어져 있는 것을 볼 수 있습니다.

[ ] **훈련 데이터(train)** 와 **테스트 데이터(test)** 의 개념을 구분하여 정리해주세요.

[ ] **과적합(overfitting)** 이란 무엇이며, 훈련 데이터와 테스트 데이터의 관계에서 어떤 문제가 발생하는지 설명해주세요.

```python
# IN - Y(정답 라벨) 데이터 만들기
train_Y = np.array([
    [int(row.split('\t')[2])]
    for row in train_text.split('\n')[1:]
    if row.count('\t') > 0
])

test_Y = np.array([
    [int(row.split('\t')[2])]
    for row in test_text.split('\n')[1:]
    if row.count('\t') > 0
])

print("train_Y shape:", train_Y.shape)
print("test_Y shape:", test_Y.shape)
print("train_Y sample:", train_Y[:5])
```

```python
# OUT
train_Y shape: (150000, 1)
test_Y shape: (50000, 1)
train_Y sample: [[0]
 [1]
 [0]
 [0]
 [1]]
```

이 코드는 전체 데이터에서 **정답 라벨(label)** 만 분리하는 과정입니다. 세 번째 항목(`[2]`)이 감정 라벨로, [0]은 **부정 리뷰**, [1]은 **긍정 리뷰**를 의미합니다. 따라서 `split('\t')[2]`를 이용해 세 번째 값인 label을 추출하고, 정수형으로 변환하여 NumPy 배열 형태로 저장합니다. 이렇게 만든 `train_Y` , `test_Y` 는 나중에 모델이 예측할 **정답(타깃)** 역할을 합니다.

```python
# IN - X(입력 문장) 데이터 추출
train_text_X = [
    row.split('\t')[1]
    for row in train_text.split('\n')[1:]
    if row.count('\t') > 0
]

test_text_X = [
    row.split('\t')[1]
    for row in test_text.split('\n')[1:]
    if row.count('\t') > 0
]

print(train_text_X[:5])
```

```python
# OUT
[['아', '더빙', '진짜', '짜증나네요', '목소리'], ['흠', '포스터보고', '초딩영화줄', '오버연기조', '가볍지', '않구나'], ['너무재밓었'], ['교도소', '이야기구먼', '솔직히', '재미는', '없다', '평점', '조정'], ['사이몬페그', '익살스런', '연기가', '돋보였던', '영화', '!', '스파이더맨', '늙어보이기', '했던', '커스틴', '던스트가', '너무나도', '이뻐보였다']]
```

이 코드는 전체 데이터에서 **리뷰 문장(document)** 만 분리하는 과정입니다. 데이터는 `id`, `document`, `label` 형태로 되어 있기 때문에, `split('\t')[1]`을 이용하여 두 번째 항목 리뷰 문장만 추출합니다. 이렇게 분리된 문장은 모델 학습 시 입력 데이터(`train_X`)로 사용됩니다.

[ ] **지도학습(Supervised Learning)** 의 개념을 구조(X, Y)와 함께 정리해주세요.

[ ] **지도학습과 비지도 학습의 차이점**을 적어주세요.

```python
VOCAB_SIZE = 2000   # 단어 사전 크기
MAX_LEN = 25        # 최대 문장 길이 (패딩 기준)

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',  # 소문자 변환 + 구두점 제거
    split='whitespace',                         # 띄어쓰기 기준 토큰화
    max_tokens=VOCAB_SIZE,                      # 단어 사전 크기
    output_mode='int',                          # 정수 인코딩
    output_sequence_length=MAX_LEN              # 자동 패딩
)

vectorize_layer.adapt(train_text_X)  # 단어 사전 학습
# 텍스트를 정수 시퀀스로 변환 (패딩 포함)
train_X = vectorize_layer(train_text_X)

print(train_X[:5])
```

```python
# OUT
tf.Tensor(
[[  23  902    5    1 1097    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0]
 [ 586    1    1    1    1    1    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0]
 [   1    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0]
 [   1    1   68  345   28   33    1    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0]
 [   1    1  102    1    2    1    1  844    1    1  570    1    0    0
     0    0    0    0    0    0    0    0    0    0    0]], shape=(5, 25), dtype=int64)
```

텍스트 데이터를 **숫자 시퀀스로 변환**하는 **전처리** 과정입니다.

- `TextVectorization` 레이어는 입력 문장을 받아, **소문자로 변환**하고 **구두점을 제거**하며, **띄어쓰기 기준**으로 단어를 나눈 뒤, 각 단어를 **정수 인덱스**로 바꿉니다. (예를 들자면, {"이": 2, "영화": 5, "정말": 9, "재미있다": 11} 라면, `"이 영화 정말 재미있다"` → `[2, 5, 9, 11]` 로 벡터화가 이루어집니다.)
- `VOCAB_SIZE`는 **모델이 기억할 단어 개수(사전 크기**)를 정하는 값입니다. 즉, 등장 빈도가 높은 상위 단어만 학습에 사용합니다.
- `MAX_LEN`은 **문장의 최대 길이**를 제한해, 짧은 문장은 **0으로 채우고(패딩)** 긴 문장은 **일정 길이에서 자르는 역할**을 합니다. 이를 통해 모든 입력 문장의 길이를 통일시켜 모델이 처리하기 쉽게 만듭니다.
- `adapt(train_text_X)`를 통해 훈련 데이터에서 단어 빈도를 학습하고, 그 결과 만들어진 단어 사전(vocabulary)을 이용해 문장을 **숫자 배열로 변환**합니다.

출력된 `train_X`는 **Tensor** 형태로 **전처리 결과**입니다.

[ ] `TextVectorization`의 개념과 함께 **전처리(Preprocessing)** 의 개념을 정리해주세요.

[ ] **벡터화(Vectorization)** 가 왜 필요한지 정리해주세요.

[ ] **패딩(Padding)** 의 역할을 정리해주세요.

```python
import matplotlib.pyplot as plt
sentence_len = [len(sentence) for sentence in sentences]
sentence_len.sort()
plt.plot(sentence_len)
plt.show()

print(sum([int(l<=25) for l in sentence_len]))
```

우리가 데이터 전처리를 하면서 단어를 `[:25]`로 제한을 걸었던 설정이 실제로 데이터의 대부분을 커버하는지를 **Matplotlib**으로 시각화하여 확인하는 과정입니다.

- `len(sentence)` : 각 문장이 가진 **단어 수(length)** 를 구합니다.
- `sentence_len.sort()` : 길이를 오름차순으로 정렬해 그래프를 그릴 준비를 합니다.
- `plt.plot(sentence_len)` : 문장 길이의 변화를 시각적으로 표현하여, 데이터셋의 **전반적인 문장 길이 분포**를 한눈에 볼 수 있습니다.
- `print(sum([int(l <= 25) for l in sentence_len]))` : 전체 문장 중 **단어 수가 25개 이하인 문장 개수**를 출력합니다.

문장 길이를 오름차순으로 정렬해 시각화한 결과, 대부분의 문장이 25단어 이하임을 확인할 수 있습니다.

[ ] 또 다른 대표적인 파이썬 라이브러리인 **Matplotlib** 에 대해서 정리해주세요.

#### TODO

[ ] 각 코드 밑에 있는 미션을 블로그글로 정리해주세요.

[ ] 300자 이내 WIL 작성하기

[ ] 코드를 보지 않고 실습하며, .ipynb형태로 github에 올려주세요.
