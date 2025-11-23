# AI 미션코스 Week3

Week2에서는 벡터화된 데이터를 활용해 **기본 신경망 모델로 감정 분석을 학습**해보며, 손실과 정확도의 변화를 관찰했습니다. 하지만 해당 모델은 단어의 **순서나 문맥**을 제대로 반영하지 못한다는 한계가 있습니다. 그래서 Week3에서는 문장의 흐름을 이해하는 **RNN과 LSTM 같은 순환 신경망 모델**을 직접 학습합니다. 이를 통해 단어 사이의 관계와 문맥을 반영하는 **더 깊은 자연어 처리 모델링**을 경험해봅니다.

### 3주차의 학습 목표

- Week2에서 벡터화된 텍스트 데이터를 바탕으로, **Simple RNN과 LSTM 모델을 직접 구현하고 학습**해봅니다.
- 두 모델이 문장의 **순서(Sequence)** 와 **흐름(Context)** 을 어떻게 이해하는지 비교합니다.
- Simple RNN과 LSTM의 구조적 차이가 **학습 과정과 성능에 어떤 영향을 주는지** 확인합니다.

### Mission1

Week2에서는 Dense 기반 모델을 활용해 기본적인 신경망 분류기를 만들어 보았습니다. 하지만 Dense 기반 모델에는 한 가지 중요한 한계가 있습니다. 바로 **문장의 단어 순서를 반영하지 못한다는 점**입니다.

예를 들어, “배우 연기는 좋았지만 스토리가 아쉬웠다”와 “스토리는 아쉬웠지만 배우 연기는 좋았다”는 거의 같은 단어를 사용하지만, 순서가 바뀌는 순간 **평가의 강조점과 전체 의미가 달라집니다**. 이러한 차이를 이해하려면 **단어가 어떤 순서로 등장했는지**가 매우 중요합니다.

이런 한계를 보완하기 위해 사용하는 것이 바로 **Simple RNN**입니다.

```python
from tensorflow.keras.layers import SimpleRNN
import random

EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 32

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LEN,)),
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN
    ),
    SimpleRNN(units=50), #RNN 사용
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

rnn_model.summary()
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

RNN 모델은 Dense 기반 모델보다 **가중치 초기화와 내부 상태 초기값**에 훨씬 민감합니다. 그래서 동일한 코드라도 실행할 때마다 결과가 조금씩 다르게 나오는 경우가 많습니다.

이를 방지하고, 실습 결과를 안정적으로 재현하기 위해 `random.seed`, `np.random.seed`, `tf.random.set_seed`를 모두 지정해 랜덤성을 고정해줍니다.

random seed를 고정한 뒤에는, 입력층(Input), Embedding, Simple RNN, Dense 층을 차례로 쌓아 RNN 기반 감정 분석 모델을 구성합니다.

- [ ] RNN(Recurrent Neural Network)이 어떻게 순서 정보를 반영할 수 있는지, 내부 구조와 함께 동작 원리를 정리해주세요.

```python
history_rnn = rnn_model.fit(
	train_X, train_Y,
	epochs=EPOCHS,
	batch_size=BATCH_SIZE,
	validation_split=0.2,
	verbose=1
)
```

`rnn_model`을 week2 과 동일한 하이퍼파라미터 조건으로 학습시켜줍니다.

```python
test_loss, test_acc = rnn_model.evaluate(test_X, test_Y, verbose=0)
print(f"Rnn Test Loss: {test_loss:.4f}, Rnn Test Accuracy: {test_acc:.4f}")
```

학습에 사용되지 않은 테스트 데이터를 이용해 `rnn_model`의 최종 성능을 평가합니다. Week2에서 학습했던 Dense 기반 모델과 비교하여, 순서 정보를 반영하는 RNN 모델이 어떤 성능 차이를 보이는지 확인해주세요. (Test Loss: 0.5171, Test Accuracy: 0.7325 → Rnn Test Loss: 0.5143, Rnn Test Accuracy: 0.7345)

```python
example_sentences = [
    "배우 연기는 좋았지만 스토리가 아쉬웠다",
    "스토리는 아쉬웠지만 배우 연기는 좋았다"
]

example_seq = vectorize_layer(example_sentences)
pred = rnn_model.predict(example_seq)

for s, p in zip(example_sentences, pred):
    print(f"문장: {s}")
    print(f"긍정 확률: {p[0]:.4f}")
    print("결과:", "긍정 😊" if p[0] > 0.5 else "부정 😞")
```

```python
# OUT
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 209ms/step
문장: 배우 연기는 좋았지만 스토리가 아쉬웠다
긍정 확률: 0.2723
결과: 부정 😞
문장: 스토리는 아쉬웠지만 배우 연기는 좋았다
긍정 확률: 0.9295
결과: 긍정 😊
```

먼저 `rnn_model`을 이용하여,

> "배우 연기는 좋았지만 스토리가 아쉬웠다"
> "스토리는 아쉬웠지만 배우 연기는 좋았다"

이 두 문장을 평가해보았습니다. 부정인 리뷰는 긍정확률이 0.2723이라는 수치가 나왔고, 긍정인 리뷰는 긍정확률이 0.9295 라는 수치가 나왔습니다.

```python
example_sentences = [
    "배우연기는 좋았지만 스토리가 아쉬웠다",
    "스토리는 아쉬웠지만 배우 연기는 좋았다"
]

example_seq = vectorize_layer(example_sentences)
pred = model.predict(example_seq)

for s, p in zip(example_sentences, pred):
    print(f"문장: {s}")
    print(f"긍정 확률: {p[0]:.4f}")
    print("결과:", "긍정 😊" if p[0] > 0.5 else "부정 😞")
```

```python
# OUT
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step
문장: 배우연기는 좋았지만 스토리가 아쉬웠다
긍정 확률: 0.0612
결과: 부정 😞
문장: 스토리는 아쉬웠지만 배우 연기는 좋았다
긍정 확률: 0.7308
결과: 긍정 😊
```

이 코드는 week2에서 dense를 기반으로 만든 모델로 동일 문장을 평가해보았습니다. 위와 동일한 문장을 예측한 결과, 부정인 리뷰는 긍정확률이 0.0612 라는 수치가 나왔고, 긍정인 리뷰는 긍정확률이 0.7308 라는 수치가 나왔습니다.

- [ ] `dense 기반 모델`과 `RNN 모델`의 예측 차이가 의미하는 바를 두 모델의 차이점과 함께 정리해주세요.

### Mission2

앞서 실습해본 RNN은 문장의 순서를 반영할 수 있지만, 여전히 **긴 문장에서 앞부분 정보를 빠르게 잊어버린다는 한계**가 있습니다.

예를 들어,

“**이 영화는 정말 재미있고, 스토리도 흥미진진하며 배우들의 연기까지 완벽해서 끝까지 눈을 뗄 수 없었어요.**”

“**영화가 전체적으로 너무 지루하고 전개가 느려서 몰입하기 힘들었으며, 기대했던 재미나 감동이 전혀 없었어요.**”

두 문장은 길고 복잡하며, 문장의 초반–중반–후반에 걸쳐 **긍정과 부정의 힌트가 시간 순서대로 연결**되어 있습니다. 하지만 RNN은 뒤로 갈수록 앞부분 정보를 잊어버리기 때문에, 이런 **긴 흐름 속 문맥 변화**를 온전히 반영하기 어렵습니다.

이러한 RNN의 한계를 보완해주는 모델이 바로 **LSTM(Long Short-Term Memory)** 입니다.

```python
# LSTM 모델 코드 실습

# [구현 요구사항]

# 1) 모델 변수명
#   - 반드시 lstm_model 이라는 이름으로 모델을 생성해주세요.

# 2) 순환 계층 변경
#   - SimpleRNN(units=50) → LSTM(units=50)

# [동일하게 유지되는 부분]

# - Input Layer             : shape=(MAX_LEN,)
# - Embedding Layer         : VOCAB_SIZE, EMBEDDING_DIM
# - Dense Layer 구조        : Dense(64, relu) + Dense(1, sigmoid)
# - 하이퍼파라미터           : 동일
# - random seed 설정        : 그대로 유지
# - compile 설정            : optimizer, loss, metrics 모두 동일
# - fit/evaluate 구조       : RNN과 동일한 방식으로 실행

# 최종적으로 lstm_model.evaluate(test_X, test_Y) 결과를 출력해주세요.
```

LSTM 모델을 구현 요구사항에 맞춰 직접 구현해보아요.

- [ ] RNN의 한계에 대해서 설명해주세요.
- [ ] LSTM (Long Short-Term Memory)이 어떻게 장기기억을 할 수 있는지, 동작 원리를 정리해주세요.

```python
# 아래 두 문장을 lstm_model에 넣어 예측 결과(긍정 확률 + 최종 분류)를 출력해주세요.

# "이 영화는 정말 재미있고, 스토리도 흥미진진하며 배우들의 연기까지 완벽해서 끝까지 눈을 뗄 수 없었었요."
# "영화가 전체적으로 너무 지루하고 전개가 느려서 몰입하기 힘들었으며, 기대했던 재미나 감동이 전혀 없었어요."
```

```python
# 아래 두 문장을 rnn_model에 넣어 예측 결과(긍정 확률 + 최종 분류)를 출력해주세요.

# "이 영화는 정말 재미있고, 스토리도 흥미진진하며 배우들의 연기까지 완벽해서 끝까지 눈을 뗄 수 없었었요."
# "영화가 전체적으로 너무 지루하고 전개가 느려서 몰입하기 힘들었으며, 기대했던 재미나 감동이 전혀 없었어요."
```

rnn_model과 lstm_model을 예시로 직접 비교해주세요.

- [ ] `RNN 모델`과 `LSTM 모델`의 예측 차이가 의미하는 바를 두 모델의 차이점과 함께 정리해주세요.

### TODO

- [ ] 각 코드 밑에 있는 미션을 블로그글로 정리해주세요.
- [ ] 300자 이상 WIL 작성하기
- [ ] 코드를 보지 않고 실습하여 `.ipynb`로 저장하여, github에 올려주세요.

### 파일 구조

```python
gdg-5th-ai-mission/
├── articles/
│   ├── assignment/
│   │   ├── week1/
│   │   │   ├── week1_mission.ipynb
│   │   │   └── week1.md
│   │   ├── week2/
│   │   │   ├── week2_mission.ipynb
│   │   │   └── week2.md
│   │   ├── week3/
│   │   │   ├── week3_mission.ipynb      # Week3 실습 코드 파일
│   │   │   └── week3.md
│   │
│   ├── week1/
│   │   └── WIL.md
│   ├── week2/
│   │   └── WIL.md
│   ├── week3/
│   │   └── WIL.md                      # Week3 WIL -> 링크만 첨부
│
└── README.md

```
