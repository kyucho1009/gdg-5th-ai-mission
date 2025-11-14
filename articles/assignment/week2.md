# Mission Course Week2

Week1에서 우리는 **텍스트 데이터를 인공지능이 이해할 수 있는 숫자 형태로 변환하는 준비 과정**을 마쳤습니다. 문장 속 단어들을 정제(cleaning)하고, 토큰화(tokenization)하여, 모델이 학습할 수 있도록 **정수 시퀀스(숫자 벡터)** 로 변환했습니다.

이제 Week2에서는, 그 데이터를 **실제로 인공지능 모델에 학습시키는 단계**로 넘어갑니다. 즉, “데이터를 이해시킬 준비”를 끝냈다면, 이번 주차는 “AI가 스스로 학습하며 패턴을 익히는 과정”을 직접 경험하는 주차입니다.

## 2주차의 학습 목표

- Week1에서 만든 데이터를 바탕으로 **인공지능 모델을 실제로 학습(fit)** 시켜봅니다.
- 학습 과정에서 **손실(Loss)** 과 **정확도(Accuracy)** 가 어떻게 변화하는지 관찰합니다.
- **검증 데이터(validation)** 를 통해 모델이 얼마나 일반화되어 있는지를 확인합니다.
- **Epoch 수를 늘려가며 과적합(Overfitting)** 이 어떻게 발생하는지를 직접 실험하고 이해합니다.
- 학습된 모델이 새로운 문장에서 감정을 어떻게 예측하는지도 확인해봅니다.

## Mission1

```python
test_text_X = [
    row.split('\t')[1]
    for row in test_text.split('\n')[1:]
    if row.count('\t') > 0
]

# Week1에서 학습된 vectorize_layer 그대로 사용합니다.
test_X = vectorize_layer(test_text_X)

print("test_X shape:", test_X.shape)
print("test_Y shape:", test_Y.shape)
```

Week1에서는 `train_text_X`로부터 **단어 사전(vocabulary)** 을 학습(`adapt`)하고, 그 사전을 사용해 `train_X`를 정수 시퀀스로 벡터화했습니다. Week2에서는 그 **동일한 사전(vectorize_layer)** 을 이용해 **`test` 데이터**를 같은 기준으로 벡터화합니다. 이렇게 해야 train과 test가 **동일한 인덱스 체계**를 사용하게 되어, 모델이 일관된 단어 인식 체계를 유지할 수 있습니다. 만약 test를 새로 `adapt()`시키면, “같은 단어인데 다른 번호로 인코딩되는 오류”가 발생할 수 있습니다.

```python
# 모델의 주요 설정값들을 미리 정의해줍니다.
VOCAB_SIZE = 2000
EMBEDDING_DIM = 128
MAX_LEN = 25
EPOCHS = 10
BATCH_SIZE = 32
```

이제 본격적으로 **감정 분석(Sentiment Analysis)** 을 수행할 수 있는 **인공지능 모델을 만들어보도록 하겠습니다.** 앞서 데이터를 전처리하고, 텍스트를 숫자로 바꾸는 벡터화 과정을 마쳤습니다. 이번 단계에서는 학습 과정 전반을 제어할 **하이퍼파라미터(Hyperparameters)** 를 설정하고, 이를 기반으로 모델의 구조를 설계해볼 것입니다.

- [ ] 하이퍼파라미터(Hyperparameter)에 대해서 정리해주세요.
- [ ] `VOCAB_SIZE` `EMBEDDING_DIM` `MAX_LEN` `EPOCHS` `BATCH_SIZE` 의 역할에 대해서 정리해주세요.
- [ ] `EPOCHS`를 늘리면 무조건 성능이 좋아질까요? **과적합(overfitting)** 과 함께 설명해주세요.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
```

신경망 구조(Neural Network Architecture)를 정의하는 부분입니다. 인공지능이 입력 데이터를 받아들이고, 내부에서 의미를 학습하며, 최종적으로 결과를 예측할 수 있도록 **신경망의 전체 흐름**(입력층 → 은닉층 → 출력층) 을 설계하는 단계입니다.

- [ ] Embedding이란 무엇인지 정리해보아요.
- [ ] Pooling이란 무엇인지 정리해보아요.
- [ ] 선형함수와 비선형함수를 비교해보고, 비선형함수를 활성화함수(`relu`, `sigmoid`)와 관련시켜 정리해보아요.
- [ ] Dense Layer의 역할에 대해서 설명해주세요.
- [ ] 위 코드에서 입력층, 은닉층, 출력층에 해당하는 코드 옆에 주석을 달아주세요.

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

모델이 학습할 때 어떤 방식으로 학습할지(최적화 알고리즘), 무엇을 기준으로 성능을 평가할지(손실 함수, 평가 지표) 정의하는 부분입니다.

- [ ] Optimizer(최적화 함수)에 대해서 정리해주세요.
- [ ] Loss Function(손실 함수)에 대해서 정리해주세요.
- [ ] Metrics(평가지표)에 대해서 정리해주세요.

```python
history = model.fit(
    train_X, train_Y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)
```

train data를 이용해 모델을 학습시킵니다. 전체 데이터를 10 epoch 동안 반복 학습하며, `validation_split=0.2`로 지정된 검증 데이터는 train data의 20%를 검증용으로 쓰는 부분입니다. 이를 통해, 모델이 과적합되는 징후를 훈련 중 실시간으로 확인할 수 있습니다.

```python
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
```

훈련에 사용되지 않은 test data를 이용해 모델의 일반화 성능을 평가합니다.

`evaluate()` 함수는

- 손실값(`test_loss`): 모델이 예측을 얼마나 잘못했는지
- 정확도(`test_acc`): 올바르게 맞춘 비율
  을 계산해줍니다.

```python
example_sentences = [
    "이 영화 진짜 재미있어요",
    "완전 지루하고 별로였음",
    "배우 연기는 좋았지만 스토리가 아쉬웠다"
]

example_seq = vectorize_layer(example_sentences)
pred = model.predict(example_seq)

for s, p in zip(example_sentences, pred):
    print(f"문장: {s}")
    print(f"긍정 확률: {p[0]:.4f}")
    print("결과:", "긍정 😊" if p[0] > 0.5 else "부정 😞")
```

학습된 모델이 실제 문장을 입력받아 감정을 예측하는 과정입니다. `vectorize_layer`를 통해 문장을 숫자 시퀀스로 변환한 뒤, `model.predict()`를 이용해 각 문장의 **긍정 확률**을 계산합니다. 출력값이 0.5보다 크면 긍정으로, 작으면 부정으로 분류합니다. 이를 통해 모델이 새로운 문장에서도 학습된 감정 패턴을 얼마나 잘 적용하는지를 확인할 수 있습니다.

## Mission2

**Epoch 수를 늘렸을 때 발생하는 과적합(Overfitting)** 현상을 직접 확인해보아요. 기존에는 **`EPOCHS = 10`** 으로 설정하여 학습을 진행했지만, 이번에는 **`Epoch = 30`으로 늘려** 학습을 오래 지속했을 때 모델의 변화가 어떻게 달라지는지를 비교해보아요. **훈련 손실(train loss)** 과 **검증 손실(val loss)**, 그리고 **훈련 정확도(train accuracy)** 와 **검증 정확도(val accuracy)** 의 변화를 비교하여 시각화해주세요.

```python
# 실습 미션 코드를 작성해주세요.
# 1. Epoch=30 으로 늘려 모델 학습
# 2. matplotlib를 이용하여 Epoch 10 과 Epoch 30을 비교하여 시각화





```

- [ ] 시각화된 그래프를 바탕으로, **과적합(Overfitting)이 왜 발생했는지** 분석하여 정리해주세요.

### TODO

- [ ] 각 코드 밑에 있는 미션을 블로그글로 정리해주세요.
- [ ] 300자 이상 WIL 작성하기
- [ ] 코드를 보지 않고 실습하여 `.ipynb`로 저장하여, github에 올려주세요.

### 파일 구조

```
gdg-5th-ai-mission/
├── articles/
│   ├── assignment/
│   │   ├── week1/
│   │   │   ├── week1_mission.ipynb
│   │   │   └── week1.md
│   │   ├── week2/
│   │   │   ├── week2_mission.ipynb   # 실습 코드 파일
│   │   │   └── week2.md
│   │
│   ├── week1/
│   │   └── WIL.md
│   ├── week2/
│   │   └── WIL.md                    # 개념 정리 블로그 글
│
└── README.md
```
