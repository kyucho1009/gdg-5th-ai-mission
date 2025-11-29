# AI 미션코스 Week4

이전 단계에서는 영화 리뷰 텍스트를 정수 시퀀스로 변환하고, 이를 활용하여 기본 모델 & RNN & LSTM 기반의 감성 분류 모델을 구현하였습니다. 이번 단계에서는 동일한 입력 데이터를 사용하되, 기존의 순차 모델과는 다른 구조와 원리를 가진 **Transformer 기반 감성 분류 모델**을 구성하고자 합니다.

Transformer는 문장 전체를 한 번에 바라보며 단어 간의 상호작용을 계산하는 Self-Attention 메커니즘을 사용합니다. 이러한 특성 덕분에 Transformer는 문맥적 의존성이 강한 자연어 처리 작업에서 매우 뛰어난 성능을 발휘합니다. 아래에서는 Transformer 모델이 어떻게 구성되는지 단계별로 설명드리겠습니다.

### 4주차의 학습 목표

- RNN & LSTM과 달리 Transformer가 **순차적 처리 없이도 문맥(Context)을 파악하는 방식**을 학습합니다.

### Mission1

```python
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout, LayerNormalization, Input,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MultiHeadAttention
```

Transformer 모델은 단순한 신경망 구조와 달리 다수의 구성 요소가 함께 동작해야 합니다. 이를 위해 먼저 Keras의 Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention 레이어를 import합니다.

- [ ] Transformer가 RNN, LSTM과 다른점을 정리해주세요.

```python
class TokenAndPositionalEmbedding(tf.keras.layers.Layer):
	# Token Embedding
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(maxlen, embed_dim)
	# Positional Embedding
    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        return self.token_emb(x) + positions
```

Transformer의 중요한 특징 중 하나는 RNN처럼 단어를 순서대로 처리하지 않는다는 점입니다.

Self-Attention은 문장 내 모든 단어를 동시에 바라보는 구조이기 때문에, **단어의 순서 정보**를 모델이 스스로 알 수 없습니다. 따라서 Transformer에서는 단어의 의미뿐 아니라 해당 단어가 문장에서 **몇 번째 위치에 존재하는지**를 모델이 학습할 수 있도록 **Positional Embedding**을 추가해야 합니다.

위 코드를 보면, `TokenAndPositionalEmbedding` 레이어는 두 가지 임베딩을 결합합니다.

1. **Token Embedding** → 이 단어가 무엇인가?
2. **Positional Embedding** → 이 단어는 문장에서 어디에 있는가?

이 두 임베딩을 더함으로써 Transformer는 단어의 **의미 + 위치** 정보를 동시에 학습할 수 있게 됩니다.

- [ ] **Positional Embedding**에 대해서 정리해주세요.

```python
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # MultiHeadAttention
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        # Feed Forward Network (FFN)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
    # Residual Connection + LayerNormalization
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)
```

Transformer의 핵심 구조는 **Self-Attention**과 Feed Forward Network(FFN)으로 구성된 TransformerBlock입니다.

**(1) Multi-Head Self-Attention**

Self-Attention은 문장의 모든 단어들이 서로를 얼마나 참조하는지 계산하는 메커니즘입니다.

예를 들어,

> “배우 연기는 좋았지만 스토리가 아쉬웠습니다.”

이라는 문장에서 “좋았지만”은 앞의 “연기”와 뒤의 “스토리” 모두와 강한 문맥적 연결을 가집니다.

Self-Attention은 이러한 관계를 점수로 계산하고, 각 단어가 문장에서 어떤 역할을 하는지 재정의합니다.

Multi-Head 구조는 이러한 Attention 연산을 여러 개의 관점에서 동시에 수행하여, 다양한 문맥적 패턴을 학습할 수 있도록 돕습니다.

**(2) Feed Forward Network (FFN)**

Self-Attention으로 얻은 정보를 다시 한 번 비선형 변환을 통해 확장하기 위해 FFN을 사용합니다.

FFN은 각 단어 위치에 독립적으로 적용되며, 단어 표현의 복잡성을 높여줍니다.

**(3) Residual Connection + Layer Normalization**

Transformer 모델이 깊어질수록 정보 손실이나 gradient 문제를 방지하기 위해 Residual Connection(잔차 연결)과 Layer Normalization을 사용합니다.

- [ ] **Multi-Head Self-Attention, Feed Forward Network (FFN), Residual Connection, Layer Normalization** 에 대해서 정리해주세요. (참고: https://sonstory.tistory.com/89)

```python
inputs = Input(shape=(MAX_LEN,))
x = TokenAndPositionalEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)

for _ in range(NUM_BLOCKS):
    x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)(x)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation='sigmoid')(x)

transformer_model = Model(inputs, outputs)
```

이 코드는 앞에서 만든 **Token+Positional Embedding**과 **TransformerBlock**을 이용해 감성 분류용 Transformer 모델을 완성하는 단계입니다.

```python
transformer_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = transformer_model.fit(
    train_X, train_Y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)
```

훈련 데이터 중 20%를 검증용으로 분리하여 모델의 과적합 여부를 함께 모니터링합니다.

```python
test_loss, test_acc = transformer_model.evaluate(test_X, test_Y, verbose=0)
print(test_loss, test_acc)
```

학습에 전혀 사용하지 않은 test 데이터를 사용하여 모델의 최종 성능을 평가합니다.

```python
example_sentences = [
    "이 영화는 정말 재미있고, 스토리도 흥미진진하며 배우들의 연기까지 완벽했다.",
    "전체적으로 지루하고 재미가 없었다.",
]

example_seq = vectorize_layer(example_sentences)
pred = transformer_model.predict(example_seq)
```

마지막으로 직접 입력한 문장에 대해 Transformer 모델이 감성을 어떻게 분류하는지 확인합니다.

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
│   │   │   ├── week3_mission.ipynb
│   │   │   └── week3.md
│   │   ├── week4/
│   │   │   ├── week4_mission.ipynb       # Week4 실습 코드 파일
│   │   │   └── week4.md
│   │
│   ├── week1/
│   │   └── WIL.md
│   ├── week2/
│   │   └── WIL.md
│   ├── week3/
│   │   └── WIL.md
│   ├── week4/
│   │   └── WIL.md                       # Week4 WIL 믈로그 링크
│
└── README.md

```

### 끝으로…

이번 주는 시험 기간과 겹쳐 많이 바빴을 텐데도, 끝까지 Week4 미션을 정리하고 Transformer 모델까지 공부하시느라 정말 수고 많으셨어요.

시험기간이라 모두 바쁘실 것 같아 Week4는 의도적으로 너무 무겁지 않게 구성했습니다. 평소보다 미션이 조금 가벼운 편이라 부담 없이 진행하셨길 바라요!

기말 시험도 잘 준비하시고, 좋은 결과 있길 진심으로 응원하겠습니다! 🍀

저희는 시험이 끝난 뒤, **12/15 기획코스**에서 다시 뵐게요!
