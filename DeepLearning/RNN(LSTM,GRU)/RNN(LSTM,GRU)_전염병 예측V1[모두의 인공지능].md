[TOC]

# 시계열 데이터(Time Sereies Data)다루기

## II. 전염병 예측_V1 [ 간단한 DataSet ]

- 문제 정의
  - 3일 동안의 확진자 수 추이를 보고 다음날의 확진자 수를 예측


```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

<details>
  <summary>Library</summary>

- from keras.models import Sequential
  - 케라스의 모델 도구(models)중 시퀀셜 모델을 불러오는 명령어
- from keras.layers import SimpleRNN, Dense, GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D
  - 레이어 도구(layers)중 SimpleRNN과 Dense도구등을 불러오는 명령어
- from sklearn.prerprocessing import MinMaxSacler
  - 데이터를 정규화하기 위한 MinMaaxScaler 함수를 불러오는 명령어
- from sklearn.metrics import mean_squared_error
  - 결과의 정확도를 계산하기 위한 함수인 mean_squared_error를 불러오는 명령어
- from sklearn.model_selection import train_test_split
  - 데이터를 훈련데이터와 검증 데이터로 나누는 명령어
- import math
  - 수학 계산을 도와주는 math 라이브러리
- import numpy as np
  - 수학 계산 라이브러리 numpy를 불러오고 np로 줄여서 사용
- import matplotlib.pyplot as plt
  - 그래프 라이브러리중 pyplot 라이브러리 사용
  - plt로 줄여서 사용

</details>

### 1. 데이터 불러오기

```python
# github 저장소에서 데이터 불러오기
# !git clone https://github.com/yhlee1627/deeplearning.git
# git이 안될경우 주소에서 다운로드

df = pd.read_csv(r'./deeplearning/corona_daily.csv', usecols = [3], engine = 'python', skipfooter = 3)
print(df)
dataset = df.values
dataset = df.astype('float32')
```

```
     Confirmed
0           24
1           24
2           27
3           27
4           28
..         ...
107      11190
108      11206
109      11225
110      11265
111      11344

[112 rows x 1 columns]
```

### 2. 데이터 전처리

#### 2-1. 정규화

```python
# 데이터를 정규화 하는 범위를 0 ~ 1
scaler = MinMaxScaler(feature_range = (0,1))
# 데이터 정규화 적용
Dataset = scaler.fit_transform(dataset)
```

#### 2-2. 데이터 분할

```python
# 전체 데이터를 Training Data(80%), Test Data(20%) 분할
train_data, test_data = train_test_split(Dataset, test_size = 0.2, shuffle = False)
print(f"Train Data 개수 : {len(train_data)}, Test Data 개수 : {len(test_data)}")
```

```
Train Data 개수 : 89, Test Data 개수 : 23
```



- 데이터 형태 변경

  - 3일치의 데이터를 사용하여 4번째 날짜의 값을 예측

    <img src='https://p.ipic.vip/1xbv63.png' width=50%>


```python
# dataset : 원 데이터, look_back : 연속되는 데이터 개수
def create_dataset(dataset, look_back):
    x_data = []
    y_data = []
    for i in range(len(dataset) - look_back - 1):
        data = dataset[i:(i+look_back), 0]
        x_data.append(data)
        y_data.append(dataset[i + look_back, 0])
    return np.array(x_data), np.array(y_data)
```

- 3일치의 데이터를 사용하므로 look_back = 3

```python
look_back = 3
x_train, y_train = create_dataset(train_data, look_back)
x_test, y_test = create_dataset(test_data, look_back)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```
(85, 3) (85,)
(19, 3) (19,)
```

- 2차원 배열을 3차원 배열로

  <img src='https://p.ipic.vip/1zq87h.png' width = 50%>

```python
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(X_train.shape)
print(X_test.shape)
```

```
(85, 3, 1)
(19, 3, 1)
```

### 3. 모델 정의

#### 3-1. 순환 신경망(RNN; Recurrent Neural Network)

- 순차적인 데이터(Sequence Data)를 처리하기 위한 인공 신경망
  - 어떤 특정 부분이 반복되는 구조에서 순서를 학습하기에 효과적
  - 시계열 데이터, 자연어 등에 적용 가능
  - 기존 Neural Network와 달리 '기억(Hidden State)'을 가지고 있음
  - 은닉 계층 안에 하나 이상의 순환 계층을 갖는 신경망 구조
    - 이전 단계의 출력 값이 현재 단계의 입력 값으로 다시 들어가는 반복 구조
    - 가중치가 모든 타임 스텝에서 공유됨

<details>
<summary>RNN Process</summary>


<img src='https://p.ipic.vip/f08ogi.png'>

</details>


```python
model = Sequential()
model.add(SimpleRNN(3, input_shape = (look_back, 1)))
model.add(Dense(1, activation = 'linear'))       # 최종 예측 값은 연속된 데이터 이후의 값, 즉 확진자의 수
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 3)                 15        
                                                                 
 dense (Dense)               (None, 1)                 4         
                                                                 
=================================================================
Total params: 19
Trainable params: 19
Non-trainable params: 0
_________________________________________________________________
```

#### 3-2. LSTM ; Long Short Term Memory

- RNN 의 **장기의존성 문제**와 **기울기 소실 문제**를 해결한 알고리즘

  - 가중치 행렬 $W$의 행렬 곱 연산이 그레이디언트 경로에 나타나지 않도록 구조 변경

- 기존 RNN에 장기 기억 셀(Cell State)을 추가함

  - $c_t$를 연결하는 경로에는 가중치 행렬 $W$의 행렬 곱 연산이 없음

    <img src='https://p.ipic.vip/p1b6vv.png' width=80%>

- 장기 기억 셀 연산에 사용되는 게이트 추가

  - Forget Gate($f_t$) : 과거의 정보를 얼마나 유지할 것인가?
  - Input Gate($i_t$) : 새로 입력된 정보를 얼만큼 활용할 것인가?
  - Output Gate($o_t$) : Cell State 나온 정보를 얼마나 출력할 것인가?

<details>
<summary>LSTM Process</summary>
<img src='https://p.ipic.vip/elqu30.png'>
</details>

```python
model = Sequential()
model.add(LSTM(3, input_shape = (look_back, 1)))
model.add(Dense(1, activation = 'linear'))       # 최종 예측 값은 연속된 데이터 이후의 값, 즉 확진자의 수
model.summary()
```

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 3)                 60        
                                                                 
 dense_1 (Dense)             (None, 1)                 4         
                                                                 
=================================================================
Total params: 64
Trainable params: 64
Non-trainable params: 0
_________________________________________________________________
```

#### 3-3. GRU ; Gated Recurrent Unit

- LSTM의 장점을 유지하면서 게이트 구조를 단순하게 만든 순환 신경망
  - 업데이트 게이트(Update Gate) = Forget Gate + Input Gate
    - 과거의 기억중 사용할 정보의 양과 현 시점의 입력 정보 중 사용할 정보 수집
  - 리셋 게이트(Reset Gate)
    - 현 시점의 입력 정보 중 새로운 정보를 추가할 때, 과거의 기억 중 필요한 정보의 양 계산
  - 장기 기억 셀(Cell State)을 삭제
    - 은닉 상태($h_{t-1}$)가 장기 기억과 단기 기억 모두를 기억하도록 함
  - 출력 게이트가 존재하지 않음
    - 전체 상태 벡터가 매 타임 스텝마다 출력

<details>
<summary>GRU Process</summary>
<img src='https://p.ipic.vip/6obmtd.png'>
</details>


```python
model = Sequential()
model.add(GRU(3, input_shape = (look_back, 1)))
model.add(Dense(1, activation = 'linear'))       # 최종 예측 값은 연속된 데이터 이후의 값, 즉 확진자의 수
model.summary()
```

```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 3)                 54        
                                                                 
 dense_2 (Dense)             (None, 1)                 4         
                                                                 
=================================================================
Total params: 58
Trainable params: 58
Non-trainable params: 0
_________________________________________________________________
```

#### 3-4. 양방향(Bidirectional) LSTM/GRU

- 양방향 순환 층(Bidirectional Recurrent Layer)
  - 순환 네트워크에 같은 정보를 다른 방향으로 주입하여 정확도를 높이고 기억을 좀 더 오래 유지
  - 이전의 층이 전체 출력 시퀀스를 반환해야 함 [ return_sequence = True ]

<details>
<summary>양방향 LSTM/GRU</summary>


<img src='https://p.ipic.vip/2hj9xp.png'>

</details>

```python
model = Sequential()
model.add(Bidirectional(LSTM(3), input_shape = (look_back, 1))) # SimpleRNN, LSTM, GRU
model.add(Dense(1, activation = 'linear'))       # 최종 예측 값은 연속된 데이터 이후의 값, 즉 확진자의 수
model.summary()
```

```
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 bidirectional (Bidirectiona  (None, 6)                120       
 l)                                                              
                                                                 
 dense_3 (Dense)             (None, 1)                 7         
                                                                 
=================================================================
Total params: 127
Trainable params: 127
Non-trainable params: 0
_________________________________________________________________
```

#### 3-5. 1D CNN + LSTM/GRU

<details>
<summary>1D CNN + LSTM/GRU</summary>
<img src='https://p.ipic.vip/bzx124.png'>
</details>
</details>

```python
model = Sequential()
model.add(Conv1D(filters=1,
                 kernel_size=1,
                 strides=1,
                 padding='valid',
                 input_shape=(look_back, 1), 
                 use_bias=False, name='c1d'))
model.add(MaxPooling1D(pool_size = 3))
model.add(LSTM(32))
model.add(Dense(1, activation = 'linear'))
model.summary()
```

```
Model: "sequential_19"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 c1d (Conv1D)                (None, 3, 1)              1         
                                                                 
 max_pooling1d_15 (MaxPoolin  (None, 1, 1)             0         
 g1D)                                                            
                                                                 
 lstm_12 (LSTM)              (None, 32)                4352      
                                                                 
 dense_14 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 4,386
Trainable params: 4,386
Non-trainable params: 0
_________________________________________________________________
```

### 4. 모델 학습

```python
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, epochs = 100, batch_size = 1, verbose = 1)
```

<details>
  <summary>Result</summary>

```
Epoch 1/100
85/85 [==============================] - 1s 2ms/step - loss: 0.2615
Epoch 2/100
85/85 [==============================] - 0s 2ms/step - loss: 0.0322
Epoch 3/100
85/85 [==============================] - 0s 3ms/step - loss: 0.0122
Epoch 4/100
85/85 [==============================] - 0s 2ms/step - loss: 0.0084
Epoch 5/100
85/85 [==============================] - 0s 2ms/step - loss: 0.0052
Epoch 6/100
85/85 [==============================] - 0s 2ms/step - loss: 0.0031
Epoch 7/100
85/85 [==============================] - 0s 2ms/step - loss: 0.0018
Epoch 8/100
85/85 [==============================] - 0s 2ms/step - loss: 0.0011
Epoch 9/100
85/85 [==============================] - 0s 2ms/step - loss: 7.9880e-04
Epoch 10/100
85/85 [==============================] - 0s 2ms/step - loss: 6.4356e-04
Epoch 11/100
85/85 [==============================] - 0s 2ms/step - loss: 5.8702e-04
Epoch 12/100
85/85 [==============================] - 0s 2ms/step - loss: 5.5295e-04
Epoch 13/100
85/85 [==============================] - 0s 2ms/step - loss: 5.5005e-04
Epoch 14/100
85/85 [==============================] - 0s 2ms/step - loss: 5.2920e-04
Epoch 15/100
85/85 [==============================] - 0s 2ms/step - loss: 5.1512e-04
Epoch 16/100
85/85 [==============================] - 0s 2ms/step - loss: 5.0810e-04
Epoch 17/100
85/85 [==============================] - 0s 2ms/step - loss: 5.0475e-04
Epoch 18/100
85/85 [==============================] - 0s 2ms/step - loss: 4.7168e-04
Epoch 19/100
85/85 [==============================] - 0s 2ms/step - loss: 4.6102e-04
Epoch 20/100
85/85 [==============================] - 0s 2ms/step - loss: 4.3395e-04
Epoch 21/100
85/85 [==============================] - 0s 2ms/step - loss: 4.1664e-04
Epoch 22/100
85/85 [==============================] - 0s 2ms/step - loss: 4.0405e-04
Epoch 23/100
85/85 [==============================] - 0s 2ms/step - loss: 3.7727e-04
Epoch 24/100
85/85 [==============================] - 0s 2ms/step - loss: 3.8226e-04
Epoch 25/100
85/85 [==============================] - 0s 2ms/step - loss: 3.4674e-04
Epoch 26/100
85/85 [==============================] - 0s 2ms/step - loss: 3.3123e-04
Epoch 27/100
85/85 [==============================] - 0s 2ms/step - loss: 3.2109e-04
Epoch 28/100
85/85 [==============================] - 0s 2ms/step - loss: 3.0810e-04
Epoch 29/100
85/85 [==============================] - 0s 2ms/step - loss: 2.9197e-04
Epoch 30/100
85/85 [==============================] - 0s 2ms/step - loss: 2.6786e-04
Epoch 31/100
85/85 [==============================] - 0s 2ms/step - loss: 2.6057e-04
Epoch 32/100
85/85 [==============================] - 0s 2ms/step - loss: 2.5553e-04
Epoch 33/100
85/85 [==============================] - 0s 2ms/step - loss: 2.4045e-04
Epoch 34/100
85/85 [==============================] - 0s 2ms/step - loss: 2.4307e-04
Epoch 35/100
85/85 [==============================] - 0s 2ms/step - loss: 2.0048e-04
Epoch 36/100
85/85 [==============================] - 0s 2ms/step - loss: 2.1084e-04
Epoch 37/100
85/85 [==============================] - 0s 2ms/step - loss: 2.0098e-04
Epoch 38/100
85/85 [==============================] - 0s 2ms/step - loss: 1.9071e-04
Epoch 39/100
85/85 [==============================] - 0s 2ms/step - loss: 1.8199e-04
Epoch 40/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7287e-04
Epoch 41/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6764e-04
Epoch 42/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6028e-04
Epoch 43/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6704e-04
Epoch 44/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6465e-04
Epoch 45/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5714e-04
Epoch 46/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4610e-04
Epoch 47/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6597e-04
Epoch 48/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4806e-04
Epoch 49/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4749e-04
Epoch 50/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5087e-04
Epoch 51/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4405e-04
Epoch 52/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5076e-04
Epoch 53/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5360e-04
Epoch 54/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4832e-04
Epoch 55/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5261e-04
Epoch 56/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4746e-04
Epoch 57/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6867e-04
Epoch 58/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5876e-04
Epoch 59/100
85/85 [==============================] - 0s 2ms/step - loss: 1.8017e-04
Epoch 60/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6197e-04
Epoch 61/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5545e-04
Epoch 62/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4853e-04
Epoch 63/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6503e-04
Epoch 64/100
85/85 [==============================] - 0s 2ms/step - loss: 1.9090e-04
Epoch 65/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7306e-04
Epoch 66/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4990e-04
Epoch 67/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6291e-04
Epoch 68/100
85/85 [==============================] - 0s 2ms/step - loss: 2.0467e-04
Epoch 69/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6585e-04
Epoch 70/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7717e-04
Epoch 71/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6153e-04
Epoch 72/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6774e-04
Epoch 73/100
85/85 [==============================] - 0s 2ms/step - loss: 2.1411e-04
Epoch 74/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6509e-04
Epoch 75/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5477e-04
Epoch 76/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5216e-04
Epoch 77/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7200e-04
Epoch 78/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7924e-04
Epoch 79/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6083e-04
Epoch 80/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5922e-04
Epoch 81/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5302e-04
Epoch 82/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7367e-04
Epoch 83/100
85/85 [==============================] - 0s 2ms/step - loss: 1.3810e-04
Epoch 84/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7868e-04
Epoch 85/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5942e-04
Epoch 86/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7223e-04
Epoch 87/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7669e-04
Epoch 88/100
85/85 [==============================] - 0s 2ms/step - loss: 2.2740e-04
Epoch 89/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7142e-04
Epoch 90/100
85/85 [==============================] - 0s 2ms/step - loss: 1.8502e-04
Epoch 91/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4546e-04
Epoch 92/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5790e-04
Epoch 93/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4681e-04
Epoch 94/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4743e-04
Epoch 95/100
85/85 [==============================] - 0s 2ms/step - loss: 1.9101e-04
Epoch 96/100
85/85 [==============================] - 0s 2ms/step - loss: 1.6783e-04
Epoch 97/100
85/85 [==============================] - 0s 2ms/step - loss: 1.5435e-04
Epoch 98/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7348e-04
Epoch 99/100
85/85 [==============================] - 0s 2ms/step - loss: 1.4981e-04
Epoch 100/100
85/85 [==============================] - 0s 2ms/step - loss: 1.7339e-04
<keras.callbacks.History at 0x7f8ab8d69e40>
```

</details>

### 5. 데이터 예측


```python
# 모델을 적용한 출력값 (0~1)
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# scaler를 통하여 실제 값으로 변환
TrainPredict = scaler.inverse_transform(trainPredict)  # 훈련 데이터의 예측값
Y_train = scaler.inverse_transform([y_train])          # 훈련 데이터의 타깃값
TestPredict = scaler.inverse_transform(testPredict)    # 테스트 데이터의 예측값
Y_test = scaler.inverse_transform([y_test])            # 테스트 데이터의 타깃값
```

```
3/3 [==============================] - 0s 1ms/step
1/1 [==============================] - 0s 14ms/step
```

### 6. 모델의 정확도

```python
trainScore = math.sqrt(mean_squared_error(Y_train[0], TrainPredict[:,0]))
print(f'Train Score: {trainScore:.2f} RMSE')

testScore = math.sqrt(mean_squared_error(Y_test[0], TestPredict[:,0]))
print(f'Test Score: {testScore:.2f} RMSE')
```

```
Train Score: 257.38 RMSE
Test Score: 87.52 RMSE
```

### 7. 시각화(실제 값과 예측값 그래프)

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(TrainPredict)+look_back, :] = TrainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(TrainPredict)+(look_back+1)*2:len(dataset), :] = TestPredict

plt.plot(dataset)               # 파란색 : 실제 값
plt.plot(trainPredictPlot)      # 주황색 : Training Data 예측값
plt.plot(testPredictPlot)       # 초록색 : Test Data 예측값
plt.show()
```

<img src='https://p.ipic.vip/r5s53u.png'>

