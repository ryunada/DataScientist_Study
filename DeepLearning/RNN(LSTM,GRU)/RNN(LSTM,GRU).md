[TOC]

# 시계열 데이터(Time Sereies Data)다루기

## I. 온도 예측 문제

- 문제 정의
  - 한 시간에 한 번씩 샘플링된 5일간의 데이터가 주어졌을 때 24시간 뒤의 온도를 예측
  - 데이터를 신경망에 주입할 수 있는 형태로 전처리해봄
  - 데이터 : 수치형이 되어 있어야 함
  - 각 시계열을 독립적으로 정규화하여 비슷한 범위를 가진 작은 값으로 바꿈

- Dataset

  - 독일 예나(Jena)시에 있는 막스 플랑크 생물지구화학연구소(Max Plank Institute for Biogeochemisty)의 기상 관측소에서 수집  

  - 수년간에 걸쳐(온도, 기압, 습도, 풍향 등) 14개의 관측치가 10분마다 기록되어 있음

  - 원본은 2003년부터 기록되어 있지만 2009~2016년 사이의 데이터 사용


|  Feature Name   | 설명           |
| :-------------: | :------------- |
|    Date Time    | 날짜-시간 참조 |
|    p (mbar)     | 압력 단위      |
|    T (degC)     | 섭씨 온도      |
|    Tpot (K)     | 켈빈 온도      |
|   Tdew (degC)   | 습도           |
|     rh (%)      | 상대 습도      |
|  VPmax (mbar)   | 포화 증기압    |
|  VPact (mbar)   | 증기압         |
|  VPdef (mbar)   | 증기압 부종    |
|    sh (g/kg)    | 비습도         |
| H2OC (mmol/mol) | 수증기 농도    |
|  rho (g/m**3)   | Airtight       |
|    wv (m/s)     | 풍속           |
|  max. wv (m/s)  | 최대 풍속      |
|    wd (deg)     | 바람 방향      |

### 1. 데이터 다운로드 및 확인

#### 1. 데이터 다운로드


```python
## 작업 위치 설정
import os

print(os.getcwd())
os.chdir(r"C:\Users\Desktop")
print(os.getcwd())

# 압축된 데이터 내려받기
!wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
# 압축 풀기
!unzip jena_climate_2009_2016.csv.zip
```

```
C:\Users\Desktop
C:\Users\Desktop
```

#### 2. 데이터 확인

```python
## 데이터 불러오기
import pandas as pd
df = pd.read_csv('./jena_climate_2009_2016.csv')
```

```python
df
```

|        |           Date Time | p (mbar) | T (degC) | Tpot (K) | Tdew (degC) | rh (%) | VPmax (mbar) | VPact (mbar) | VPdef (mbar) | sh (g/kg) | H2OC (mmol/mol) | rho (g/m**3) | wv (m/s) | max. wv (m/s) | wd (deg) |
| -----: | ------------------: | -------: | -------: | -------: | ----------: | -----: | -----------: | -----------: | -----------: | --------: | --------------: | -----------: | -------: | ------------: | -------- |
|      0 | 01.01.2009 00:10:00 |   996.52 |    -8.02 |   265.40 |       -8.90 |  93.30 |         3.33 |         3.11 |         0.22 |      1.94 |            3.12 |      1307.75 |     1.03 |          1.75 | 152.3    |
|      1 | 01.01.2009 00:20:00 |   996.57 |    -8.41 |   265.01 |       -9.28 |  93.40 |         3.23 |         3.02 |         0.21 |      1.89 |            3.03 |      1309.80 |     0.72 |          1.50 | 136.1    |
|      2 | 01.01.2009 00:30:00 |   996.53 |    -8.51 |   264.91 |       -9.31 |  93.90 |         3.21 |         3.01 |         0.20 |      1.88 |            3.02 |      1310.24 |     0.19 |          0.63 | 171.6    |
|      3 | 01.01.2009 00:40:00 |   996.51 |    -8.31 |   265.12 |       -9.07 |  94.20 |         3.26 |         3.07 |         0.19 |      1.92 |            3.08 |      1309.19 |     0.34 |          0.50 | 198.0    |
|      4 | 01.01.2009 00:50:00 |   996.51 |    -8.27 |   265.15 |       -9.04 |  94.10 |         3.27 |         3.08 |         0.19 |      1.92 |            3.09 |      1309.00 |     0.32 |          0.63 | 214.3    |
|    ... |                 ... |      ... |      ... |      ... |         ... |    ... |          ... |          ... |          ... |       ... |             ... |          ... |      ... |           ... | ...      |
| 420446 | 31.12.2016 23:20:00 |  1000.07 |    -4.05 |   269.10 |       -8.13 |  73.10 |         4.52 |         3.30 |         1.22 |      2.06 |            3.30 |      1292.98 |     0.67 |          1.52 | 240.0    |
| 420447 | 31.12.2016 23:30:00 |   999.93 |    -3.35 |   269.81 |       -8.06 |  69.71 |         4.77 |         3.32 |         1.44 |      2.07 |            3.32 |      1289.44 |     1.14 |          1.92 | 234.3    |
| 420448 | 31.12.2016 23:40:00 |   999.82 |    -3.16 |   270.01 |       -8.21 |  67.91 |         4.84 |         3.28 |         1.55 |      2.05 |            3.28 |      1288.39 |     1.08 |          2.00 | 215.2    |
| 420449 | 31.12.2016 23:50:00 |   999.81 |    -4.23 |   268.94 |       -8.53 |  71.80 |         4.46 |         3.20 |         1.26 |      1.99 |            3.20 |      1293.56 |     1.49 |          2.16 | 225.8    |
| 420450 | 01.01.2017 00:00:00 |   999.82 |    -4.82 |   268.36 |       -8.42 |  75.70 |         4.27 |         3.23 |         1.04 |      2.01 |            3.23 |      1296.38 |     1.23 |          1.96 | 184.9    |


```python
# 데이터 형태 확인 및 null값 확인
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 420451 entries, 0 to 420450
    Data columns (total 15 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   Date Time        420451 non-null  object 
     1   p (mbar)         420451 non-null  float64
     2   T (degC)         420451 non-null  float64
     3   Tpot (K)         420451 non-null  float64
     4   Tdew (degC)      420451 non-null  float64
     5   rh (%)           420451 non-null  float64
     6   VPmax (mbar)     420451 non-null  float64
     7   VPact (mbar)     420451 non-null  float64
     8   VPdef (mbar)     420451 non-null  float64
     9   sh (g/kg)        420451 non-null  float64
     10  H2OC (mmol/mol)  420451 non-null  float64
     11  rho (g/m**3)     420451 non-null  float64
     12  wv (m/s)         420451 non-null  float64
     13  max. wv (m/s)    420451 non-null  float64
     14  wd (deg)         420451 non-null  float64
    dtypes: float64(14), object(1)
    memory usage: 48.1+ MB

### 2. 데이터 전처리

#### 2-1. 데이터 파싱

- Target 데이터 분리
  - Target : 온도(temperature)[ T (degC) ]

```python
# 날짜 데이터(Date Time)
DateTime = df.iloc[:,0]

# 날짜 데이터를 제외한 나머지 데이터
raw_data = df.iloc[:,1:]

# 기온 데이터
temperature = raw_data['T (degC)']
```

#### 2-2. 데이터 시각화

```python
import matplotlib.pyplot as plt
# 한글 깨짐 해결
plt.rcParams['font.family'] = 'Malgun Gothic' # mac => AppleGhothic
# 마이너스기호(-)깨짐 해결
plt.rcParams['axes.unicode_minus'] = False
```


```python
## 데이터셋의 전체 기간의 온도 그래프(범위 8년)

plt.title('데이터셋의 전체 기간의 온도 그래프(범위 8년)')
plt.plot(range(len(temperature)), temperature)
plt.show()
```

<img src='https://p.ipic.vip/trhfaj.png' width = 80%>


- 매년 온도에 **주기성**이 있다는 것을 볼 수 있음

#### 2-3. 데이터 분할

- 각 분할에 사용할 샘플 개수 계산
  - Training Data : 50%
  - Validation Data : 25%
  - Test Data : 25%


```python
train_samples_n = int(0.5 * len(raw_data))
val_samples_n = int(0.25 * len(raw_data))
test_samples_n = len(raw_data) - train_samples_n - val_samples_n
print(f"Train Data 개수 : {train_samples_n}")
print(f"Val Data 개수 : {val_samples_n}")
print(f"Test Data 개수 : {test_samples_n}")
```

    Train Data 개수 : 210225
    Val Data 개수 : 105112
    Test Data 개수 : 105114

#### 2-4. 데이터 정규화

- 이미 수치형이기 때문에 벡터화가 필요 X

```python
# 시계열은 스케일이 각각 다르므로 독립적으로 정규화(평균과 표준편차 사용)
mean = raw_data[:train_samples_n].mean(axis = 0)
raw_data -= mean
std = raw_data[:train_samples_n].std(axis = 0)
raw_data /= std
```

#### 2-5. 시퀀스 데이터 준비 및 분할

- 과거 5일치 데이터와 24시간 뒤 타깃 온도의 배치를 반환하는 Dataset 객체

- 과정

  I. 현재 데이터의 시간 단위(10분)와 예측하려는 시간 단위(1시간)을 통일

  - 10분 * 6 = 60분(1시간)

  - $sampling\_rate = 6$

  II. 사용할 데이터의 기간 정의 (5일 = 120시간)

  - $sequence\_length = 120$
  - $sequence\_stride = 1$

  III. 예측 시점(24시간 뒤)

  - $delay = sampling\_rate * (sequence\_length + 24 - 1)$
  - ***이틀 뒤를 예측***
    - $delay = sampling\_rate * (sequence\_length + 48 -1)$

<img src='https://p.ipic.vip/k0njck.png'>

- timeseries_dataset_from_array() : 중복된 데이터 때문에 생기는 메모리 낭비를 줄여줌

<details>
<summary>Option</summary>
```python
tf.keras.utils.timeseries_dataset_from_array(
        data,
        targets,
        sampling_rate = 1,
			  sequence_length,
        delay = sampling_rate * (sequence_length + 24 - 1)      
        sequence_stride = 1,
        batch_size = 128,
        shuffle = Fasle,
        seed None,
        start_index = None,
        end_index = None
    )
```



| Option                       |              Explanation              |
| :--------------------------- | :-----------------------------------: |
| data                         |      타깃 데이터를 제외한 데이터      |
| targets                      |              타깃 데이터              |
| sampling_rate                |         시퀀스 데이터의 단위          |
| sequence_length              |  훈련에 사용할 시퀀스 데이터의 길이   |
| delay                        |         예측하고자 하는 시점          |
| sequence_stride(default = 1) |         연속 시계열 간의 거리         |
| batch_size(default = 128)    | 각 배치의 시계열 샘플 수(마지막 제외) |
| shuffle                      |        출력 샘플을 섞을지 말지        |
| seed                         |                고정값                 |
| start_index                  |   사용할 데이터의 시작 인덱스 위치    |
| end_index                    |    사용할 데이터의 끝 인덱스 위치     |

</details>


```python
from tensorflow import keras

sampling_rate = 6  # 시간당 하나의 데이터 포인트가 샘플링됨
sequence_length = 120 # 이전 5일(120시간)의 데이터 사용
delay = sampling_rate * (sequence_length + 24 - 1)    # 시퀀스 끝에서 24시간 후의 온도
batch_size = 256

# Training Data : 0 ~ train_samples_n
train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets = temperature[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = True,
    batch_size = batch_size,
    start_index = 0,
    end_index = train_samples_n
)

# Validation Data : train_samples_n ~ train_samples_n + val_samples_n
val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets = temperature[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = True,
    batch_size = batch_size,
    start_index = train_samples_n,
    end_index = train_samples_n + val_samples_n
)

# Test Data : train_samples_n + val_sampes_n + End
test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets = temperature[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    shuffle = True,
    batch_size = batch_size,
    start_index = train_samples_n + val_samples_n
)
```


```python
for samples, targets in train_dataset:
    print(f"샘플 크기 : {samples.shape}")
    print(f"타깃 크기 : {targets.shape}")
    break
```

    샘플 크기 : (256, 120, 14)
    타깃 크기 : (256,)

→ (batch_size, sequence_length, column 개수)

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



<img src='https://p.ipic.vip/lynhpo.png'>

</details>


```python
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.SimpleRNN(16)(inputs) 
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_20"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_21 (InputLayer)        [(None, 120, 14)]         0         
_________________________________________________________________
simple_rnn_11 (SimpleRNN)    (None, 16)                496       
_________________________________________________________________
dense_68 (Dense)             (None, 1)                 17        
=================================================================
Total params: 513
Trainable params: 513
Non-trainable params: 0
_________________________________________________________________
```

#### 3-2. LSTM ; Long Short Term Memory

- RNN 의 **장기의존성 문제**와 **기울기 소실 문제**를 해결한 알고리즘

  - 가중치 행렬 $W$의 행렬 곱 연산이 그레이디언트 경로에 나타나지 않도록 구조 변경

- 기존 RNN에 장기 기억 셀(Cell State)을 추가함

  - $c_t$를 연결하는 경로에는 가중치 행렬 $W$의 행렬 곱 연산이 없음

    <img src='https://p.ipic.vip/q7fi9u.png' width=80%>

- 장기 기억 셀 연산에 사용되는 게이트 추가

  - Forget Gate($f_t$) : 과거의 정보를 얼마나 유지할 것인가?
  - Input Gate($i_t$) : 새로 입력된 정보를 얼만큼 활용할 것인가?
  - Output Gate($o_t$) : Cell State 나온 정보를 얼마나 출력할 것인가?

<details>
<summary>LSTM Process</summary>
<img src='https://p.ipic.vip/elqu30.png'>
</details>


```python
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
WARNING:tensorflow:Layer lstm_42 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
Model: "model_21"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_22 (InputLayer)        [(None, 120, 14)]         0         
_________________________________________________________________
lstm_42 (LSTM)               (None, 32)                6016      
_________________________________________________________________
dense_69 (Dense)             (None, 1)                 33        
=================================================================
Total params: 6,049
Trainable params: 6,049
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
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.GRU(32)(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

```
WARNING:tensorflow:Layer gru_21 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
Model: "model_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_27 (InputLayer)        [(None, 120, 14)]         0         
_________________________________________________________________
gru_21 (GRU)                 (None, 32)                4608      
_________________________________________________________________
dense_74 (Dense)             (None, 1)                 33        
=================================================================
Total params: 4,641
Trainable params: 4,641
Non-trainable params: 0
_________________________________________________________________
```

#### 3-4. 양방향(Bidirectional) LSTM/GRU

- 양방향 순환 층(Bidirectional Recurrent Layer)
  - 순환 네트워크에 같은 정보를 다른 방향으로 주입하여 정확도를 높이고 기억을 좀 더 오래 유지
  - 이전의 층이 전체 출력 시퀀스를 반환해야 함 [ return_sequence = True ]

<details>
<summary>양방향 LSTM/GRU</summary>
<img src='https://p.ipic.vip/0oj5et.png'>


</details>

```python
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.Bidirectional(keras.layers.LSTM(16))(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_28"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_29 (InputLayer)        [(None, 120, 14)]         0         
_________________________________________________________________
bidirectional_11 (Bidirectio (None, 32)                3968      
_________________________________________________________________
dense_76 (Dense)             (None, 1)                 33        
=================================================================
Total params: 4,001
Trainable params: 4,001
Non-trainable params: 0
_________________________________________________________________
```

#### 3-5. 1D CNN + LSTM/GRU

<details>
<summary>1D CNN + LSTM/GRU</summary>
<img src='https://p.ipic.vip/bzx124.png'>
</details>

```python
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu')(inputs)
x = keras.layers.MaxPooling1D(pool_size = 4)(x)
x = keras.layers.LSTM(32, recurrent_dropout = 0.25)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_18"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_19 (InputLayer)        [(None, 120, 14)]         0         
_________________________________________________________________
conv1d_28 (Conv1D)           (None, 113, 32)           3616      
_________________________________________________________________
max_pooling1d_15 (MaxPooling (None, 28, 32)            0         
_________________________________________________________________
lstm_34 (LSTM)               (None, 32)                8320      
_________________________________________________________________
dense_58 (Dense)             (None, 1)                 33        
=================================================================
Total params: 11,969
Trainable params: 11,969
Non-trainable params: 0
_________________________________________________________________
```

### 4. 모델 학습

```python
model.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['mae'])
history = model.fit(train_dataset,
                   epochs = 50,
                   validation_data = val_dataset)
```

<details>
<summary>Result</summary>


```
Epoch 1/50
819/819 [==============================] - 10s 10ms/step - loss: 18.8326 - mae: 3.1639 - val_loss: 11.3457 - val_mae: 2.6436
Epoch 2/50
819/819 [==============================] - 8s 9ms/step - loss: 7.5487 - mae: 2.1354 - val_loss: 12.3939 - val_mae: 2.7926
Epoch 3/50
819/819 [==============================] - 8s 9ms/step - loss: 6.1319 - mae: 1.9127 - val_loss: 13.2097 - val_mae: 2.8667
Epoch 4/50
819/819 [==============================] - 8s 9ms/step - loss: 5.3190 - mae: 1.7756 - val_loss: 13.8082 - val_mae: 2.9343
Epoch 5/50
819/819 [==============================] - 8s 9ms/step - loss: 4.7604 - mae: 1.6784 - val_loss: 14.2529 - val_mae: 2.9634
Epoch 6/50
819/819 [==============================] - 8s 9ms/step - loss: 4.3245 - mae: 1.5984 - val_loss: 14.4667 - val_mae: 2.9821
Epoch 7/50
819/819 [==============================] - 7s 9ms/step - loss: 3.9437 - mae: 1.5281 - val_loss: 14.8976 - val_mae: 3.0351
Epoch 8/50
819/819 [==============================] - 8s 9ms/step - loss: 3.6439 - mae: 1.4680 - val_loss: 15.0247 - val_mae: 3.0486
Epoch 9/50
819/819 [==============================] - 7s 9ms/step - loss: 3.3711 - mae: 1.4119 - val_loss: 15.3134 - val_mae: 3.0891
Epoch 10/50
819/819 [==============================] - 8s 9ms/step - loss: 3.1544 - mae: 1.3656 - val_loss: 15.2457 - val_mae: 3.0834
Epoch 11/50
819/819 [==============================] - 8s 9ms/step - loss: 2.9813 - mae: 1.3269 - val_loss: 16.1958 - val_mae: 3.1793
Epoch 12/50
819/819 [==============================] - 7s 9ms/step - loss: 2.8324 - mae: 1.2934 - val_loss: 15.3415 - val_mae: 3.0982
Epoch 13/50
819/819 [==============================] - 7s 9ms/step - loss: 2.6927 - mae: 1.2617 - val_loss: 15.4643 - val_mae: 3.1161
Epoch 14/50
819/819 [==============================] - 8s 9ms/step - loss: 2.5780 - mae: 1.2337 - val_loss: 16.0362 - val_mae: 3.1598
Epoch 15/50
819/819 [==============================] - 7s 9ms/step - loss: 2.4840 - mae: 1.2111 - val_loss: 15.8293 - val_mae: 3.1265
Epoch 16/50
819/819 [==============================] - 7s 9ms/step - loss: 2.3850 - mae: 1.1863 - val_loss: 16.2681 - val_mae: 3.1831
Epoch 17/50
819/819 [==============================] - 7s 9ms/step - loss: 2.3156 - mae: 1.1687 - val_loss: 16.4217 - val_mae: 3.1789
Epoch 18/50
819/819 [==============================] - 7s 9ms/step - loss: 2.2367 - mae: 1.1484 - val_loss: 16.3539 - val_mae: 3.1843
Epoch 19/50
819/819 [==============================] - 7s 9ms/step - loss: 2.1853 - mae: 1.1359 - val_loss: 16.7141 - val_mae: 3.2121
Epoch 20/50
819/819 [==============================] - 7s 9ms/step - loss: 2.1192 - mae: 1.1186 - val_loss: 16.5897 - val_mae: 3.2126
Epoch 21/50
819/819 [==============================] - 7s 9ms/step - loss: 2.0811 - mae: 1.1075 - val_loss: 16.7286 - val_mae: 3.2304
Epoch 22/50
819/819 [==============================] - 7s 9ms/step - loss: 2.0282 - mae: 1.0945 - val_loss: 17.0738 - val_mae: 3.2693
Epoch 23/50
819/819 [==============================] - 7s 9ms/step - loss: 1.9801 - mae: 1.0808 - val_loss: 16.9164 - val_mae: 3.2285
Epoch 24/50
819/819 [==============================] - 7s 9ms/step - loss: 1.9485 - mae: 1.0726 - val_loss: 16.9744 - val_mae: 3.2474
Epoch 25/50
819/819 [==============================] - 7s 9ms/step - loss: 1.9113 - mae: 1.0622 - val_loss: 17.0581 - val_mae: 3.2425
Epoch 26/50
819/819 [==============================] - 7s 9ms/step - loss: 1.8766 - mae: 1.0525 - val_loss: 17.1100 - val_mae: 3.2527
Epoch 27/50
819/819 [==============================] - 7s 9ms/step - loss: 1.8380 - mae: 1.0418 - val_loss: 16.9964 - val_mae: 3.2490
Epoch 28/50
819/819 [==============================] - 7s 9ms/step - loss: 1.8128 - mae: 1.0343 - val_loss: 17.1390 - val_mae: 3.2641
Epoch 29/50
819/819 [==============================] - 7s 9ms/step - loss: 1.7830 - mae: 1.0248 - val_loss: 16.7854 - val_mae: 3.2279
Epoch 30/50
819/819 [==============================] - 7s 9ms/step - loss: 1.7611 - mae: 1.0184 - val_loss: 17.8748 - val_mae: 3.3447
Epoch 31/50
819/819 [==============================] - 7s 9ms/step - loss: 1.7348 - mae: 1.0105 - val_loss: 16.9679 - val_mae: 3.2536
Epoch 32/50
819/819 [==============================] - 7s 9ms/step - loss: 1.7115 - mae: 1.0043 - val_loss: 17.1717 - val_mae: 3.2788
Epoch 33/50
819/819 [==============================] - 7s 9ms/step - loss: 1.6876 - mae: 0.9983 - val_loss: 16.6467 - val_mae: 3.2324
Epoch 34/50
819/819 [==============================] - 7s 9ms/step - loss: 1.6589 - mae: 0.9901 - val_loss: 16.9705 - val_mae: 3.2549
Epoch 35/50
819/819 [==============================] - 7s 9ms/step - loss: 1.6392 - mae: 0.9836 - val_loss: 17.3274 - val_mae: 3.2906
Epoch 36/50
819/819 [==============================] - 7s 9ms/step - loss: 1.6182 - mae: 0.9774 - val_loss: 17.5664 - val_mae: 3.3065
Epoch 37/50
819/819 [==============================] - 7s 9ms/step - loss: 1.5998 - mae: 0.9712 - val_loss: 16.9406 - val_mae: 3.2536
Epoch 38/50
819/819 [==============================] - 8s 9ms/step - loss: 1.5876 - mae: 0.9674 - val_loss: 17.0346 - val_mae: 3.2664
Epoch 39/50
819/819 [==============================] - 7s 9ms/step - loss: 1.5614 - mae: 0.9602 - val_loss: 17.0298 - val_mae: 3.2787
Epoch 40/50
819/819 [==============================] - 7s 9ms/step - loss: 1.5422 - mae: 0.9545 - val_loss: 16.9384 - val_mae: 3.2534
Epoch 41/50
819/819 [==============================] - 7s 9ms/step - loss: 1.5431 - mae: 0.9540 - val_loss: 16.9735 - val_mae: 3.2668
Epoch 42/50
819/819 [==============================] - 7s 9ms/step - loss: 1.5282 - mae: 0.9491 - val_loss: 17.0248 - val_mae: 3.2698
Epoch 43/50
819/819 [==============================] - 7s 9ms/step - loss: 1.5047 - mae: 0.9428 - val_loss: 17.1972 - val_mae: 3.2916
Epoch 44/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4923 - mae: 0.9376 - val_loss: 17.3157 - val_mae: 3.2891
Epoch 45/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4817 - mae: 0.9355 - val_loss: 16.9533 - val_mae: 3.2569
Epoch 46/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4633 - mae: 0.9294 - val_loss: 16.9981 - val_mae: 3.2582
Epoch 47/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4535 - mae: 0.9260 - val_loss: 17.5801 - val_mae: 3.3033
Epoch 48/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4425 - mae: 0.9228 - val_loss: 16.9507 - val_mae: 3.2606
Epoch 49/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4358 - mae: 0.9197 - val_loss: 16.9329 - val_mae: 3.2601
Epoch 50/50
819/819 [==============================] - 7s 9ms/step - loss: 1.4296 - mae: 0.9179 - val_loss: 17.0347 - val_mae: 3.2619
```

</details>

### 5. 고급 기법

#### 5-1. 스태킹 순환 층(Stacking Recurrent Layer) 


- 모델의 표현 능력(Representational Power)을 증가 시킴  


```python
# 스태킹
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.GRU(32, return_sequences = True)(inputs)
x = keras.layers.GRU(32)(x)     
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer = 'rmsprop', loss = 'mse',  metrics = ['mae'])
history = model.fit(train_dataset,
                   epochs = 50,
                   validation_data = val_dataset)
```

<details>
  <summary>Result</summary>


    Epoch 1/50
    819/819 [==============================] - 19s 19ms/step - loss: 20.7605 - mae: 3.2645 - val_loss: 9.1614 - val_mae: 2.3476
    Epoch 2/50
    819/819 [==============================] - 15s 19ms/step - loss: 8.7637 - mae: 2.3062 - val_loss: 9.8531 - val_mae: 2.4330
    Epoch 3/50
    819/819 [==============================] - 15s 18ms/step - loss: 7.7038 - mae: 2.1745 - val_loss: 9.8144 - val_mae: 2.4462
    Epoch 4/50
    819/819 [==============================] - 15s 19ms/step - loss: 6.7814 - mae: 2.0463 - val_loss: 10.4869 - val_mae: 2.5180
    Epoch 5/50
    819/819 [==============================] - 15s 18ms/step - loss: 6.0027 - mae: 1.9226 - val_loss: 11.8817 - val_mae: 2.6752
    Epoch 6/50
    819/819 [==============================] - 15s 19ms/step - loss: 5.3034 - mae: 1.8030 - val_loss: 11.6635 - val_mae: 2.6475
    Epoch 7/50
    819/819 [==============================] - 15s 19ms/step - loss: 4.7214 - mae: 1.7000 - val_loss: 12.7908 - val_mae: 2.7771
    Epoch 8/50
    819/819 [==============================] - 15s 18ms/step - loss: 4.2568 - mae: 1.6142 - val_loss: 12.6994 - val_mae: 2.7767
    Epoch 9/50
    819/819 [==============================] - 15s 19ms/step - loss: 3.8598 - mae: 1.5376 - val_loss: 13.3369 - val_mae: 2.8303
    Epoch 10/50
    819/819 [==============================] - 15s 18ms/step - loss: 3.5110 - mae: 1.4655 - val_loss: 13.7982 - val_mae: 2.8919
    Epoch 11/50
    819/819 [==============================] - 15s 19ms/step - loss: 3.2218 - mae: 1.4038 - val_loss: 15.2973 - val_mae: 3.0374
    Epoch 12/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.9758 - mae: 1.3473 - val_loss: 14.7022 - val_mae: 2.9835
    Epoch 13/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.7767 - mae: 1.3011 - val_loss: 14.7804 - val_mae: 2.9903
    Epoch 14/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.6038 - mae: 1.2594 - val_loss: 15.2338 - val_mae: 3.0276
    Epoch 15/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.4555 - mae: 1.2220 - val_loss: 15.4960 - val_mae: 3.0618
    Epoch 16/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.3256 - mae: 1.1886 - val_loss: 15.5085 - val_mae: 3.0685
    Epoch 17/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.2218 - mae: 1.1607 - val_loss: 15.6405 - val_mae: 3.0674
    Epoch 18/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.1258 - mae: 1.1341 - val_loss: 15.3376 - val_mae: 3.0478
    Epoch 19/50
    819/819 [==============================] - 15s 19ms/step - loss: 2.0270 - mae: 1.1081 - val_loss: 15.7157 - val_mae: 3.0831
    Epoch 20/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.9528 - mae: 1.0862 - val_loss: 15.9530 - val_mae: 3.1123
    Epoch 21/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.8829 - mae: 1.0659 - val_loss: 16.0436 - val_mae: 3.0981
    Epoch 22/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.8144 - mae: 1.0465 - val_loss: 15.6861 - val_mae: 3.0832
    Epoch 23/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.7512 - mae: 1.0281 - val_loss: 15.5396 - val_mae: 3.0554
    Epoch 24/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.6968 - mae: 1.0118 - val_loss: 15.7263 - val_mae: 3.0845
    Epoch 25/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.6441 - mae: 0.9956 - val_loss: 15.8185 - val_mae: 3.0969
    Epoch 26/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.6024 - mae: 0.9828 - val_loss: 15.6266 - val_mae: 3.0643
    Epoch 27/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.5656 - mae: 0.9711 - val_loss: 16.1176 - val_mae: 3.1165
    Epoch 28/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.5224 - mae: 0.9567 - val_loss: 16.1985 - val_mae: 3.1160
    Epoch 29/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.4866 - mae: 0.9450 - val_loss: 16.2315 - val_mae: 3.1157
    Epoch 30/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.4554 - mae: 0.9354 - val_loss: 15.6923 - val_mae: 3.0814
    Epoch 31/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.4230 - mae: 0.9244 - val_loss: 16.3601 - val_mae: 3.1201
    Epoch 32/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.3973 - mae: 0.9167 - val_loss: 15.7678 - val_mae: 3.0782
    Epoch 33/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.3675 - mae: 0.9069 - val_loss: 16.1809 - val_mae: 3.1073
    Epoch 34/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.3385 - mae: 0.8966 - val_loss: 15.9925 - val_mae: 3.0916
    Epoch 35/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.3164 - mae: 0.8896 - val_loss: 16.3942 - val_mae: 3.1308
    Epoch 36/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.2913 - mae: 0.8805 - val_loss: 15.9569 - val_mae: 3.0910
    Epoch 37/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.2710 - mae: 0.8728 - val_loss: 15.8651 - val_mae: 3.0886
    Epoch 38/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.2548 - mae: 0.8680 - val_loss: 16.1707 - val_mae: 3.1234
    Epoch 39/50
    819/819 [==============================] - 15s 19ms/step - loss: 1.2344 - mae: 0.8609 - val_loss: 16.0831 - val_mae: 3.1057
    Epoch 40/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.2139 - mae: 0.8540 - val_loss: 16.1603 - val_mae: 3.1134
    Epoch 41/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1970 - mae: 0.8469 - val_loss: 16.2911 - val_mae: 3.1385
    Epoch 42/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1832 - mae: 0.8425 - val_loss: 16.0256 - val_mae: 3.1028
    Epoch 43/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1776 - mae: 0.8394 - val_loss: 16.5835 - val_mae: 3.1628
    Epoch 44/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1531 - mae: 0.8308 - val_loss: 16.1866 - val_mae: 3.1171
    Epoch 45/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1441 - mae: 0.8280 - val_loss: 16.3154 - val_mae: 3.1201
    Epoch 46/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1284 - mae: 0.8216 - val_loss: 16.4925 - val_mae: 3.1382
    Epoch 47/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1175 - mae: 0.8179 - val_loss: 16.1988 - val_mae: 3.1188
    Epoch 48/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.1048 - mae: 0.8131 - val_loss: 16.2245 - val_mae: 3.1315
    Epoch 49/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.0957 - mae: 0.8102 - val_loss: 16.6803 - val_mae: 3.1570
    Epoch 50/50
    819/819 [==============================] - 15s 18ms/step - loss: 1.0792 - mae: 0.8038 - val_loss: 16.1994 - val_mae: 3.1220

</details>

#### 5-2. 순환 드롭아웃(Recurrent Dropout) 

- 드롭아웃의 한 종류로 순환 층에서 과대적합을 방지하기 위해 사용

- 조건 : 모든 중간층은 전체 출력 시퀀스를 반환해야 함 [return_sequence = True]

```python
# 스태킹 + 드롭아웃
inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.GRU(32, recurrent_dropout = 0.5, return_sequences = True)(inputs)
x = keras.layers.GRU(32, recurrent_dropout = 0.5)(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['mae'])
history = model.fit(train_dataset,
                   epochs = 50,
                   validation_data = val_dataset)
```

<details>
<summary>Result</summary>


```
    WARNING:tensorflow:Layer gru_3 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
    WARNING:tensorflow:Layer gru_4 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
    Epoch 1/50
    819/819 [==============================] - 504s 612ms/step - loss: 25.6638 - mae: 3.7360 - val_loss: 9.7905 - val_mae: 2.4143
    Epoch 2/50
    819/819 [==============================] - 494s 603ms/step - loss: 14.0098 - mae: 2.9012 - val_loss: 9.7943 - val_mae: 2.4303
    Epoch 3/50
    819/819 [==============================] - 496s 606ms/step - loss: 13.1811 - mae: 2.8184 - val_loss: 9.6573 - val_mae: 2.4038
    Epoch 4/50
    819/819 [==============================] - 500s 610ms/step - loss: 12.5538 - mae: 2.7486 - val_loss: 9.1804 - val_mae: 2.3410
    Epoch 5/50
    819/819 [==============================] - 497s 607ms/step - loss: 12.0851 - mae: 2.6958 - val_loss: 10.2158 - val_mae: 2.4681
    Epoch 6/50
    819/819 [==============================] - 500s 610ms/step - loss: 11.6978 - mae: 2.6546 - val_loss: 9.2993 - val_mae: 2.3602
    Epoch 7/50
    819/819 [==============================] - 497s 607ms/step - loss: 11.3450 - mae: 2.6168 - val_loss: 10.1305 - val_mae: 2.4635
    Epoch 8/50
    819/819 [==============================] - 499s 610ms/step - loss: 10.9783 - mae: 2.5760 - val_loss: 9.7279 - val_mae: 2.4007
    Epoch 9/50
    819/819 [==============================] - 496s 605ms/step - loss: 10.7272 - mae: 2.5437 - val_loss: 9.7634 - val_mae: 2.4128
    Epoch 10/50
    819/819 [==============================] - 498s 608ms/step - loss: 10.4038 - mae: 2.5067 - val_loss: 10.3190 - val_mae: 2.4831
    Epoch 11/50
    819/819 [==============================] - 494s 603ms/step - loss: 10.1392 - mae: 2.4784 - val_loss: 9.9239 - val_mae: 2.4310
    Epoch 12/50
    819/819 [==============================] - 494s 604ms/step - loss: 9.9004 - mae: 2.4492 - val_loss: 10.7084 - val_mae: 2.5135
    Epoch 13/50
    819/819 [==============================] - 497s 606ms/step - loss: 9.6588 - mae: 2.4198 - val_loss: 11.5962 - val_mae: 2.6132
    Epoch 14/50
    819/819 [==============================] - 497s 607ms/step - loss: 9.5097 - mae: 2.4033 - val_loss: 10.3947 - val_mae: 2.4738
    Epoch 15/50
    819/819 [==============================] - 494s 603ms/step - loss: 9.2832 - mae: 2.3738 - val_loss: 10.6319 - val_mae: 2.5232
    Epoch 16/50
    819/819 [==============================] - 500s 610ms/step - loss: 9.1200 - mae: 2.3525 - val_loss: 11.2430 - val_mae: 2.5823
    Epoch 17/50
    819/819 [==============================] - 494s 603ms/step - loss: 8.9038 - mae: 2.3254 - val_loss: 10.6121 - val_mae: 2.5164
    Epoch 18/50
    819/819 [==============================] - 493s 602ms/step - loss: 8.8183 - mae: 2.3126 - val_loss: 11.4088 - val_mae: 2.5978
    Epoch 19/50
    819/819 [==============================] - 500s 610ms/step - loss: 8.6260 - mae: 2.2896 - val_loss: 12.2087 - val_mae: 2.7005
    Epoch 20/50
    819/819 [==============================] - 500s 611ms/step - loss: 8.5349 - mae: 2.2769 - val_loss: 11.9267 - val_mae: 2.6670
    Epoch 21/50
    819/819 [==============================] - 499s 609ms/step - loss: 8.4812 - mae: 2.2655 - val_loss: 11.4025 - val_mae: 2.6139
    Epoch 22/50
    819/819 [==============================] - 502s 613ms/step - loss: 8.3706 - mae: 2.2529 - val_loss: 11.6848 - val_mae: 2.6377
    Epoch 23/50
    819/819 [==============================] - 496s 606ms/step - loss: 8.2790 - mae: 2.2400 - val_loss: 11.6981 - val_mae: 2.6444
    Epoch 24/50
    819/819 [==============================] - 496s 606ms/step - loss: 8.1903 - mae: 2.2248 - val_loss: 11.6473 - val_mae: 2.6349
    Epoch 25/50
    819/819 [==============================] - 500s 610ms/step - loss: 8.1033 - mae: 2.2164 - val_loss: 11.7885 - val_mae: 2.6498
    Epoch 26/50
    819/819 [==============================] - 499s 609ms/step - loss: 8.0311 - mae: 2.2074 - val_loss: 12.4579 - val_mae: 2.7241
    Epoch 27/50
    819/819 [==============================] - 495s 605ms/step - loss: 7.9373 - mae: 2.1937 - val_loss: 12.3874 - val_mae: 2.7021
    Epoch 28/50
    819/819 [==============================] - 497s 607ms/step - loss: 7.9238 - mae: 2.1889 - val_loss: 12.2006 - val_mae: 2.7040
    Epoch 29/50
    819/819 [==============================] - 498s 608ms/step - loss: 7.7956 - mae: 2.1720 - val_loss: 12.9720 - val_mae: 2.7740
    Epoch 30/50
    819/819 [==============================] - 492s 601ms/step - loss: 7.7375 - mae: 2.1653 - val_loss: 12.6591 - val_mae: 2.7516
    Epoch 31/50
    819/819 [==============================] - 494s 603ms/step - loss: 7.6859 - mae: 2.1581 - val_loss: 12.0321 - val_mae: 2.6811
    Epoch 32/50
    819/819 [==============================] - 497s 607ms/step - loss: 7.6583 - mae: 2.1525 - val_loss: 12.8679 - val_mae: 2.7813
    Epoch 33/50
    819/819 [==============================] - 499s 610ms/step - loss: 7.6339 - mae: 2.1460 - val_loss: 12.2878 - val_mae: 2.7077
    Epoch 34/50
    819/819 [==============================] - 496s 605ms/step - loss: 7.5489 - mae: 2.1357 - val_loss: 12.1099 - val_mae: 2.6958
    Epoch 35/50
    819/819 [==============================] - 497s 607ms/step - loss: 7.5323 - mae: 2.1337 - val_loss: 12.2774 - val_mae: 2.7080
    Epoch 36/50
    819/819 [==============================] - 503s 614ms/step - loss: 7.4458 - mae: 2.1202 - val_loss: 12.2762 - val_mae: 2.7244
    Epoch 37/50
    819/819 [==============================] - 507s 619ms/step - loss: 7.3962 - mae: 2.1143 - val_loss: 12.5186 - val_mae: 2.7426
    Epoch 38/50
    819/819 [==============================] - 506s 617ms/step - loss: 7.4129 - mae: 2.1139 - val_loss: 13.0123 - val_mae: 2.7951
    Epoch 39/50
    819/819 [==============================] - 501s 612ms/step - loss: 7.3612 - mae: 2.1065 - val_loss: 12.4012 - val_mae: 2.7309
    Epoch 40/50
    819/819 [==============================] - 508s 620ms/step - loss: 7.3229 - mae: 2.0987 - val_loss: 12.3146 - val_mae: 2.7286
    Epoch 41/50
    819/819 [==============================] - 504s 615ms/step - loss: 7.3051 - mae: 2.0977 - val_loss: 12.7901 - val_mae: 2.7583
    Epoch 42/50
    819/819 [==============================] - 505s 617ms/step - loss: 7.2431 - mae: 2.0903 - val_loss: 12.9836 - val_mae: 2.7910
    Epoch 43/50
    819/819 [==============================] - 509s 621ms/step - loss: 7.2648 - mae: 2.0926 - val_loss: 12.7437 - val_mae: 2.7526
    Epoch 44/50
    819/819 [==============================] - 507s 619ms/step - loss: 7.1886 - mae: 2.0828 - val_loss: 13.5040 - val_mae: 2.8450
    Epoch 45/50
    819/819 [==============================] - 506s 617ms/step - loss: 7.1726 - mae: 2.0787 - val_loss: 13.0302 - val_mae: 2.7963
    Epoch 46/50
    819/819 [==============================] - 504s 616ms/step - loss: 7.1553 - mae: 2.0768 - val_loss: 12.8353 - val_mae: 2.7626
    Epoch 47/50
    819/819 [==============================] - 511s 624ms/step - loss: 7.1000 - mae: 2.0688 - val_loss: 12.8553 - val_mae: 2.7734
    Epoch 48/50
    819/819 [==============================] - 509s 622ms/step - loss: 7.0557 - mae: 2.0587 - val_loss: 13.4551 - val_mae: 2.8400
    Epoch 49/50
    819/819 [==============================] - 508s 620ms/step - loss: 7.0751 - mae: 2.0597 - val_loss: 12.3828 - val_mae: 2.7391
    Epoch 50/50
    819/819 [==============================] - 507s 618ms/step - loss: 7.0688 - mae: 2.0593 - val_loss: 13.0976 - val_mae: 2.8137
```

</details>

#### 5-3. Early Stopping

- 너무 많은 Epoch는 Overfitting을 일으킨다
- 너무 적은 Epoch는 Underfitting을 일으킨다
- Epoch를 많이 돌린 후, 특정 시점에서 멈춤
  - monitor : 모델 학습을 멈추는 기준
  - patience : 최소 반복수
  - verbose : 자세한 정보 표시 모드 ( 0 or 1 )

```python
# Early Stopping
from keras.callbacks import EarlyStopping

inputs = keras.Input(shape = (sequence_length, raw_data.shape[-1]))
x = keras.layers.GRU(32, recurrent_dropout = 0.5, return_sequences = True)(inputs)
x = keras.layers.GRU(32, recurrent_dropout = 0.5)(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['mae'])
history = model.fit(train_dataset,
                   epochs = 50,
                   validation_data = val_dataset,
                   callbacks = [early_stopping])
```

```
WARNING:tensorflow:Layer gru will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
WARNING:tensorflow:Layer gru_1 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
Epoch 1/50
819/819 [==============================] - 501s 608ms/step - loss: 27.5714 - mae: 3.8488 - val_loss: 9.9952 - val_mae: 2.4443
Epoch 2/50
819/819 [==============================] - 497s 607ms/step - loss: 13.9617 - mae: 2.8960 - val_loss: 8.9847 - val_mae: 2.3212
Epoch 3/50
819/819 [==============================] - 496s 605ms/step - loss: 13.1481 - mae: 2.8131 - val_loss: 8.9414 - val_mae: 2.3150
Epoch 4/50
819/819 [==============================] - 494s 603ms/step - loss: 12.6829 - mae: 2.7626 - val_loss: 8.8438 - val_mae: 2.3150
Epoch 5/50
819/819 [==============================] - 494s 603ms/step - loss: 12.2708 - mae: 2.7164 - val_loss: 9.1045 - val_mae: 2.3352
Epoch 6/50
819/819 [==============================] - 493s 602ms/step - loss: 11.8606 - mae: 2.6714 - val_loss: 9.5416 - val_mae: 2.3956
Epoch 7/50
819/819 [==============================] - 498s 608ms/step - loss: 11.4571 - mae: 2.6260 - val_loss: 9.2911 - val_mae: 2.3589
Epoch 8/50
819/819 [==============================] - 497s 607ms/step - loss: 11.1376 - mae: 2.5875 - val_loss: 9.0102 - val_mae: 2.3324
Epoch 9/50
819/819 [==============================] - 493s 602ms/step - loss: 10.8352 - mae: 2.5561 - val_loss: 9.3462 - val_mae: 2.3681
Epoch 00009: early stopping
```

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

```python
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

## III. 전염병 예측_V2 [ 시계열 데이터 일반화 ]

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

```python
# github 저장소에서 데이터 불러오기
# !git clone https://github.com/yhlee1627/deeplearning.git
# git이 안될경우 주소에서 다운로드

df = pd.read_csv('./deeplearning/corona_daily.csv', usecols = [3], engine = 'python', skipfooter = 3)
dataframe = df.values
dataframe = df.astype('float32')
```

| Confirmed |       |
| --------: | ----- |
|         0 | 24    |
|         1 | 24    |
|         2 | 27    |
|         3 | 27    |
|         4 | 28    |
|       ... | ...   |
|       107 | 11190 |
|       108 | 11206 |
|       109 | 11225 |
|       110 | 11265 |
|       111 | 11344 |

112 rows × 1 columns

### 2. 데이터 전처리

#### 2-1. 데이터 분할

- 각 분할에 사용할 샘플 개수 계산
  - Train Data : 50%
  - Val Data : 25%
  - Test Data : 25%

```python
train_samples_n = int(0.5 * len(df))
val_samples_n = int(0.25 * len(df))
test_samples_n = len(df) - train_samples_n - val_samples_n
print(f"Train Data 개수 : {train_samples_n}")
print(f"Val Data 개수 : {val_samples_n}")
print(f"Test Data 개수 : {test_samples_n}")
```

```
Train Data 개수 : 56
Val Data 개수 : 28
Test Data 개수 : 28
```

#### 2-2. 데이터 정규화

```python
# 시계열은 스케일이 각각 다르므로 독립적으로 정규화(평균과 표준편차 사용)
mean = df[:train_samples_n].mean(axis = 0)
df -= mean
std = df[:train_samples_n].std(axis = 0)
df /= std
```

#### 2-3. 시퀀스 데이터 준비 및 분할

- 3일치의 데이터를 사용하여 4번째 날짜의 값을 예측

  <img src='https://p.ipic.vip/0wzzyd.png' width=50%>

- 과정

  - I. 현재 데이터의 시간 단위(1일)와 예측하려는 시간 단위 (1일)을 통일
    - $sampling\_rate = 1$
  - II. 사용할 데이터의 기간 정의(3일)
    - $sequenc\_length = 3$
  - III. 예측 시점 하루 뒤
    - $delay = sampling\_rate * (sequence\_length + 1 - 1)$

<img src='https://p.ipic.vip/cg3n6r.png'>

- timeseries_dataset_from_array() : 중복된 데이터 때문에 생기는 메모리 낭비를 줄여줌

<details>
<summary>Option</summary>



```python
tf.keras.utils.timeseries_dataset_from_array(
        data,
        targets,
        sampling_rate = 1,
			  sequence_length,
        delay = sampling_rate * (sequence_length + 24 - 1)      
        sequence_stride = 1,
        batch_size = 128,
        shuffle = Fasle,
        seed None,
        start_index = None,
        end_index = None
    )
```


| Option                       |              Explanation              |
| :--------------------------- | :-----------------------------------: |
| data                         |      타깃 데이터를 제외한 데이터      |
| targets                      |              타깃 데이터              |
| sampling_rate                |         시퀀스 데이터의 단위          |
| sequence_length              |  훈련에 사용할 시퀀스 데이터의 길이   |
| delay                        |         예측하고자 하는 시점          |
| sequence_stride(default = 1) |         연속 시계열 간의 거리         |
| batch_size(default = 128)    | 각 배치의 시계열 샘플 수(마지막 제외) |
| shuffle                      |        출력 샘플을 섞을지 말지        |
| seed                         |                고정값                 |
| start_index                  |   사용할 데이터의 시작 인덱스 위치    |
| end_index                    |    사용할 데이터의 끝 인덱스 위치     |

</details>

```python
from tensorflow import keras

sampling_rate = 1  
sequence_length = 3                                    # 이전 3일의 데이터 사용
delay = sampling_rate * (sequence_length + 1 - 1)      # 하루 뒤 
batch_size = 54                                        # 이 데이터에 최대 batch_size = 54

# Training Data : 0 ~ train_samples_n
train_dataset = keras.utils.timeseries_dataset_from_array(
    df[:-delay],
    targets = df[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    batch_size = batch_size,
    start_index = 0,
    end_index = train_samples_n
)

# Validation Data : train_samples_n ~ train_samples_n + val_samples_n
val_dataset = keras.utils.timeseries_dataset_from_array(
    df[:-delay],
    targets = df[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    batch_size = batch_size,
    start_index = train_samples_n,
    end_index = train_samples_n + val_samples_n
)

# Test Data : train_samples_n + val_sampes_n + End
test_dataset = keras.utils.timeseries_dataset_from_array(
    df[:-delay],
    targets = df[delay:],
    sampling_rate = sampling_rate,
    sequence_length = sequence_length,
    batch_size = batch_size,
    start_index = train_samples_n + val_samples_n
)
```

```python
for samples, targets in train_dataset:
    print(f"샘플 크기 : {samples.shape}")
    print(f"타깃 크기 : {targets.shape}")
    break
```

```
샘플 크기 : (54, 3, 1)
타깃 크기 : (54, 1)
```

→ 샘플 크기 : (batch_size, sequence_length, column 개수)

​	-> 데이터의 크기가 작아 batch_size가 최대 54

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



<img src='https://p.ipic.vip/2bdtgi.png'>

</details>


```python
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.SimpleRNN(16, recurrent_dropout = 0.25)(inputs) 
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 3, 1)]            0         
                                                                 
 simple_rnn_1 (SimpleRNN)    (None, 16)                288       
                                                                 
 dense_15 (Dense)            (None, 1)                 17        
                                                                 
=================================================================
Total params: 305
Trainable params: 305
Non-trainable params: 0
_________________________________________________________________

```

####  3-2. LSTM ; Long Short Term Memory

- RNN 의 **장기의존성 문제**와 **기울기 소실 문제**를 해결한 알고리즘

  - 가중치 행렬 $W$의 행렬 곱 연산이 그레이디언트 경로에 나타나지 않도록 구조 변경

- 기존 RNN에 장기 기억 셀(Cell State)을 추가함

  - $c_t$를 연결하는 경로에는 가중치 행렬 $W$의 행렬 곱 연산이 없음

    <img src='https://p.ipic.vip/hiuty3.png' width=80%>

- 장기 기억 셀 연산에 사용되는 게이트 추가

  - Forget Gate($f_t$) : 과거의 정보를 얼마나 유지할 것인가?
  - Input Gate($i_t$) : 새로 입력된 정보를 얼만큼 활용할 것인가?
  - Output Gate($o_t$) : Cell State 나온 정보를 얼마나 출력할 것인가?

<details>
<summary>LSTM Process</summary>
<img src='https://p.ipic.vip/elqu30.png'>
</details>


```python
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.LSTM(32, recurrent_dropout = 0.25)(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 3, 1)]            0         
                                                                 
 lstm_13 (LSTM)              (None, 32)                4352      
                                                                 
 dense_16 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 4,385
Trainable params: 4,385
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
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.GRU(32, recurrent_dropout = 0.25)(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 3, 1)]            0         
                                                                 
 gru_1 (GRU)                 (None, 32)                3360      
                                                                 
 dense_17 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 3,393
Trainable params: 3,393
Non-trainable params: 0
_________________________________________________________________
```



#### 3-4. 양방향(Bidirectional) LSTM/GRU

- 양방향 순환 층(Bidirectional Recurrent Layer)
  - 순환 네트워크에 같은 정보를 다른 방향으로 주입하여 정확도를 높이고 기억을 좀 더 오래 유지
  - 이전의 층이 전체 출력 시퀀스를 반환해야 함 [ return_sequence = True ]

<details>
<summary>양방향 LSTM/GRU</summary>



<img src='https://p.ipic.vip/t6kz5h.png'>

</details>

```python
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.Bidirectional(keras.layers.LSTM(16))(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_4 (InputLayer)        [(None, 3, 1)]            0         
                                                                 
 bidirectional_1 (Bidirectio  (None, 32)               2304      
 nal)                                                            
                                                                 
 dense_18 (Dense)            (None, 1)                 33        
                                                                 
=================================================================
Total params: 2,337
Trainable params: 2,337
Non-trainable params: 0
_________________________________________________________________
```



#### 3-5. 1D CNN + LSTM/GRU

<details>
<summary>1D CNN + LSTM/GRU</summary>
<img src='https://p.ipic.vip/bzx124.png'>
</details>

```python
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.Conv1D(filters=1,
               kernel_size=1,
               strides=1,
               activation='relu')(inputs)
x = keras.layers.MaxPooling1D(pool_size = 3)(x)
x = keras.layers.LSTM(32, recurrent_dropout = 0.25)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
```

```
Model: "model_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_5 (InputLayer)        [(None, 3, 1)]            0         
                                                                 
 conv1d (Conv1D)             (None, 3, 32)             64        
                                                                 
 max_pooling1d (MaxPooling1D  (None, 1, 32)            0         
 )                                                               
                                                                 
 lstm_2 (LSTM)               (None, 32)                8320      
                                                                 
 dense_4 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
Total params: 8,417
Trainable params: 8,417
Non-trainable params: 0
_________________________________________________________________
```

### 4. 모델 학습

```python
model.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['mae'])
history = model.fit(train_dataset,
                   epochs = 100,
                   validation_data = val_dataset)
```

<details>
  <summary>Result</summary>


```
Epoch 1/100
1/1 [==============================] - 2s 2s/step - loss: 0.6861 - mae: 0.7655 - val_loss: 1.5458 - val_mae: 1.2428
Epoch 2/100
1/1 [==============================] - 0s 47ms/step - loss: 0.6743 - mae: 0.7585 - val_loss: 1.5337 - val_mae: 1.2379
Epoch 3/100
1/1 [==============================] - 0s 53ms/step - loss: 0.6655 - mae: 0.7533 - val_loss: 1.5248 - val_mae: 1.2343
Epoch 4/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6580 - mae: 0.7488 - val_loss: 1.5142 - val_mae: 1.2300
Epoch 5/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6512 - mae: 0.7447 - val_loss: 1.5078 - val_mae: 1.2274
Epoch 6/100
1/1 [==============================] - 0s 52ms/step - loss: 0.6450 - mae: 0.7410 - val_loss: 1.4979 - val_mae: 1.2234
Epoch 7/100
1/1 [==============================] - 0s 52ms/step - loss: 0.6391 - mae: 0.7373 - val_loss: 1.4919 - val_mae: 1.2209
Epoch 8/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6334 - mae: 0.7339 - val_loss: 1.4832 - val_mae: 1.2174
Epoch 9/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6279 - mae: 0.7305 - val_loss: 1.4769 - val_mae: 1.2148
Epoch 10/100
1/1 [==============================] - 0s 53ms/step - loss: 0.6226 - mae: 0.7272 - val_loss: 1.4691 - val_mae: 1.2116
Epoch 11/100
1/1 [==============================] - 0s 52ms/step - loss: 0.6175 - mae: 0.7240 - val_loss: 1.4627 - val_mae: 1.2089
Epoch 12/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6124 - mae: 0.7208 - val_loss: 1.4554 - val_mae: 1.2059
Epoch 13/100
1/1 [==============================] - 0s 48ms/step - loss: 0.6074 - mae: 0.7177 - val_loss: 1.4489 - val_mae: 1.2032
Epoch 14/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6024 - mae: 0.7146 - val_loss: 1.4418 - val_mae: 1.2002
Epoch 15/100
1/1 [==============================] - 0s 51ms/step - loss: 0.5975 - mae: 0.7115 - val_loss: 1.4354 - val_mae: 1.1976
Epoch 16/100
1/1 [==============================] - 0s 49ms/step - loss: 0.5927 - mae: 0.7084 - val_loss: 1.4284 - val_mae: 1.1946
Epoch 17/100
1/1 [==============================] - 0s 50ms/step - loss: 0.5878 - mae: 0.7053 - val_loss: 1.4220 - val_mae: 1.1920
Epoch 18/100
1/1 [==============================] - 0s 50ms/step - loss: 0.5830 - mae: 0.7023 - val_loss: 1.4150 - val_mae: 1.1890
Epoch 19/100
1/1 [==============================] - 0s 48ms/step - loss: 0.5783 - mae: 0.6992 - val_loss: 1.4088 - val_mae: 1.1864
Epoch 20/100
1/1 [==============================] - 0s 48ms/step - loss: 0.5735 - mae: 0.6962 - val_loss: 1.4015 - val_mae: 1.1833
Epoch 21/100
1/1 [==============================] - 0s 47ms/step - loss: 0.5688 - mae: 0.6931 - val_loss: 1.3957 - val_mae: 1.1809
Epoch 22/100
1/1 [==============================] - 0s 49ms/step - loss: 0.5641 - mae: 0.6901 - val_loss: 1.3878 - val_mae: 1.1775
Epoch 23/100
1/1 [==============================] - 0s 48ms/step - loss: 0.5594 - mae: 0.6870 - val_loss: 1.3826 - val_mae: 1.1753
Epoch 24/100
1/1 [==============================] - 0s 50ms/step - loss: 0.5546 - mae: 0.6839 - val_loss: 1.3742 - val_mae: 1.1717
Epoch 25/100
1/1 [==============================] - 0s 48ms/step - loss: 0.5499 - mae: 0.6808 - val_loss: 1.3692 - val_mae: 1.1696
Epoch 26/100
1/1 [==============================] - 0s 50ms/step - loss: 0.5452 - mae: 0.6777 - val_loss: 1.3607 - val_mae: 1.1660
Epoch 27/100
1/1 [==============================] - 0s 50ms/step - loss: 0.5405 - mae: 0.6746 - val_loss: 1.3555 - val_mae: 1.1637
Epoch 28/100
1/1 [==============================] - 0s 48ms/step - loss: 0.5358 - mae: 0.6715 - val_loss: 1.3472 - val_mae: 1.1602
Epoch 29/100
1/1 [==============================] - 0s 47ms/step - loss: 0.5311 - mae: 0.6683 - val_loss: 1.3417 - val_mae: 1.1578
Epoch 30/100
1/1 [==============================] - 0s 47ms/step - loss: 0.5264 - mae: 0.6651 - val_loss: 1.3336 - val_mae: 1.1543
Epoch 31/100
1/1 [==============================] - 0s 47ms/step - loss: 0.5217 - mae: 0.6619 - val_loss: 1.3278 - val_mae: 1.1518
Epoch 32/100
1/1 [==============================] - 0s 49ms/step - loss: 0.5170 - mae: 0.6587 - val_loss: 1.3197 - val_mae: 1.1482
Epoch 33/100
1/1 [==============================] - 0s 48ms/step - loss: 0.5122 - mae: 0.6555 - val_loss: 1.3139 - val_mae: 1.1457
Epoch 34/100
1/1 [==============================] - 0s 52ms/step - loss: 0.5075 - mae: 0.6522 - val_loss: 1.3056 - val_mae: 1.1421
Epoch 35/100
1/1 [==============================] - 0s 51ms/step - loss: 0.5028 - mae: 0.6489 - val_loss: 1.2998 - val_mae: 1.1396
Epoch 36/100
1/1 [==============================] - 0s 51ms/step - loss: 0.4980 - mae: 0.6457 - val_loss: 1.2914 - val_mae: 1.1358
Epoch 37/100
1/1 [==============================] - 0s 50ms/step - loss: 0.4933 - mae: 0.6423 - val_loss: 1.2857 - val_mae: 1.1333
Epoch 38/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4885 - mae: 0.6390 - val_loss: 1.2769 - val_mae: 1.1295
Epoch 39/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4837 - mae: 0.6356 - val_loss: 1.2714 - val_mae: 1.1270
Epoch 40/100
1/1 [==============================] - 0s 48ms/step - loss: 0.4789 - mae: 0.6323 - val_loss: 1.2624 - val_mae: 1.1230
Epoch 41/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4742 - mae: 0.6288 - val_loss: 1.2569 - val_mae: 1.1206
Epoch 42/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4694 - mae: 0.6255 - val_loss: 1.2478 - val_mae: 1.1165
Epoch 43/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4646 - mae: 0.6219 - val_loss: 1.2421 - val_mae: 1.1140
Epoch 44/100
1/1 [==============================] - 0s 48ms/step - loss: 0.4598 - mae: 0.6185 - val_loss: 1.2331 - val_mae: 1.1099
Epoch 45/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4549 - mae: 0.6150 - val_loss: 1.2272 - val_mae: 1.1072
Epoch 46/100
1/1 [==============================] - 0s 48ms/step - loss: 0.4501 - mae: 0.6115 - val_loss: 1.2182 - val_mae: 1.1032
Epoch 47/100
1/1 [==============================] - 0s 47ms/step - loss: 0.4453 - mae: 0.6079 - val_loss: 1.2122 - val_mae: 1.1004
Epoch 48/100
1/1 [==============================] - 0s 48ms/step - loss: 0.4405 - mae: 0.6043 - val_loss: 1.2032 - val_mae: 1.0963
Epoch 49/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4356 - mae: 0.6007 - val_loss: 1.1971 - val_mae: 1.0935
Epoch 50/100
1/1 [==============================] - 0s 52ms/step - loss: 0.4308 - mae: 0.5971 - val_loss: 1.1879 - val_mae: 1.0894
Epoch 51/100
1/1 [==============================] - 0s 52ms/step - loss: 0.4259 - mae: 0.5934 - val_loss: 1.1819 - val_mae: 1.0866
Epoch 52/100
1/1 [==============================] - 0s 50ms/step - loss: 0.4211 - mae: 0.5898 - val_loss: 1.1726 - val_mae: 1.0823
Epoch 53/100
1/1 [==============================] - 0s 48ms/step - loss: 0.4162 - mae: 0.5860 - val_loss: 1.1665 - val_mae: 1.0795
Epoch 54/100
1/1 [==============================] - 0s 48ms/step - loss: 0.4114 - mae: 0.5823 - val_loss: 1.1571 - val_mae: 1.0751
Epoch 55/100
1/1 [==============================] - 0s 57ms/step - loss: 0.4065 - mae: 0.5785 - val_loss: 1.1510 - val_mae: 1.0723
Epoch 56/100
1/1 [==============================] - 0s 49ms/step - loss: 0.4017 - mae: 0.5748 - val_loss: 1.1415 - val_mae: 1.0678
Epoch 57/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3968 - mae: 0.5709 - val_loss: 1.1354 - val_mae: 1.0650
Epoch 58/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3920 - mae: 0.5672 - val_loss: 1.1257 - val_mae: 1.0604
Epoch 59/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3871 - mae: 0.5632 - val_loss: 1.1196 - val_mae: 1.0575
Epoch 60/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3822 - mae: 0.5594 - val_loss: 1.1099 - val_mae: 1.0529
Epoch 61/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3774 - mae: 0.5554 - val_loss: 1.1037 - val_mae: 1.0500
Epoch 62/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3725 - mae: 0.5516 - val_loss: 1.0940 - val_mae: 1.0454
Epoch 63/100
1/1 [==============================] - 0s 47ms/step - loss: 0.3677 - mae: 0.5475 - val_loss: 1.0877 - val_mae: 1.0423
Epoch 64/100
1/1 [==============================] - 0s 46ms/step - loss: 0.3628 - mae: 0.5436 - val_loss: 1.0780 - val_mae: 1.0376
Epoch 65/100
1/1 [==============================] - 0s 46ms/step - loss: 0.3580 - mae: 0.5395 - val_loss: 1.0716 - val_mae: 1.0346
Epoch 66/100
1/1 [==============================] - 0s 48ms/step - loss: 0.3531 - mae: 0.5355 - val_loss: 1.0618 - val_mae: 1.0298
Epoch 67/100
1/1 [==============================] - 0s 48ms/step - loss: 0.3483 - mae: 0.5314 - val_loss: 1.0554 - val_mae: 1.0267
Epoch 68/100
1/1 [==============================] - 0s 50ms/step - loss: 0.3435 - mae: 0.5274 - val_loss: 1.0455 - val_mae: 1.0219
Epoch 69/100
1/1 [==============================] - 0s 51ms/step - loss: 0.3387 - mae: 0.5232 - val_loss: 1.0391 - val_mae: 1.0188
Epoch 70/100
1/1 [==============================] - 0s 45ms/step - loss: 0.3338 - mae: 0.5191 - val_loss: 1.0292 - val_mae: 1.0139
Epoch 71/100
1/1 [==============================] - 0s 45ms/step - loss: 0.3290 - mae: 0.5149 - val_loss: 1.0228 - val_mae: 1.0107
Epoch 72/100
1/1 [==============================] - 0s 46ms/step - loss: 0.3242 - mae: 0.5108 - val_loss: 1.0127 - val_mae: 1.0057
Epoch 73/100
1/1 [==============================] - 0s 45ms/step - loss: 0.3195 - mae: 0.5065 - val_loss: 1.0063 - val_mae: 1.0025
Epoch 74/100
1/1 [==============================] - 0s 45ms/step - loss: 0.3147 - mae: 0.5023 - val_loss: 0.9962 - val_mae: 0.9975
Epoch 75/100
1/1 [==============================] - 0s 45ms/step - loss: 0.3099 - mae: 0.4980 - val_loss: 0.9898 - val_mae: 0.9942
Epoch 76/100
1/1 [==============================] - 0s 45ms/step - loss: 0.3052 - mae: 0.4937 - val_loss: 0.9796 - val_mae: 0.9891
Epoch 77/100
1/1 [==============================] - 0s 54ms/step - loss: 0.3004 - mae: 0.4895 - val_loss: 0.9731 - val_mae: 0.9858
Epoch 78/100
1/1 [==============================] - 0s 45ms/step - loss: 0.2957 - mae: 0.4852 - val_loss: 0.9629 - val_mae: 0.9807
Epoch 79/100
1/1 [==============================] - 0s 46ms/step - loss: 0.2910 - mae: 0.4808 - val_loss: 0.9564 - val_mae: 0.9773
Epoch 80/100
1/1 [==============================] - 0s 45ms/step - loss: 0.2863 - mae: 0.4765 - val_loss: 0.9462 - val_mae: 0.9721
Epoch 81/100
1/1 [==============================] - 0s 45ms/step - loss: 0.2816 - mae: 0.4721 - val_loss: 0.9397 - val_mae: 0.9687
Epoch 82/100
1/1 [==============================] - 0s 45ms/step - loss: 0.2770 - mae: 0.4678 - val_loss: 0.9295 - val_mae: 0.9634
Epoch 83/100
1/1 [==============================] - 0s 44ms/step - loss: 0.2723 - mae: 0.4633 - val_loss: 0.9228 - val_mae: 0.9600
Epoch 84/100
1/1 [==============================] - 0s 44ms/step - loss: 0.2677 - mae: 0.4589 - val_loss: 0.9126 - val_mae: 0.9547
Epoch 85/100
1/1 [==============================] - 0s 46ms/step - loss: 0.2631 - mae: 0.4544 - val_loss: 0.9060 - val_mae: 0.9512
Epoch 86/100
1/1 [==============================] - 0s 48ms/step - loss: 0.2585 - mae: 0.4499 - val_loss: 0.8957 - val_mae: 0.9458
Epoch 87/100
1/1 [==============================] - 0s 52ms/step - loss: 0.2540 - mae: 0.4454 - val_loss: 0.8890 - val_mae: 0.9422
Epoch 88/100
1/1 [==============================] - 0s 45ms/step - loss: 0.2494 - mae: 0.4409 - val_loss: 0.8788 - val_mae: 0.9368
Epoch 89/100
1/1 [==============================] - 0s 46ms/step - loss: 0.2449 - mae: 0.4363 - val_loss: 0.8721 - val_mae: 0.9332
Epoch 90/100
1/1 [==============================] - 0s 49ms/step - loss: 0.2404 - mae: 0.4318 - val_loss: 0.8618 - val_mae: 0.9276
Epoch 91/100
1/1 [==============================] - 0s 52ms/step - loss: 0.2359 - mae: 0.4273 - val_loss: 0.8550 - val_mae: 0.9240
Epoch 92/100
1/1 [==============================] - 0s 47ms/step - loss: 0.2315 - mae: 0.4227 - val_loss: 0.8448 - val_mae: 0.9184
Epoch 93/100
1/1 [==============================] - 0s 45ms/step - loss: 0.2271 - mae: 0.4182 - val_loss: 0.8380 - val_mae: 0.9147
Epoch 94/100
1/1 [==============================] - 0s 47ms/step - loss: 0.2227 - mae: 0.4136 - val_loss: 0.8277 - val_mae: 0.9091
Epoch 95/100
1/1 [==============================] - 0s 48ms/step - loss: 0.2183 - mae: 0.4090 - val_loss: 0.8209 - val_mae: 0.9054
Epoch 96/100
1/1 [==============================] - 0s 47ms/step - loss: 0.2140 - mae: 0.4044 - val_loss: 0.8106 - val_mae: 0.8997
Epoch 97/100
1/1 [==============================] - 0s 48ms/step - loss: 0.2097 - mae: 0.3998 - val_loss: 0.8038 - val_mae: 0.8959
Epoch 98/100
1/1 [==============================] - 0s 47ms/step - loss: 0.2054 - mae: 0.3951 - val_loss: 0.7936 - val_mae: 0.8901
Epoch 99/100
1/1 [==============================] - 0s 47ms/step - loss: 0.2012 - mae: 0.3904 - val_loss: 0.7868 - val_mae: 0.8863
Epoch 100/100
1/1 [==============================] - 0s 46ms/step - loss: 0.1969 - mae: 0.3857 - val_loss: 0.7765 - val_mae: 0.8805
```

</details>

### 5. 고급기법

#### 5-1. 스태킹 순환 층(Stacking Recurrent Layer) 


- 모델의 표현 능력(Representational Power)을 증가 시킴  


```python
# 스태킹
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.GRU(32, return_sequences = True)(inputs)
x = keras.layers.GRU(32)(x)
x = keras.layers.Dense(1)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer = 'rmsprop', loss = 'mse',  metrics = ['mae'])
history = model.fit(train_dataset,
                   epochs = 100,
                   validation_data = val_dataset)
```

<details>
  <summary>Result</summary>


    Epoch 1/100
    1/1 [==============================] - 2s 2s/step - loss: 1.0117 - mae: 0.9435 - val_loss: 1.6641 - val_mae: 1.2896
    Epoch 2/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.8312 - mae: 0.8506 - val_loss: 1.3773 - val_mae: 1.1732
    Epoch 3/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.7144 - mae: 0.7856 - val_loss: 1.1652 - val_mae: 1.0791
    Epoch 4/100
    1/1 [==============================] - 0s 54ms/step - loss: 0.6207 - mae: 0.7301 - val_loss: 0.9933 - val_mae: 0.9963
    Epoch 5/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.5402 - mae: 0.6794 - val_loss: 0.8468 - val_mae: 0.9199
    Epoch 6/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.4685 - mae: 0.6313 - val_loss: 0.7183 - val_mae: 0.8473
    Epoch 7/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.4039 - mae: 0.5848 - val_loss: 0.6042 - val_mae: 0.7771
    Epoch 8/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.3453 - mae: 0.5395 - val_loss: 0.5020 - val_mae: 0.7083
    Epoch 9/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.2920 - mae: 0.4951 - val_loss: 0.4114 - val_mae: 0.6412
    Epoch 10/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.2441 - mae: 0.4516 - val_loss: 0.3316 - val_mae: 0.5756
    Epoch 11/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.2013 - mae: 0.4087 - val_loss: 0.2620 - val_mae: 0.5117
    Epoch 12/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.1635 - mae: 0.3673 - val_loss: 0.2024 - val_mae: 0.4498
    Epoch 13/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.1307 - mae: 0.3282 - val_loss: 0.1525 - val_mae: 0.3904
    Epoch 14/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.1027 - mae: 0.2911 - val_loss: 0.1116 - val_mae: 0.3339
    Epoch 15/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0794 - mae: 0.2563 - val_loss: 0.0791 - val_mae: 0.2812
    Epoch 16/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0605 - mae: 0.2235 - val_loss: 0.0542 - val_mae: 0.2328
    Epoch 17/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0456 - mae: 0.1930 - val_loss: 0.0357 - val_mae: 0.1889
    Epoch 18/100
    1/1 [==============================] - 0s 62ms/step - loss: 0.0343 - mae: 0.1659 - val_loss: 0.0226 - val_mae: 0.1501
    Epoch 19/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0259 - mae: 0.1417 - val_loss: 0.0137 - val_mae: 0.1170
    Epoch 20/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0201 - mae: 0.1205 - val_loss: 0.0080 - val_mae: 0.0895
    Epoch 21/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0161 - mae: 0.1034 - val_loss: 0.0045 - val_mae: 0.0671
    Epoch 22/100
    1/1 [==============================] - 0s 61ms/step - loss: 0.0135 - mae: 0.0913 - val_loss: 0.0025 - val_mae: 0.0502
    Epoch 23/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0118 - mae: 0.0830 - val_loss: 0.0014 - val_mae: 0.0375
    Epoch 24/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0108 - mae: 0.0784 - val_loss: 8.6765e-04 - val_mae: 0.0293
    Epoch 25/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0101 - mae: 0.0757 - val_loss: 5.7286e-04 - val_mae: 0.0238
    Epoch 26/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0097 - mae: 0.0744 - val_loss: 5.0436e-04 - val_mae: 0.0223
    Epoch 27/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0094 - mae: 0.0730 - val_loss: 3.8748e-04 - val_mae: 0.0195
    Epoch 28/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0091 - mae: 0.0729 - val_loss: 6.6207e-04 - val_mae: 0.0255
    Epoch 29/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0089 - mae: 0.0709 - val_loss: 2.4908e-04 - val_mae: 0.0154
    Epoch 30/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0088 - mae: 0.0739 - val_loss: 0.0021 - val_mae: 0.0459
    Epoch 31/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0089 - mae: 0.0675 - val_loss: 1.7102e-05 - val_mae: 0.0032
    Epoch 32/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0094 - mae: 0.0833 - val_loss: 0.0040 - val_mae: 0.0632
    Epoch 33/100
    1/1 [==============================] - 0s 54ms/step - loss: 0.0092 - mae: 0.0669 - val_loss: 2.3654e-04 - val_mae: 0.0147
    Epoch 34/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0085 - mae: 0.0761 - val_loss: 0.0028 - val_mae: 0.0523
    Epoch 35/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0081 - mae: 0.0664 - val_loss: 0.0010 - val_mae: 0.0317
    Epoch 36/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0078 - mae: 0.0704 - val_loss: 0.0027 - val_mae: 0.0514
    Epoch 37/100
    1/1 [==============================] - 0s 66ms/step - loss: 0.0076 - mae: 0.0665 - val_loss: 0.0017 - val_mae: 0.0414
    Epoch 38/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0075 - mae: 0.0683 - val_loss: 0.0031 - val_mae: 0.0554
    Epoch 39/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0073 - mae: 0.0659 - val_loss: 0.0024 - val_mae: 0.0482
    Epoch 40/100
    1/1 [==============================] - 0s 61ms/step - loss: 0.0072 - mae: 0.0673 - val_loss: 0.0038 - val_mae: 0.0617
    Epoch 41/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0070 - mae: 0.0650 - val_loss: 0.0029 - val_mae: 0.0530
    Epoch 42/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0069 - mae: 0.0666 - val_loss: 0.0050 - val_mae: 0.0704
    Epoch 43/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0068 - mae: 0.0639 - val_loss: 0.0031 - val_mae: 0.0555
    Epoch 44/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0068 - mae: 0.0668 - val_loss: 0.0068 - val_mae: 0.0823
    Epoch 45/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0068 - mae: 0.0628 - val_loss: 0.0030 - val_mae: 0.0541
    Epoch 46/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.0068 - mae: 0.0682 - val_loss: 0.0091 - val_mae: 0.0952
    Epoch 47/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0069 - mae: 0.0617 - val_loss: 0.0034 - val_mae: 0.0578
    Epoch 48/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0068 - mae: 0.0690 - val_loss: 0.0096 - val_mae: 0.0977
    Epoch 49/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0066 - mae: 0.0611 - val_loss: 0.0048 - val_mae: 0.0690
    Epoch 50/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0064 - mae: 0.0664 - val_loss: 0.0094 - val_mae: 0.0965
    Epoch 51/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0062 - mae: 0.0607 - val_loss: 0.0063 - val_mae: 0.0788
    Epoch 52/100
    1/1 [==============================] - 0s 63ms/step - loss: 0.0060 - mae: 0.0645 - val_loss: 0.0098 - val_mae: 0.0984
    Epoch 53/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0059 - mae: 0.0602 - val_loss: 0.0074 - val_mae: 0.0857
    Epoch 54/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0059 - mae: 0.0634 - val_loss: 0.0106 - val_mae: 0.1023
    Epoch 55/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0058 - mae: 0.0595 - val_loss: 0.0084 - val_mae: 0.0913
    Epoch 56/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0058 - mae: 0.0632 - val_loss: 0.0115 - val_mae: 0.1068
    Epoch 57/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0058 - mae: 0.0586 - val_loss: 0.0095 - val_mae: 0.0970
    Epoch 58/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0058 - mae: 0.0643 - val_loss: 0.0121 - val_mae: 0.1096
    Epoch 59/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0059 - mae: 0.0578 - val_loss: 0.0114 - val_mae: 0.1061
    Epoch 60/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0061 - mae: 0.0668 - val_loss: 0.0113 - val_mae: 0.1057
    Epoch 61/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0063 - mae: 0.0573 - val_loss: 0.0151 - val_mae: 0.1223
    Epoch 62/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0066 - mae: 0.0693 - val_loss: 0.0093 - val_mae: 0.0961
    Epoch 63/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0067 - mae: 0.0581 - val_loss: 0.0189 - val_mae: 0.1369
    Epoch 64/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0069 - mae: 0.0702 - val_loss: 0.0087 - val_mae: 0.0925
    Epoch 65/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0066 - mae: 0.0584 - val_loss: 0.0201 - val_mae: 0.1414
    Epoch 66/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0066 - mae: 0.0686 - val_loss: 0.0095 - val_mae: 0.0971
    Epoch 67/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0062 - mae: 0.0575 - val_loss: 0.0200 - val_mae: 0.1408
    Epoch 68/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0061 - mae: 0.0663 - val_loss: 0.0107 - val_mae: 0.1028
    Epoch 69/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.0059 - mae: 0.0568 - val_loss: 0.0201 - val_mae: 0.1413
    Epoch 70/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0059 - mae: 0.0650 - val_loss: 0.0116 - val_mae: 0.1069
    Epoch 71/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0058 - mae: 0.0563 - val_loss: 0.0208 - val_mae: 0.1438
    Epoch 72/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0059 - mae: 0.0649 - val_loss: 0.0120 - val_mae: 0.1090
    Epoch 73/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0058 - mae: 0.0562 - val_loss: 0.0220 - val_mae: 0.1478
    Epoch 74/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0060 - mae: 0.0657 - val_loss: 0.0122 - val_mae: 0.1099
    Epoch 75/100
    1/1 [==============================] - 0s 63ms/step - loss: 0.0060 - mae: 0.0562 - val_loss: 0.0233 - val_mae: 0.1520
    Epoch 76/100
    1/1 [==============================] - 0s 65ms/step - loss: 0.0062 - mae: 0.0667 - val_loss: 0.0124 - val_mae: 0.1109
    Epoch 77/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0061 - mae: 0.0574 - val_loss: 0.0241 - val_mae: 0.1547
    Epoch 78/100
    1/1 [==============================] - 0s 62ms/step - loss: 0.0063 - mae: 0.0667 - val_loss: 0.0129 - val_mae: 0.1132
    Epoch 79/100
    1/1 [==============================] - 0s 54ms/step - loss: 0.0060 - mae: 0.0568 - val_loss: 0.0243 - val_mae: 0.1555
    Epoch 80/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0061 - mae: 0.0657 - val_loss: 0.0136 - val_mae: 0.1161
    Epoch 81/100
    1/1 [==============================] - 0s 54ms/step - loss: 0.0058 - mae: 0.0555 - val_loss: 0.0245 - val_mae: 0.1559
    Epoch 82/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0058 - mae: 0.0645 - val_loss: 0.0143 - val_mae: 0.1190
    Epoch 83/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0057 - mae: 0.0550 - val_loss: 0.0247 - val_mae: 0.1567
    Epoch 84/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0057 - mae: 0.0638 - val_loss: 0.0148 - val_mae: 0.1210
    Epoch 85/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.0056 - mae: 0.0548 - val_loss: 0.0254 - val_mae: 0.1587
    Epoch 86/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0057 - mae: 0.0638 - val_loss: 0.0150 - val_mae: 0.1219
    Epoch 87/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0057 - mae: 0.0550 - val_loss: 0.0261 - val_mae: 0.1611
    Epoch 88/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.0058 - mae: 0.0643 - val_loss: 0.0152 - val_mae: 0.1227
    Epoch 89/100
    1/1 [==============================] - 0s 54ms/step - loss: 0.0058 - mae: 0.0560 - val_loss: 0.0266 - val_mae: 0.1627
    Epoch 90/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.0059 - mae: 0.0645 - val_loss: 0.0155 - val_mae: 0.1238
    Epoch 91/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0058 - mae: 0.0563 - val_loss: 0.0270 - val_mae: 0.1637
    Epoch 92/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0059 - mae: 0.0642 - val_loss: 0.0158 - val_mae: 0.1251
    Epoch 93/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0057 - mae: 0.0559 - val_loss: 0.0272 - val_mae: 0.1643
    Epoch 94/100
    1/1 [==============================] - 0s 56ms/step - loss: 0.0058 - mae: 0.0637 - val_loss: 0.0162 - val_mae: 0.1265
    Epoch 95/100
    1/1 [==============================] - 0s 59ms/step - loss: 0.0056 - mae: 0.0553 - val_loss: 0.0273 - val_mae: 0.1647
    Epoch 96/100
    1/1 [==============================] - 0s 55ms/step - loss: 0.0057 - mae: 0.0632 - val_loss: 0.0165 - val_mae: 0.1278
    Epoch 97/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0055 - mae: 0.0548 - val_loss: 0.0275 - val_mae: 0.1654
    Epoch 98/100
    1/1 [==============================] - 0s 60ms/step - loss: 0.0056 - mae: 0.0629 - val_loss: 0.0167 - val_mae: 0.1285
    Epoch 99/100
    1/1 [==============================] - 0s 58ms/step - loss: 0.0055 - mae: 0.0549 - val_loss: 0.0279 - val_mae: 0.1664
    Epoch 100/100
    1/1 [==============================] - 0s 57ms/step - loss: 0.0056 - mae: 0.0630 - val_loss: 0.0168 - val_mae: 0.1291

</details>

#### 5-2. 순환 드롭아웃(Recurrent Dropout) 

- 드롭아웃의 한 종류로 순환 층에서 과대적합을 방지하기 위해 사용

- 조건 : 모든 중간층은 전체 출력 시퀀스를 반환해야 함 [return_sequence = True]

```python
# 드롭아웃
inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.GRU(32, recurrent_dropout = 0.25, return_sequences = True)(inputs)
x = keras.layers.GRU(32)(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['mae'])
history = model.fit(train_dataset,
                   epochs = 100,
                   validation_data = val_dataset)
```

<details>
<summary>Result</summary>


```
Epoch 1/100
1/1 [==============================] - 3s 3s/step - loss: 0.8150 - mae: 0.8397 - val_loss: 1.3966 - val_mae: 1.1814
Epoch 2/100
1/1 [==============================] - 0s 52ms/step - loss: 0.6650 - mae: 0.7599 - val_loss: 1.1692 - val_mae: 1.0810
Epoch 3/100
1/1 [==============================] - 0s 51ms/step - loss: 0.5647 - mae: 0.6948 - val_loss: 0.9948 - val_mae: 0.9971
Epoch 4/100
1/1 [==============================] - 0s 54ms/step - loss: 0.4836 - mae: 0.6394 - val_loss: 0.8562 - val_mae: 0.9250
Epoch 5/100
1/1 [==============================] - 0s 51ms/step - loss: 0.4271 - mae: 0.5980 - val_loss: 0.7352 - val_mae: 0.8571
Epoch 6/100
1/1 [==============================] - 0s 53ms/step - loss: 0.4027 - mae: 0.5769 - val_loss: 0.6344 - val_mae: 0.7963
Epoch 7/100
1/1 [==============================] - 0s 52ms/step - loss: 0.3139 - mae: 0.5051 - val_loss: 0.5406 - val_mae: 0.7350
Epoch 8/100
1/1 [==============================] - 0s 49ms/step - loss: 0.2970 - mae: 0.4913 - val_loss: 0.4557 - val_mae: 0.6748
Epoch 9/100
1/1 [==============================] - 0s 51ms/step - loss: 0.2604 - mae: 0.4470 - val_loss: 0.3834 - val_mae: 0.6190
Epoch 10/100
1/1 [==============================] - 0s 51ms/step - loss: 0.2313 - mae: 0.4209 - val_loss: 0.3163 - val_mae: 0.5622
Epoch 11/100
1/1 [==============================] - 0s 52ms/step - loss: 0.1569 - mae: 0.3363 - val_loss: 0.2602 - val_mae: 0.5099
Epoch 12/100
1/1 [==============================] - 0s 53ms/step - loss: 0.1560 - mae: 0.3392 - val_loss: 0.2201 - val_mae: 0.4690
Epoch 13/100
1/1 [==============================] - 0s 51ms/step - loss: 0.1374 - mae: 0.3070 - val_loss: 0.1759 - val_mae: 0.4192
Epoch 14/100
1/1 [==============================] - 0s 53ms/step - loss: 0.1461 - mae: 0.2990 - val_loss: 0.1464 - val_mae: 0.3824
Epoch 15/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0945 - mae: 0.2439 - val_loss: 0.1287 - val_mae: 0.3586
Epoch 16/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0831 - mae: 0.2428 - val_loss: 0.0915 - val_mae: 0.3024
Epoch 17/100
1/1 [==============================] - 0s 55ms/step - loss: 0.0975 - mae: 0.2358 - val_loss: 0.0770 - val_mae: 0.2773
Epoch 18/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0759 - mae: 0.2211 - val_loss: 0.0534 - val_mae: 0.2309
Epoch 19/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0596 - mae: 0.1973 - val_loss: 0.0511 - val_mae: 0.2258
Epoch 20/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0610 - mae: 0.1969 - val_loss: 0.0418 - val_mae: 0.2042
Epoch 21/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0787 - mae: 0.2301 - val_loss: 0.0370 - val_mae: 0.1922
Epoch 22/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0627 - mae: 0.2015 - val_loss: 0.0232 - val_mae: 0.1522
Epoch 23/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0728 - mae: 0.2195 - val_loss: 0.0189 - val_mae: 0.1373
Epoch 24/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0810 - mae: 0.2235 - val_loss: 0.0137 - val_mae: 0.1170
Epoch 25/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0486 - mae: 0.1800 - val_loss: 0.0189 - val_mae: 0.1373
Epoch 26/100
1/1 [==============================] - 0s 55ms/step - loss: 0.0566 - mae: 0.1899 - val_loss: 0.0110 - val_mae: 0.1048
Epoch 27/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0693 - mae: 0.2148 - val_loss: 0.0158 - val_mae: 0.1253
Epoch 28/100
1/1 [==============================] - 0s 55ms/step - loss: 0.0564 - mae: 0.1857 - val_loss: 0.0148 - val_mae: 0.1215
Epoch 29/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0728 - mae: 0.2203 - val_loss: 0.0153 - val_mae: 0.1236
Epoch 30/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0540 - mae: 0.1906 - val_loss: 0.0145 - val_mae: 0.1202
Epoch 31/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0591 - mae: 0.1860 - val_loss: 0.0139 - val_mae: 0.1178
Epoch 32/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0468 - mae: 0.1698 - val_loss: 0.0162 - val_mae: 0.1270
Epoch 33/100
1/1 [==============================] - 0s 55ms/step - loss: 0.0570 - mae: 0.1969 - val_loss: 0.0149 - val_mae: 0.1216
Epoch 34/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0571 - mae: 0.1757 - val_loss: 0.0125 - val_mae: 0.1117
Epoch 35/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0670 - mae: 0.1941 - val_loss: 0.0196 - val_mae: 0.1398
Epoch 36/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0502 - mae: 0.1778 - val_loss: 0.0272 - val_mae: 0.1647
Epoch 37/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0346 - mae: 0.1514 - val_loss: 0.0247 - val_mae: 0.1568
Epoch 38/100
1/1 [==============================] - 0s 56ms/step - loss: 0.0352 - mae: 0.1578 - val_loss: 0.0173 - val_mae: 0.1311
Epoch 39/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0657 - mae: 0.2058 - val_loss: 0.0160 - val_mae: 0.1261
Epoch 40/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0516 - mae: 0.1843 - val_loss: 0.0068 - val_mae: 0.0820
Epoch 41/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0648 - mae: 0.2076 - val_loss: 0.0136 - val_mae: 0.1161
Epoch 42/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0730 - mae: 0.2107 - val_loss: 0.0045 - val_mae: 0.0667
Epoch 43/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0381 - mae: 0.1605 - val_loss: 0.0076 - val_mae: 0.0870
Epoch 44/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0533 - mae: 0.1925 - val_loss: 0.0014 - val_mae: 0.0366
Epoch 45/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0663 - mae: 0.2068 - val_loss: 0.0201 - val_mae: 0.1416
Epoch 46/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0387 - mae: 0.1631 - val_loss: 0.0207 - val_mae: 0.1436
Epoch 47/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0535 - mae: 0.1866 - val_loss: 0.0228 - val_mae: 0.1507
Epoch 48/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0580 - mae: 0.1886 - val_loss: 0.0244 - val_mae: 0.1561
Epoch 49/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0594 - mae: 0.1816 - val_loss: 0.0227 - val_mae: 0.1502
Epoch 50/100
1/1 [==============================] - 0s 75ms/step - loss: 0.0895 - mae: 0.2385 - val_loss: 0.0147 - val_mae: 0.1208
Epoch 51/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0393 - mae: 0.1603 - val_loss: 0.0171 - val_mae: 0.1303
Epoch 52/100
1/1 [==============================] - 0s 54ms/step - loss: 0.0452 - mae: 0.1702 - val_loss: 0.0301 - val_mae: 0.1732
Epoch 53/100
1/1 [==============================] - 0s 57ms/step - loss: 0.0477 - mae: 0.1605 - val_loss: 0.0248 - val_mae: 0.1571
Epoch 54/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0389 - mae: 0.1620 - val_loss: 0.0161 - val_mae: 0.1266
Epoch 55/100
1/1 [==============================] - 0s 48ms/step - loss: 0.0457 - mae: 0.1744 - val_loss: 0.0254 - val_mae: 0.1590
Epoch 56/100
1/1 [==============================] - 0s 47ms/step - loss: 0.0417 - mae: 0.1627 - val_loss: 0.0282 - val_mae: 0.1675
Epoch 57/100
1/1 [==============================] - 0s 48ms/step - loss: 0.0365 - mae: 0.1474 - val_loss: 0.0374 - val_mae: 0.1931
Epoch 58/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0513 - mae: 0.1792 - val_loss: 0.0228 - val_mae: 0.1506
Epoch 59/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0506 - mae: 0.1779 - val_loss: 0.0082 - val_mae: 0.0898
Epoch 60/100
1/1 [==============================] - 0s 55ms/step - loss: 0.0488 - mae: 0.1776 - val_loss: 0.0101 - val_mae: 0.1001
Epoch 61/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0538 - mae: 0.1750 - val_loss: 0.0088 - val_mae: 0.0935
Epoch 62/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0538 - mae: 0.1812 - val_loss: 0.0277 - val_mae: 0.1660
Epoch 63/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0433 - mae: 0.1678 - val_loss: 0.0178 - val_mae: 0.1330
Epoch 64/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0535 - mae: 0.1807 - val_loss: 0.0097 - val_mae: 0.0982
Epoch 65/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0389 - mae: 0.1580 - val_loss: 0.0271 - val_mae: 0.1643
Epoch 66/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0354 - mae: 0.1439 - val_loss: 0.0111 - val_mae: 0.1049
Epoch 67/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0388 - mae: 0.1520 - val_loss: 0.0233 - val_mae: 0.1524
Epoch 68/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0437 - mae: 0.1643 - val_loss: 0.0156 - val_mae: 0.1246
Epoch 69/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0282 - mae: 0.1369 - val_loss: 0.0191 - val_mae: 0.1379
Epoch 70/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0327 - mae: 0.1415 - val_loss: 0.0130 - val_mae: 0.1136
Epoch 71/100
1/1 [==============================] - 0s 57ms/step - loss: 0.0464 - mae: 0.1742 - val_loss: 0.0149 - val_mae: 0.1214
Epoch 72/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0499 - mae: 0.1775 - val_loss: 0.0283 - val_mae: 0.1679
Epoch 73/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0385 - mae: 0.1587 - val_loss: 0.0204 - val_mae: 0.1425
Epoch 74/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0415 - mae: 0.1556 - val_loss: 0.0171 - val_mae: 0.1303
Epoch 75/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0337 - mae: 0.1471 - val_loss: 0.0454 - val_mae: 0.2128
Epoch 76/100
1/1 [==============================] - 0s 59ms/step - loss: 0.0336 - mae: 0.1358 - val_loss: 0.0197 - val_mae: 0.1398
Epoch 77/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0359 - mae: 0.1519 - val_loss: 0.0302 - val_mae: 0.1733
Epoch 78/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0404 - mae: 0.1599 - val_loss: 0.0248 - val_mae: 0.1570
Epoch 79/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0322 - mae: 0.1371 - val_loss: 0.0292 - val_mae: 0.1704
Epoch 80/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0504 - mae: 0.1658 - val_loss: 0.0092 - val_mae: 0.0955
Epoch 81/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0468 - mae: 0.1642 - val_loss: 0.0289 - val_mae: 0.1695
Epoch 82/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0289 - mae: 0.1307 - val_loss: 0.0114 - val_mae: 0.1061
Epoch 83/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0388 - mae: 0.1559 - val_loss: 0.0074 - val_mae: 0.0852
Epoch 84/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0399 - mae: 0.1526 - val_loss: 0.0179 - val_mae: 0.1333
Epoch 85/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0332 - mae: 0.1449 - val_loss: 0.0178 - val_mae: 0.1330
Epoch 86/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0519 - mae: 0.1781 - val_loss: 0.0309 - val_mae: 0.1754
Epoch 87/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0415 - mae: 0.1542 - val_loss: 0.0129 - val_mae: 0.1130
Epoch 88/100
1/1 [==============================] - 0s 56ms/step - loss: 0.0257 - mae: 0.1268 - val_loss: 0.0088 - val_mae: 0.0932
Epoch 89/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0319 - mae: 0.1553 - val_loss: 0.0219 - val_mae: 0.1476
Epoch 90/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0334 - mae: 0.1478 - val_loss: 0.0345 - val_mae: 0.1853
Epoch 91/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0369 - mae: 0.1500 - val_loss: 0.0351 - val_mae: 0.1870
Epoch 92/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0431 - mae: 0.1490 - val_loss: 0.0159 - val_mae: 0.1255
Epoch 93/100
1/1 [==============================] - 0s 47ms/step - loss: 0.0336 - mae: 0.1509 - val_loss: 0.0196 - val_mae: 0.1397
Epoch 94/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0251 - mae: 0.1213 - val_loss: 0.0300 - val_mae: 0.1727
Epoch 95/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0465 - mae: 0.1632 - val_loss: 0.0186 - val_mae: 0.1357
Epoch 96/100
1/1 [==============================] - 0s 59ms/step - loss: 0.0378 - mae: 0.1504 - val_loss: 0.0187 - val_mae: 0.1363
Epoch 97/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0360 - mae: 0.1537 - val_loss: 0.0539 - val_mae: 0.2318
Epoch 98/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0236 - mae: 0.1150 - val_loss: 0.0484 - val_mae: 0.2196
Epoch 99/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0480 - mae: 0.1707 - val_loss: 0.0213 - val_mae: 0.1455
Epoch 100/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0455 - mae: 0.1572 - val_loss: 0.0253 - val_mae: 0.1586
```

</details>

#### 5-3. Early Stopping

- 너무 많은 Epoch는 Overfitting을 일으킨다
- 너무 적은 Epoch는 Underfitting을 일으킨다
- Epoch를 많이 돌린 후, 특정 시점에서 멈춤
  - monitor : 모델 학습을 멈추는 기준
  - patience : 최소 반복수
  - verbose : 자세한 정보 표시 모드 ( 0 or 1 )


```python
# Early Stopping
from keras.callbacks import EarlyStopping

inputs = keras.Input(shape = (sequence_length, df.shape[-1]))
x = keras.layers.GRU(32, recurrent_dropout = 0.5, return_sequences = True)(inputs)
x = keras.layers.GRU(32, recurrent_dropout = 0.5)(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1) 

model.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['mae'])
history = model.fit(train_dataset,
                   epochs = 100,
                   validation_data = val_dataset,
                   callbacks = [early_stopping])
```

<details>
  <summary>Reseult</summary>


```
Epoch 1/100
1/1 [==============================] - 3s 3s/step - loss: 1.0261 - mae: 0.9484 - val_loss: 1.8189 - val_mae: 1.3482
Epoch 2/100
1/1 [==============================] - 0s 52ms/step - loss: 0.8877 - mae: 0.8771 - val_loss: 1.5520 - val_mae: 1.2454
Epoch 3/100
1/1 [==============================] - 0s 52ms/step - loss: 0.7711 - mae: 0.8121 - val_loss: 1.3620 - val_mae: 1.1667
Epoch 4/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6672 - mae: 0.7533 - val_loss: 1.2037 - val_mae: 1.0968
Epoch 5/100
1/1 [==============================] - 0s 51ms/step - loss: 0.6018 - mae: 0.7196 - val_loss: 1.0586 - val_mae: 1.0286
Epoch 6/100
1/1 [==============================] - 0s 52ms/step - loss: 0.5777 - mae: 0.6974 - val_loss: 0.9405 - val_mae: 0.9695
Epoch 7/100
1/1 [==============================] - 0s 50ms/step - loss: 0.5211 - mae: 0.6551 - val_loss: 0.8297 - val_mae: 0.9106
Epoch 8/100
1/1 [==============================] - 0s 52ms/step - loss: 0.4403 - mae: 0.6017 - val_loss: 0.7203 - val_mae: 0.8484
Epoch 9/100
1/1 [==============================] - 0s 50ms/step - loss: 0.3782 - mae: 0.5543 - val_loss: 0.6334 - val_mae: 0.7956
Epoch 10/100
1/1 [==============================] - 0s 53ms/step - loss: 0.3294 - mae: 0.5174 - val_loss: 0.5415 - val_mae: 0.7356
Epoch 11/100
1/1 [==============================] - 0s 52ms/step - loss: 0.3311 - mae: 0.5178 - val_loss: 0.4515 - val_mae: 0.6717
Epoch 12/100
1/1 [==============================] - 0s 51ms/step - loss: 0.2755 - mae: 0.4621 - val_loss: 0.3825 - val_mae: 0.6183
Epoch 13/100
1/1 [==============================] - 0s 63ms/step - loss: 0.2130 - mae: 0.4045 - val_loss: 0.3174 - val_mae: 0.5632
Epoch 14/100
1/1 [==============================] - 0s 49ms/step - loss: 0.2006 - mae: 0.3836 - val_loss: 0.2651 - val_mae: 0.5146
Epoch 15/100
1/1 [==============================] - 0s 48ms/step - loss: 0.2091 - mae: 0.3923 - val_loss: 0.2179 - val_mae: 0.4666
Epoch 16/100
1/1 [==============================] - 0s 49ms/step - loss: 0.1808 - mae: 0.3680 - val_loss: 0.1786 - val_mae: 0.4225
Epoch 17/100
1/1 [==============================] - 0s 50ms/step - loss: 0.1714 - mae: 0.3355 - val_loss: 0.1346 - val_mae: 0.3668
Epoch 18/100
1/1 [==============================] - 0s 50ms/step - loss: 0.1091 - mae: 0.2740 - val_loss: 0.1094 - val_mae: 0.3307
Epoch 19/100
1/1 [==============================] - 0s 49ms/step - loss: 0.1257 - mae: 0.2673 - val_loss: 0.1033 - val_mae: 0.3213
Epoch 20/100
1/1 [==============================] - 0s 50ms/step - loss: 0.1144 - mae: 0.2675 - val_loss: 0.0912 - val_mae: 0.3019
Epoch 21/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0869 - mae: 0.2391 - val_loss: 0.0732 - val_mae: 0.2704
Epoch 22/100
1/1 [==============================] - 0s 49ms/step - loss: 0.1092 - mae: 0.2658 - val_loss: 0.0565 - val_mae: 0.2375
Epoch 23/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0815 - mae: 0.2240 - val_loss: 0.0449 - val_mae: 0.2118
Epoch 24/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0826 - mae: 0.2332 - val_loss: 0.0300 - val_mae: 0.1731
Epoch 25/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0616 - mae: 0.2148 - val_loss: 0.0300 - val_mae: 0.1731
Epoch 26/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0821 - mae: 0.2312 - val_loss: 0.0244 - val_mae: 0.1560
Epoch 27/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0953 - mae: 0.2304 - val_loss: 0.0242 - val_mae: 0.1553
Epoch 28/100
1/1 [==============================] - 0s 53ms/step - loss: 0.0696 - mae: 0.2129 - val_loss: 0.0127 - val_mae: 0.1125
Epoch 29/100
1/1 [==============================] - 0s 56ms/step - loss: 0.0680 - mae: 0.2084 - val_loss: 0.0120 - val_mae: 0.1094
Epoch 30/100
1/1 [==============================] - 0s 51ms/step - loss: 0.0816 - mae: 0.2153 - val_loss: 0.0145 - val_mae: 0.1202
Epoch 31/100
1/1 [==============================] - 0s 56ms/step - loss: 0.0741 - mae: 0.2040 - val_loss: 0.0145 - val_mae: 0.1202
Epoch 32/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0637 - mae: 0.2069 - val_loss: 0.0107 - val_mae: 0.1032
Epoch 33/100
1/1 [==============================] - 0s 48ms/step - loss: 0.0528 - mae: 0.1919 - val_loss: 0.0135 - val_mae: 0.1159
Epoch 34/100
1/1 [==============================] - 0s 48ms/step - loss: 0.0702 - mae: 0.2228 - val_loss: 0.0060 - val_mae: 0.0775
Epoch 35/100
1/1 [==============================] - 0s 48ms/step - loss: 0.0753 - mae: 0.2190 - val_loss: 0.0100 - val_mae: 0.0999
Epoch 36/100
1/1 [==============================] - 0s 47ms/step - loss: 0.0611 - mae: 0.1983 - val_loss: 0.0137 - val_mae: 0.1171
Epoch 37/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0709 - mae: 0.2062 - val_loss: 0.0141 - val_mae: 0.1186
Epoch 38/100
1/1 [==============================] - 0s 50ms/step - loss: 0.0761 - mae: 0.2318 - val_loss: 0.0228 - val_mae: 0.1510
Epoch 39/100
1/1 [==============================] - 0s 49ms/step - loss: 0.0666 - mae: 0.2134 - val_loss: 0.0184 - val_mae: 0.1355
Epoch 39: early stopping
```

</details>

### 6. 데이터 예측

```python
# Dataset에서 samples과 target 분리
for train_samples, train_targets in train_dataset:
    pass
for val_samples, val_targets in val_dataset:
    pass
for test_samples, test_targets in test_dataset:
    pass
  
# 모델을 적용한 출력값
trainPredict = model.predict(train_samples)
valPredict = model.predict(val_samples)
testPredict = model.predict(test_samples)

# * std + mean을 통하여 실제 값으로 변환
# Training Data 예측값
Train_Predict = list()
for i in trainPredict:
    Train_Predict.append(i * std + mean)

# Training Data 타깃 값
Train_Targets = list()
for i in train_targets:
    Train_Targets.append(i * std + mean)
    
# Validation Data 예측값
Val_Predict = list()
for i in valPredict:
    Val_Predict.append(i * std + mean)

# Validation Data 타깃 
Val_Targets = list()
for i in val_targets:
    Val_Targets.append(i * std + mean)
    
# Test Data 예측값
Test_Predict = list()
for i in testPredict:
    Test_Predict.append(i * std + mean)
    
# Test Data 타깃값
Test_Targets = list()
for i in test_targets:
    Test_Targets.append(i * std + mean)
```

```
2/2 [==============================] - 0s 2ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 14ms/step
```



### 7. 모델의 정확도

```python
trainScore = math.sqrt(mean_squared_error(Train_Targets, Train_Predict))
print(f"Train Score : {trainScore:.2f} RMSE")

valScore = math.sqrt(mean_squared_error(Val_Targets, Val_Predict))
print(f"Train Score : {valScore:.2f} RMSE")

testScore = math.sqrt(mean_squared_error(Test_Targets, Test_Predict))
print(f"Train Score : {testScore:.2f} RMSE")
```

```
Train Score : 255.92 RMSE
Train Score : 137.97 RMSE
Train Score : 180.13 RMSE
```

### 8. 시각화(실제 값과 예측값 그래프)

```python
# train
trainPredictPlot = np.empty_like(dataframe)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[sequence_length:len(Train_Predict)+sequence_length, :] = Train_Predict
# val
valPredictPlot = np.empty_like(dataframe)
valPredictPlot[:, :] = np.nan
valPredictPlot[len(Train_Predict)+sequence_length :len(Train_Predict)+len(Val_Predict)+sequence_length, :] = Val_Predict
# test
testPredictPlot = np.empty_like(dataframe)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(Train_Predict)+len(Val_Predict)+ sequence_length : len(Train_Predict)+len(Val_Predict)+ sequence_length + len(Test_Predict), :] = Test_Predict

plt.plot(dataframe)               # 파란색 : 실제 값
plt.plot(trainPredictPlot)      # 주황색 : Training Data 예측값
plt.plot(valPredictPlot)        # 초록색 : Validation Data 예측값
plt.plot(testPredictPlot)       # 빨간색 : Test Data 예측값
plt.show()
```
