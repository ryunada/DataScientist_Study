# K-평균 알고리즘(K-Means Clustering Algorithm)
-> 데이터를 k개의 클러스터(군집)으로 묶는 알고리즘

## K-Menas 작동 방식
I. 군집의 개수(K) 설정  
II. 초기 군집의 중심점 설정  
III. 각각 데이터 포인터를 군집에 할당 (유클리드 거리를 사용하여 중심점과의 거리 계산)  
유클리드 거리 : 두 점 사이의 거리를 계산할때 일상에서 가장 많이 사용하는 방법  
-> ![스크린샷 2022-11-27 오후 3 48 49](https://user-images.githubusercontent.com/87309905/204122985-c564fa50-9082-42ee-8110-bb3b125412ec.png)  
IV. 군집의 중심점 업데이트  
V. 앞의 두 방법(III, IV)을 반복  
VI. 더이상의 이동이 없으면 종료  

![스크린샷 2022-11-27 오후 3 58 09](https://user-images.githubusercontent.com/87309905/204123290-1b4850b8-f4ed-4f0c-a146-2bde12ab3419.png)
---
## < 실습 >
### I. 전처리 : 데이터 불러오기 -> 차원 축소(이미지를 2차원에서 1차원으로 축소하여 연산)  
### II. K-Means : K-Means적용 -> 결과 확인 -> 중심점 출력 -> 군집된 결과 확인  
### III. 최적의 군집 개수(K) 찾기 (K-Means의 단점 중 하나는 K값을 지정해 주어야 함 이를 해결하기 위한 것)
---
---
## 06-2 k-평균

### KMeans 클래스
#### I. 무작위로 K개의 클러스터 중심을 정함
#### II. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정합니다.
#### III. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경
#### IV. 클러스터 중심에 변화가 없을 때까지 II번으로 돌아가 반복합니다.


```python
# 데이터 다운로드/
!wget https://bit.ly/fruits_300 -O fruits_300.npy
```
    fruits_300.npy      100%[===================>]   2.86M  --.-KB/s    in 0.07s   
    
    2022-11-27 07:24:02 (40.8 MB/s) - ‘fruits_300.npy’ saved [3000128/3000128]


```python
# k모델을 훈련하기 위해 (샘플 개수, 너비, 높이)크기의 3차원배열을 (샘플 개수, 너비*높이)크기를 가진 2차원 배열로 변경
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```


```python
# 클러스터 개수(n_clusters) 3으로 지정
# 군집의 개수를 3으로 지정
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d)
```

    KMeans(n_clusters=3, random_state=42)


```python
# 군집(cluster)을 3으로 지정했기 때문에 0,1,2 세가지가 나온다.
print(km.labels_)
```

    [2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 0 2 2 2 2 2 2 2 2 0 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1]



```python
# 0, 1, 2로 나눠진 개수 확인
print(np.unique(km.labels_, return_counts = True))
```

    (array([0, 1, 2], dtype=int32), array([111,  98,  91]))



```python
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio =1):
  n = len(arr)  # n : 샘플의 개수  
  rows = int(np.ceil(n/10)) # 한 줄에 10개씩 이미지를 그림. 샘플 개수를 10으로 나누어 전체 행 개수를 계산
  cols = n if rows < 2 else 10 # 행이 1개이면 열의 개수는 샘플 개수. 그렇지 않으면 10개
  fig, axs = plt.subplots(rows, cols,
                          figsize = (cols*ratio, rows*ratio), squeeze = False)
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n:
        axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
      axs[i, j].axis('off')
  plt.show()
```


```python
draw_fruits(fruits[km.labels_==0])
```

![output_8_0](https://user-images.githubusercontent.com/87309905/204124053-ea00cd95-2720-46b7-a1cc-ade378a08b0f.png)

  



```python
draw_fruits(fruits[km.labels_==1])
```


![output_9_0](https://user-images.githubusercontent.com/87309905/204124054-6d0e07fd-a773-4b84-b038-2d6e2470b80b.png)
    


```python
draw_fruits(fruits[km.labels_ == 2])
```


![output_10_0](https://user-images.githubusercontent.com/87309905/204124060-17802bc9-6bb1-4473-8454-b7da1292c634.png)
 



```python
km.cluster_centers_.shape
```




    (3, 10000)




```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio = 3)
```


![output_12_0](https://user-images.githubusercontent.com/87309905/204124068-834effd0-af17-4b38-b872-89e87cfff9ad.png)
 



```python
print(km.transform(fruits_2d[100:101]))
```

    [[3393.8136117  8837.37750892 5267.70439881]]



```python
print(km.predict(fruits_2d[100:101]))
```

    [0]



```python
draw_fruits(fruits[100:101])
```


    

![output_15_0](https://user-images.githubusercontent.com/87309905/204124075-fc2cf955-9c2b-43e2-a781-8dac2d982b5e.png)



```python
print(km.n_iter_)
```

    4



```python
inertia = []
for k in range(2,7):
  km = KMeans(n_clusters = k, random_state = 42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2,7),inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```

![output_17_0](https://user-images.githubusercontent.com/87309905/204124080-f8ccacb4-15b9-48cf-883d-339a65374149.png)














