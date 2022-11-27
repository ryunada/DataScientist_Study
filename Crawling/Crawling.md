# Crawling 
I. Chrome 버전 확인 
  - Chrome 오른쪽 상단 <img width="12" alt="스크린샷 2022-11-27 오후 3 00 56" src="https://user-images.githubusercontent.com/87309905/204121527-ec567ccc-7b99-4102-9e47-fd526f4d9869.png"> 클릭
  - 설정 클릭
  - Chrome 정보 클릭
  - 버전 확인  

II. Chrome 버전에 맞춰 Chrome driver 다운로드
  - chromedriver.exe 경로 확인 (window)
  - chromedriver 경로 확인 (mac)


## 라이브러리 임포트
```python 
import urllib.request
from urllib import parse
from bs4 import BeautifulSoup
import ssl
import numpy as np

import pandas as pd
from selenium import webdriver as wd
import time
```

## naver crawling
```python
driver = wd.Chrome('Chromedriver.exe 경로')
driver.get('크롤링할 페이지 경로')

df = pd.read_csv('크롤링할 키워드 csv 경로')

df_dict = df.to_dict()
df_dict = list(df_dict['크롤링할 키워드 csv 열 이름'].values())

for i in range(len(df_dict)):
  페이지에서 할 것들
  ex)
  driver.find_element_by_id('item_keyword1').send_keys(df_dict[i])
  # 뒤로가기
  driver.back()
```

| element(단일), elements(복수) | 설명 |
|:-------------------------|:--|
| By.ID | 태그의 id값으로 추출 |
| By.NAME | 태그의 name값으로 추출 |
| By.XPATH | 태그의 경로로 추출 |
| By.LINK_TEXT | 링크 텍스트값으로 추출 |
| By.PARTIAL_LINK_TEXT | 링크 텍스트의 자식 텍스트 값을 추출 |
| By.TAG_NAME | 태그 이름으로 추출 |
| By.CLASS_NAME | 태그의 클래스명으로 추출 |
| By.CSS+SELECTOR | css선택자로 추출 |

