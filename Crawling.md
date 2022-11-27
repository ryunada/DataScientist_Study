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
from selenium import webdriver
import time
import os
```
## 작업 경로 설정
```python
# 내 작업 경로 확인
print(os.getcwd())

# 작업 경로 설정
os.chdir('경로')

# 변경된 경로 확인
print(os.getcwd())
```
