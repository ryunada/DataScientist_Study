# Crawling 
I. chrome 버전 확인 
  - chrome 

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
