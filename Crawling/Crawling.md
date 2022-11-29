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

---

# naverTrend Crawling
from selenium import webdriver as wd
import time
import pandas as pd

# 크롬 브라우저 위치 설정
driver = wd.Chrome('C:/Users/rst30/Desktop/1st_Project/ChromeDriver(win)/chromedriver.exe')
# 크롬 브라우저를 통하여 웹사이트 가져오기
driver.get('https://datalab.naver.com/keyword/trendSearch.naver')

# 레시피 명이 적힌 csv 불러오기
df = pd.read_csv('C:/Users/rst30/Desktop/1st_Project/Data/RCPTitle.csv')
# print(df.head())

# 딕셔너리 형태로 가져오기
df_dict = df.to_dict()
df_dict = list(df_dict['RCP_NM'].values())

for i in range(len(df_dict)):
    # 키워드
    driver.find_element_by_id('item_keyword1').send_keys(df_dict[i])
    # 시작 년도 월 일
    driver.find_element_by_id('startYear').click()
    driver.find_element_by_xpath('//*[@id="startYearDiv"]/ul/li[6]/a').click()
    driver.find_element_by_id('startMonth').click()
    driver.find_element_by_xpath('//*[@id="startMonthDiv"]/ul/li[1]/a').click()
    driver.find_element_by_id('startDay').click()
    driver.find_element_by_xpath('//*[@id="startDayDiv"]/ul/li[1]/a').click()

    # 종료 년도 월 일
    driver.find_element_by_id('endYear').click()
    driver.find_element_by_xpath('//*[@id="endYearDiv"]/ul/li[6]/a').click()
    driver.find_element_by_id('endMonth').click()
    driver.find_element_by_xpath('//*[@id="endMonthDiv"]/ul/li[12]/a').click()
    driver.find_element_by_id('endDay').click()
    driver.find_element_by_xpath('//*[@id="endDayDiv"]/ul/li[31]/a').click()
    
    if i == 0 :
        # 여성 / 남성 -> tiem_gender_2
        driver.find_element_by_id('item_gender_1').click()

        # 10대
        driver.find_element_by_id('item_age_11').click()
        

    # 검색
    driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
    time.sleep(2)

    # 다운로드 후 뒤로가기
    driver.find_element_by_xpath('//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
    time.sleep(2)
    driver.back()

    # 검색어 지우기
    driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
    driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
    time.sleep(2)


