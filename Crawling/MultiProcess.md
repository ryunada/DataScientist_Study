코어의 개수 == 사용 가능한 process를 사용할 수 있다.
ex) 8개

# NaverTrend_Crawling.ipynb
```python
from multiprocessing import Process, Queue

# from .py파일 import .py파일안에 있는 함수명들
from Naver_Trend_Crawling.py import Crawling1, Crawling2, Crawling3, Crawling4, Crawling5, Crawling6, Crawling7, Crawling8


if __name__ == "__main__" :
    th1 = Process(target=Crawling1)
    th2 = Process(target=Crawling2)
    th3 = Process(target=Crawling3)
    th4 = Process(target=Crawling4)
    th5 = Process(target=Crawling5)
    th6 = Process(target=Crawling6)
    th7 = Process(target=Crawling7)
    th8 = Process(target=Crawling8)
    
    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    th6.start()
    th7.start()
    th8.start()
    th1.join()
    th2.join()
    th3.join()
    th4.join()
    th5.join()
    th6.join()
    th7.join()
    th8.join()
```

# Naver_Trend_Crawling.py
```python
from selenium import webdriver as wd
import time
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

options = Options()
options.use_chromium = True
options.add_experimental_option("prefs", {
    "download.default_directory": r"C:\Users\rst30\Desktop\1st_Project\Data\keyword",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})
driver = wd.Chrome(r'C:\Users\rst30\Desktop\1st_Project\ChromeDriver(win)\chromedriver.exe', options=options)
driver.get('https://keywordsound.com/service/keyword-analysis')

df = pd.read_csv(r'C:\Users\rst30\Desktop\1st_Project\Data\RCP_RE_NM.csv', encoding='cp949')
# print(df.head())

df_dict = df.to_dict()
df_dict = list(df_dict['CKG_NM'].values())

def Crawling1():
    for i in range(1823, 6323): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
        
def Crawling2():
    for i in range(6323, 10823): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()

def Crawling3():
    for i in range(10823, 15323): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()

def Crawling4():
    for i in range(15323, 19823): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
        
def Crawling5():
    for i in range(19823,24323): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
        
def Crawling6():
    for i in range(24323, 28823): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()

def Crawling7():
    for i in range(28823,33323): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
        
def Crawling8():
    for i in range(33323,len(df_dict)): # 1788, 1789
        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(10)  # 10초대기

        # 날짜 선택
        driver.find_element(By.XPATH,'//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element(By.XPATH,'//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element(By.XPATH,'//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()

        driver.find_element(By.XPATH,'//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()```
