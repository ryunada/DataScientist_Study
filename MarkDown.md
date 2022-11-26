
MarkDown 문법 정리  
==================
---
> ## 1. 마크다운  
> > ### 1.1 마크다운이란?  
> > ### 1.2 마크다운의 장점  
> > > #### 1.2.1 장점  
> > > #### 1.2.2 단점  

> ## 2. 마크다운 문법
> > ### 2.1 헤더(Header)
> > ### 2.2 BlockQuote
> > ### 2.3 목록
> > > #### I. 순서있는 목록(번호)
> > >  #### II. 순서 없는 목록(글머리 기호 : `*`,`+`,`-` 지원)

> > ### 2.4 코드
> > > #### 2.4.1 들여쓰기



---
## 1. 마크다운


### 1.1 마크다운이란?

---
### 1.2 마크다운의 장-단점
#### 1.2.1 장점
```markdown
1. 간결하가.
2. 별도구없이 작성 가능하다.
3. 다양한 형태로 변환이 가능하다.
4. 텍스트(Text)로 저장되기 때문에 용량이 적어 보관이 용이하다.
5. 텍스트파일이기 때문에 버전관리시스템을 이용하여 변경이력을 관리할 수 있다.
6. 지원하는 프로그램과 플랫폼이 다양하다. 
```
#### 1.2.2 단점
```markdown
1. 표준이 없다.
2. 표준이 없기 때문에 도구에 따라서 변환방식이나 생성물이 다르다.
3. 모든 HTML 마크업을 대신하지 못한다.
```

---
## 2. 마크다운 문법
---
### 2.1 헤더(Header)
- 큰 제목 : 문서 제목
```markdown
Document Title
==============
```
Document Title
==============

- 작은 제목 : 문서 부제목
```markdown
Document SubTitle
-----------------
```
Document SubTitle
-----------------

- 글머리 : 1~6까지만 지원
```markdown
# This is a H1
## This is a H2
### This is a H3
#### This is a H4
##### This is a H5
###### This is a H6
```
# This is a H1
## This is a H2
### This is a H3
#### This is a H4
##### This is a H5
###### This is a H6
####### 7개 이상은 지원하지 않아요~

---
### 2.2. BlockQuote
이메일에서 사용하는 ```>```블럭인용 문자를 사용한다.
```markdown
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.
```

> This is a first blockqute.
>> This is a second blockqute.
>>> This is a third blockqute.


---
### 2.3. 목록
#### I. 순서있는 목록(번호)
순서있는 목록은 숫자와 점을 사용한다.
```markdown
1. 첫번째
2. 두번째
3. 세번째
```

1. 첫번째
2. 두번째
3. 세번째

현재까지는 어떤 번호를 입력해도 순서는 내림차순으로 정의된다.
```markdown
1. 첫번째
3. 세번째
2. 두번째 
```

1. 첫번째
2. 세번째
3. 두번째

#### II. 순서없는 목록(글머리 기호 : `*`,`+`,`-` 지원)
``` markdown
* 첫번째 글머리
    * 두번째 글머리
        * 세번째 글머리
        
+ 첫번째 글머리
    + 두번째 글머리
        + 세번째 글머리
        
- 첫번째 글머리
    - 두번째 글머리
        - 세번째 글머리
```
* 첫번째 글머리
    * 두번째 글머리
        * 세번째 글머리
        
+ 첫번째 글머리
    + 두번째 글머리
        + 세번째 글머리
        
- 첫번째 글머리
    - 두번째 글머리
        - 세번째 글머리

#### 혼합해서 사용 가능하다.
```markdown
* 첫 번째 글머리
    + 두번재 글머리
        - 세번째 글머리
```

* 첫 번째 글머리
    + 두번재 글머리
        - 세번째 글머리

---
### 2.4. 코드
4개의 공백 또는 하나의 탭으로 들여쓰기를 만나면 변환되기 시작하여 들여쓰지 않은 행을 만날때까지 변환이 계속된다.  
#### 2.4.1 들여쓰기
```python
This is a normal paragraph:

    This is a code block.
    
end code block.
```
This is a normal paragraph:

    This is a code block.
    
end code block.

> 한줄 띄어쓰지 않으면 인식이 제대로 안되는 문제 발생

```python
This is a normal paragraph:
    This is a code block.
end code block.
```

---

This is a normal paragraph:
    This is a code block.
end code block.

---
#### 2.4.2. 코드블럭
코드블럭은 다음과 같이 2가지 방식을 사용할 수 있습니다.:
- ```<pre><code>{code}</code></pre>```이용방식

``` python
<pre>
<code>
    print("```<pre<code>{code}</code></pre>```이렇게 코드 작성 가능")
</code>
</pre>
```

```python
for i in range(0,100):
    print('집가고 싶다.')
```

- 코드블럭코드("```")을 이용하는 방법
```python
```
   Write code here
```
```

```
Write code here
```

깃허브에서는 코드블럭코드("```") 시작점에 사용하는 언어를 선언하여 문법강조(Syntax Highlighting)이 가능하다.


```python
print('Welcome to Python')
``` 

---
### 2.5. 수평선 `<hr/>`
```python
* * *
***
*****
- - -
-----------------------------------------------
```
적용 예시
* * *
***
*****
- - -
-----------------------------------------------
### 2.6 링크
- 참조 링크
```
[link keyword][id]
[id]: URL "Optional Title here"
// code
Link: [Google][googlelink]
[googlelink]: https://google.com "Go google"
```

Google Link : [Google][http://google.com]

- 외부 링크
```python
사용문법 : [Title](link)
적용예 : [Google](https://google.com, "Google Link")
```
Link : [Google](https://google.com, "google link")

- 자동연결
```python
일반적인 URL 혹은 이메일주 소인 경우 적절한 형식으로 링크를 형성한다.

* 외부 링크 : <http://example.com/>
* 이메일 링크 : <example@email.com>
```
* 외부 링크 : <http://example.com/>
* 이메일 링크 : <example@email.com>

### 2.7. 강조
```python
* single asterisks *
_single underscores_
**double asterisks**
__double underscores__
~~cancelline~~
```
* single asterisks *
_single underscores_
**double asterisks**
__double underscores__
~~cancelline~~

> ```문장 중간에 사용할 경우에는 **띄어쓰기**를 사용하는 것이 좋다.```
> 문장 중간에 사용할 경우에는 **띄어쓰기**를 사용하는 것이 좋다.

---
### 2.8. 이미지
---
```
![image](https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png)
![image](https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png "Optional title")
```

![image](https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png)
![image](https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png "Optional title")

사이즈 조절 기능이 없기 때문에  ```<img witdh="" height=""></img>```를 이용한다.

예

```
<img src="https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png" width="450px" height="300px", title="px(픽셀) 크기 설정" alt="Black"></img><br/>
<img src="https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png" width="40%" height="30%", title="px(픽셀) 크기 설정" alt="Black"></img><
```
<img src="https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png" width="450px" height="300px" title="px(픽셀) 크기 설정" alt="Black"></img><br/>
<img src="https://user-images.githubusercontent.com/87309905/204093936-90bad287-30e4-4ae5-ab7c-17c6b36377f3.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Black"></img>


















