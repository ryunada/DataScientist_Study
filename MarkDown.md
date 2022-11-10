
MarkDown 문법 정리  
==================
---
> ## 1. 마크다운  
> > ### 1.1 마크다운이란?  
> > ### 1.2 마크다운의 장점  
> > > #### 1.2.1 장점  
> > > #### 1.2.2 단점  

---
## 1. 마크다운
### 1.1 마크다운이란?
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

## 2. 마크다운 문법
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

### 2.2. BlockQuote
이메일에서 사용하는 ```>```블럭인용 문자를 사용한다.
```markdown
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.
```
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.

