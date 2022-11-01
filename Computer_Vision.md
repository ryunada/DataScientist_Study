# 컴퓨터 비전 문제 영역
### Image Classification
=> 인풋 이미지가 어떤 라벨(Label)에 대응되는지 인풋 이미지에 대한 분류 (Classification)를 수행하는 문제 영역.  
[EX), 이 이미지는 강아지이다, 이 이미지는 고양이이다.]  
<img width="417" alt="스크린샷 2022-11-01 오후 1 57 32" src="https://user-images.githubusercontent.com/87309905/199162165-e14d922e-926d-46e3-a65e-dad2b7cf395c.png">


### Face Detection  
=> 얼굴이있는영역의위치정보를Boundingbox로찾는문제영역  
<img width="427" alt="스크린샷 2022-11-01 오후 1 58 06" src="https://user-images.githubusercontent.com/87309905/199162234-7964365b-8cc3-4c11-b728-b53995487037.png">


### Face Alignment
=> 얼굴의특징영역(예를들어,얼굴의눈,코,입)을포인트(Landmark)로찾는문제영역
<img width="916" alt="스크린샷 2022-11-01 오후 1 58 16" src="https://user-images.githubusercontent.com/87309905/199162252-2b458358-314a-4c12-806c-0b14718945f0.png">

### Steering Angle Prediction
=> 자동차의적합한SteeringAngle조작값을예측하는문제영역
<img width="659" alt="스크린샷 2022-11-01 오후 1 58 51" src="https://user-images.githubusercontent.com/87309905/199162320-0f4489a2-ab4b-4205-9e05-abc5521c016f.png">


### Super Resolution
=> 저해상도이미지를인풋으로받으면이를고해상도이미지로변경해주는문제영역
<img width="520" alt="스크린샷 2022-11-01 오후 1 59 06" src="https://user-images.githubusercontent.com/87309905/199162350-f0b8cccb-0e8a-4996-9c13-4db372ef5123.png">


### Object Detection
=> 물체가있는영역의위치정보를BoundingBox로찾고BoundingBox내에존재하는 물의 라벨(Label)을 분류하는 문제 영역
<img width="448" alt="스크린샷 2022-11-01 오후 1 59 39" src="https://user-images.githubusercontent.com/87309905/199162418-413ca488-b906-4f27-8cbd-74ba1ae43934.png">


### Image Captioning
=> 이미지에대한설명문을자동생성하는문제영역
<img width="811" alt="스크린샷 2022-11-01 오후 2 00 25" src="https://user-images.githubusercontent.com/87309905/199162492-ce55dbad-7fdb-4045-9ed5-16aba7c41888.png">


### Neural Style Transfer
=> 콘텐츠이미지에스타일이미지를덧씌운합성이미지를만드는문제영역
<img width="599" alt="스크린샷 2022-11-01 오후 2 00 57" src="https://user-images.githubusercontent.com/87309905/199162537-cf691feb-1519-4113-a5f1-34329bdd7308.png">


### Generative Model
=> 트레이닝데이터의분포를학습하고이를이용해서새로운가짜데이터를생성하는문 제영역
<img width="700" alt="스크린샷 2022-11-01 오후 2 01 35" src="https://user-images.githubusercontent.com/87309905/199162610-0816ca0f-3735-4c86-9712-6893ca2c5953.png">


### Semantic Image Segmentation
=> 이미지의전체픽셀에대한분류를수행하는문제영역
<img width="472" alt="스크린샷 2022-11-01 오후 2 02 07" src="https://user-images.githubusercontent.com/87309905/199162654-2b457bd8-1ce3-4b2f-a187-cf0de4d01587.png">



### Brain Tumor Segmentation
=> SemanticImageSegmentation을이용해서Brain이미지내에종양 (Tumor)이 있는 부분을 자동 분류하는 문제 영역
<img width="662" alt="스크린샷 2022-11-01 오후 2 02 30" src="https://user-images.githubusercontent.com/87309905/199162691-79b2a4b0-d0cd-4ef9-9cf4-3421b94fa5a9.png">


### Face Recognition
=> 얼굴 인식

<img width="393" alt="스크린샷 2022-11-01 오후 2 02 45" src="https://user-images.githubusercontent.com/87309905/199162717-f11fbfaf-4fb8-46d8-aa66-f837f53ef304.png">

### Face Verification
=> 두개의얼굴이미지를인풋으로받아서해당얼굴이미지가동일인물인지아닌지를 판단하는문제영역
<img width="486" alt="스크린샷 2022-11-01 오후 2 04 09" src="https://user-images.githubusercontent.com/87309905/199162843-dee3731b-c061-478f-a388-9f79c7d1ae2c.png">


### Face Hallucination
=> 얼굴이미지에대한SuperResolution을수행하는문제영역
<img width="437" alt="스크린샷 2022-11-01 오후 2 04 36" src="https://user-images.githubusercontent.com/87309905/199162877-a863aaf1-c80e-460c-802a-4707e9f618bb.png">


### Text Detection
=> 이미지내에텍스트가존재하는영역의위치정보를BoundingBox로찾는문제영역
<img width="418" alt="스크린샷 2022-11-01 오후 2 05 10" src="https://user-images.githubusercontent.com/87309905/199162937-2a0b737e-10e8-469c-ba4b-c4c6b09b1f24.png">


### Optical Character Recognition(OCR)
=> TextDetection이수행된BoundingBox내에존재하는글자 가 어떤 글자인지를 인식하는 문제 영역
<img width="431" alt="스크린샷 2022-11-01 오후 2 05 27" src="https://user-images.githubusercontent.com/87309905/199162965-04e0c247-5761-493f-80c1-ec1818d89d74.png">



### License Platte Detection
=> TextDetection과OCR을이용해서차량번호판을인식하는문제영역
<img width="442" alt="스크린샷 2022-11-01 오후 2 05 47" src="https://user-images.githubusercontent.com/87309905/199163000-e270ec17-de8b-43f5-919e-dc9b6d84988b.png">

### Defect Detection
=> 공정 과정   에 불량(Defect)을 검출하는 문제 영역
<img width="447" alt="스크린샷 2022-11-01 오후 2 05 56" src="https://user-images.githubusercontent.com/87309905/199163019-fbb97b62-8db2-4351-9e7f-1884dc62e94d.png">

### Human Pose Estimation
=> 인간의중요신체부위를Keypoint라는점으로추정해서현재포즈를예측하 는문제영역
<img width="705" alt="스크린샷 2022-11-01 오후 2 06 14" src="https://user-images.githubusercontent.com/87309905/199163046-3844e84c-eb46-4ff4-b0db-f5b151a20e2d.png">

[ 이미지 출처 ]
https://arxiv.org/abs/1902.10859
https://github.com/commaai/research
https://arxiv.org/abs/1609.04802
https://arxiv.org/abs/1411.4555
https://arxiv.org/abs/2001.00179
https://arxiv.org/abs/2001.00179
https://arxiv.org/abs/1505.03540
https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/lrs/pubs/koestinger_cvpr_2012.pdf
https://arxiv.org/abs/1806.10726
https://arxiv.org/abs/1903.06593

