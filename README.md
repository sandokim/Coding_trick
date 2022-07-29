# Coding_trick

```terminal
I train and test in the training process with new dataset successfully. but when i only test with new dataset, the following error occurs.
'NewDataset is not in the dataset registry'

해결책 __init__.py 생성!

I met with similar problems here. For those who might be interested, remember to add your .py file to __init__.py located in datasets folder.
```

[mmsegmentation analysis](https://better-tomorrow.tistory.com/entry/MMSegmentation-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-Inference)

### [mmcv, Registry, Configuration, builder, 모듈 관리](https://better-tomorrow.tistory.com/entry/MMCV-%EC%9D%98-Registry)

```terminal
MMdetection을 정리하다가 Registry 항목이 많아 정리해본다.
MMCV는 유사한 기능을 공유하는 서로 다른 모듈(backbone, head, necks..)을 관리하기 위해 레지스트리(Registry) 구현
OpenMMLab의 대부분의 프로젝트에서 Registry를 사용하여
MMDetection, MMDetection3D, MMClassification, MMEditing 등과 같은
데이터 세트 및 모델의 모듈을 관리.
```

```terminal
[Registry]
-> 클래스를 문자열에 maps하는 mapping으로 볼 수 있다. Registry로 문자열을 통해 클래스 찾고 인스턴스화
ex) Registry를 사용하여 config를 통해 hook, runner, models 및 datasets를 생성하는 대부분의 OpenMMLab 프로젝트의 config 시스템

Registry에서 코드베이스의 모듈 관리하려면
1. 빌드 방법 만듦
2. Registry 만듦
3. Registry이용해서 모듈 관리
```

```terminal
[build_func]
Registry의 build_func argument는 class instance를 instance화 하는 방법을 사용자가 정의하기 위한 것
기본 : build_from_cfg
```


# 경로지정

[상위경로 잘 지정하는 법](https://m.blog.naver.com/wideeyed/221839634437)
```bash
따라서 python -m 모듈옵션과
패키지정보가 포함된 형태로 aa.cc.human2로 실행해야 올바른 결과를 얻을 수 있습니다.
만약 이를 지키지 않으면 아래 오류등이 발생합니다.
ImportError: attempted relative import with no known parent package
ValueError: attempted relative import beyond top-level package
```
```python
# tiger.py
class Tiger:
    def __str__(self):
        return "tiger!"
```
```python
# lion.py
class Lion:
    def __str__(self):
        return "lion!"
```
```python
# eagle.py
class Eagle:
    def __str__(self):
        return "eagle!"
```

```bash
/---aa
    |   tiger.py
    |
    +---bb
    |   |   lion.py
    |
    +---cc
    |   |   eagle.py
    |   |   human2.py // 신규 추가
```
```python
// human2.py
from ..tiger import Tiger
from ..bb.lion import Lion
from .eagle import Eagle

if __name__ == "__main__":
    print(Tiger())
    print(Lion())
    print(Eagle())
```

```terminal
$ cd /
$ python -m aa.cc.human2

[결과]
tiger!
lion!
eagle!
```


* 동일 경로 파일
실행파일(main.py)과 동일한 경로에 있는 python 파일들은 현재 경로를 의미하는 .를 사용하여 import할 수 있습니다.

```python
# main.py
from . import my_module
```

```bash
$ tree
.
├─ my_module.py
└─ main.py
```

* 하위 경로 파일
하위 경로의 파일은 from 하위 폴더 처럼 폴더를 지정해주어 import할 수 있습니다.
```python
from subdir import my module
```
```bash
$ tree
.
├─ subdir
│  └─ my_module.py
└─ main.py
```

* 상위 경로 파일
상위 폴더를 참조할 때는 from에 상위 경로를 입력해서 import할 수 없고, 절대경로 path에 상위 경로에 대한 path를 추가해줘야 합니다. 그러면 추가된 상위폴더 경로에서 상대적으로 파일들을 참조할 수 있습니다.
* 실행파일 경로의 상위 경로를 구하는 코드는 os.path.dirname(os.path.abspath(os.path.dirname(__file__))) 입니다. 이 경로를 sys.path.append로 절대경로에 추가할 수 있습니다.
```python
# main.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from . import library
```
```bash
$ tree
.
├── main
│   └── main.py
└── library.py
```


#### nnU-Net npz 파일, nii.gz 파일은 (251, 421, 456)과 같이 나온다. 원래는 (456, 421, 251)이다. 

* original shape: (h, w, d)
* converted nii.gz shape: (d, w, h)

```python
data_output = np.transpose(data_output, (2,1,0)) # nii.gz (d,w,h) --> (h,w,d)
```

#### import matplotlib.pyplot as plt는 np.float16 타입을 지원하지 않는다 --> np.float32로 바꾼다.

```python
data_output = data_output.astype(np.float32) # np.float16 --> np.float32
```

## Config.py로 arguments 관리하기

config.py 파일 만들기

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/config file.png" width="60%">

import config 하기

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/import config.png" width="60%">

config.~~ 써먹기

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/config file usage.png" width="60%">

train.py에서 def main(): 부분에서 사용하면 좋다. (train.py 첫줄에 import config 되어있다.)

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/main config usage.png" width="80%">

#### dictionary로 model configuration 구성하기!

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/model configuration dictionary create.png" width="60%">

#### dictionary로 model의 train, val, test configuration 구성하기 (dataset_type, data_root, image normalization config, train data augmentation, samples per gpu 등) 

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/model pipeline dictionary config.png" width="60%">

위의 img_norm_config 같은 경우 dictionary에 ** 연산자를 사용함으로써 dictionary를 합쳐 새로운 하나의 dictionary를 만든다.

** 로 dictionary를 합치는 예시
```python
>>> dic1 = {"A": 1, "B": 2}
>>> dic2 = {"B": 3, "C": 4}
>>> { **dic1, **dic2 }
{'A': 1, 'B': 3, 'C': 4}
```

[Visual Studio Code / 자동 줄바꿈 설정하는 방법](https://www.codingfactory.net/12959)

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/path_trick.png" width="60%">

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/zip_trick.png" width="60%">

## Python

### 산술연산자

* *는 곱하기
* / 는 나누기
* % 는 나머지
* // 는 몫
* ** 는 거듭제곱

### python Extended Slices :: 
 
* arr[::] --> 처음부터 끝까지 1칸 간격 인덱싱
* arr[1:2:3] --> index1부터 index2까지 3칸 간격으로 인덱싱
* arr[::-1] --> 처음부터 끝까지 역순으로 1칸 간격 인덱싱
* arr[::-2] --> 처음부터 끝까지 역순으로 2칸 간격 인덱싱

[[Tip] Python Array[::] 사용법](https://blog.wonkyunglee.io/3)

input이 3d일때, 2d일때 if문으로 나누는 방법

* len(input.shape)==5면 b,c,d,w,h 3D 
* len(input.shape)==4면 b,c,w,d 2D

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/If 3d or 2d.PNG" width="100%">

### torch.no_grad()일때는 copy하여서 gpu의 weights&biases&데이터들을 cpu로 옮겨야한다.

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/torch no grad copy.PNG" width="100%">

### image (c, w, h) --np.transpose--> numpy (w, h, c)

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/check_sample.PNG" width="60%">

## Pytorch Modeling

#### Avoiding forward method by subclassing nn.Sequential

<img src="https://github.com/sandokim/Coding_trick/blob/main/images/feedforward.JPG" width="60%">

## Python

[파이썬 Super 명령 알아보기](https://harry24k.github.io/super/)
