# Coding_trick

# 경로지정

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
