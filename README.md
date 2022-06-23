# Coding_trick

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
