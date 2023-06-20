### 고려대학교 23-1 비정형데이터 Term project 10조 Github

---

본 github은 10조 팔도비빔면(박진용, 박태남, 정용태, 백두산)의 VATMAN논문에 대한 코드를 정리하여 업로드한 페이지입니다. 

<br/><br/>

### 모델 아키텍쳐 👀

---

![image](https://github.com/Kingdu97/VATMAN_30h/assets/122776983/2e2fd1cb-2de4-4dcb-b479-6c0ed9b9e081)

<br/><br/>

### 논문 요약😃

---

이 논문에서는 **멀티모달 생성요약 태스크에서** 트랜스포머 기반 Trimodal Hierarchical Multi-head Attention 기법을 사용하여 Vision, Audio, Text의 핵심 정보를 한번에 융합처리하는 모델(VATMAN : Video-Audio-Text Multimodal Abstractive SummarizatioN)을 제안한다.

기존의 멀티모달 생성요약 분야에선 순환신경망보다 트랜스포머를 기반으로 한 융합이 우수한 성능을 보여주었지만, 트랜스포머 기반 대규모 생성형 사전학습 언어모델(Generative Pretrained Language Models)에 트리모달 계층적 어탠션이 적용된 사례는 찾기 어려웠다.

따라서 본 논문에서는 **비교적 간단하지만 효과적인 어탠션 레이어를 추가사용하여 텍스트 생성능력을 해치지 않으면서 시각적, 청각적 정보를 차례로 주입시키는 방법으로 성능을 향상시킨다**.

<br/><br/>

### dataset✔

---

> 우리는 Sanabria 등이 공개한 How2 데이터셋[1]을 사용합니다. 이 데이터셋은 다양한 모달리티를 이해하기 위한 대규모 데이터로, 비디오, 오디오, 비디오 자막(Transcripts) 및 요약문(Summary)으로 구성되어 있습니다.

- 비디오 : .npy 형태로 **개인적**으로 제공
- 오디오 : .npy 형태로 **개인적**으로 제공
- 비디오 자막(Transcripts): 화자의 음성을 텍스트로 변환한 데이터로, 비디오의 전반적인 내용을 담고 있습니다.
- 요약문(Summary): 전체 비디오에 대한 추상적인 개요를 제공합니다.

30시간의 데이터셋은 300시간의 비디오, 오디오 및 텍스트 데이터셋에서 랜덤 샘플링하여 추출한 데이터입니다. 이 데이터셋에는 학습 데이터 1,279개, 검증 데이터 52개, 테스트 데이터 12개로 총 1,343개의 데이터가 포함되어 있습니다. 데이터 용량이 크기 때문에 데이터를 받고자 하시면 이메일 주소로 문의해 주시면 제공해 드릴 수 있습니다.

[97dosan@naver.com](mailto:97dosan@naver.com)

[1] Ramon Sanabria, Ozan Caglayan, Shruti Palaskar, Desmond Elliott, Loïc Barrault, Lucia Specia, and Florian Metze. 2018. "How2: a large-scale dataset for multimodal language understanding." arXiv preprint arXiv:1811.00347. ↩

<br/><br/>

### Code🐱‍🏍

---

- 유니모달(**Text** to text)

`python .src/run_30_text_only_t5.py`

- 다이모달(**Text+Video** to text)

 `python .src/run_30_multi_modal_bart.py`

- 트리모달(**Video+Audio+Text** to text)

`python .src/run_30_tri_modal_bart.py`

> backbone 모델을 bart로 바꾸고싶으면 config 내 model name 만 수정

<br/><br/>

<br/><br/>
### Results🎁

---

![image](https://github.com/Kingdu97/VATMAN_30h/assets/122776983/12482ae2-33a6-48a8-91d3-aaf18d182e64)

 
