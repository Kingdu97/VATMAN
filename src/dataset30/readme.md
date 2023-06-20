
### 30시간 데이터셋
우리는 Sanabria 등이 공개한 How2 데이터셋[1]을 사용합니다. 이 데이터셋은 비디오, 오디오, 비디오 자막(Transcripts) 및 요약문(Summary)으로 구성되어 있습니다. 비디오 자막은 화자의 음성을 텍스트로 변환한 데이터로, 비디오의 전반적인 내용을 담고 있습니다. 반면에 요약문은 전체 비디오에 대한 추상적인 개요를 제공합니다.

300시간의 비디오, 오디오 및 텍스트 데이터셋에서 랜덤 샘플링을 통해 vid_id_30을 추출해내었고 이를 통해 30시간의 텍스트/비디오/오디오 데이터를 선택하여 사용하였습니다. 이에 따라 1,279개의 학습 데이터, 52개의 검증 데이터 및 12개의 테스트 데이터를 총 1,343개의 데이터로 실험에 활용합니다. 데이터 용량이 매우 크기 때문에 데이터를 받고자 하시면 이메일로 문의해주시면 제공해 드릴 수 있습니다.

[1] Ramon Sanabria, Ozan Caglayan, Shruti Palaskar, Desmond Elliott, Loïc Barrault, Lucia Specia, and Florian Metze. 2018. How2: a large-scale dataset for multimodal language understanding. arXiv preprint arXiv:1811.00347.
