# Anomaly Detection - PatchCore

- **작업 기간**
2021.12~2021.12 (1개월)

- **인력 구성(기여도)**
AI modeling 1명 (100%), 총 1명

- **프로젝트 개요**
간 데이터를 이용해 암을 찾아내는 anomaly detection project (독성병리학).
기존 Hover-Net을 이용한 Cell-Based 기반 모델을 대체할 모델 연구, 현재 Unsupervised anomaly detection 분야에서 우수한 성능을 보인 PatchCore를 활용하여 접근.

- **평가 방식**
학습 시 정상만을 학습하기 때문에 train(정상) / test(정상,비정상) 형식으로 데이터를 분리한다. 그 후 ROCAUC로 평가를 진행한다.

- **제한 사항**
    - 지도 학습 불가.
    - 클래스 불균형 (class imbalance).
    - 모든 label 제공 불가.
    - recall 1.
    - 빠른 inference 속도.
    - 데이터 공개 불가.

---

# 데이터 설명

- 총 데이터 개수 100개, 정상 90개, 비정상 10개.
- 데이터 형식 : Whole Slide Image (WSI), .mrxs 파일 ( + metadata, .dat 파일 )
- 데이터 크기 : 평균 (77000, 185000, 4), bitmap 기준 대략 56GB

- Label 설명
    - 데이터 이미지에서 비정상 영역에 xml 파일로 boundary 및 병명이 기제 되어 있다.
    - 한 비정상 데이터당 label 개수 : 2~5개, 20~30개 등으로 다양하다.
    
    [Anomaly feature (데이터 상세)](https://www.notion.so/Anomaly-feature-0b625be87f2048228be6184ed3acf6c5)
    

---

# 결과

전처리는 [Cell Based Model](https://www.notion.so/Anomaly-Detection-Cell-Based-Model-dc4f87510468429b8f0f607be7eb64dd)과 동일하게 진행했습니다.

실험은 MHIST Open Dataset으로 진행했습니다.

[MHIST: A Minimalist Histopathology Image Analysis Dataset](https://bmirds.github.io/MHIST/)

### Result

![Untitled](Anomaly%20De%207ab1a/Untitled.png)

[Mid level feature vs High level feature](https://www.notion.so/8615507407fc47e9b147ec2bb84de992)

---

# 문제점

- 고객 사 요구 사항 중 ‘빠른 inference 속도’과 ‘supervised 불가’는 달성 했지만 ‘recall 1’에 실패했다.
- 비록 정확도와 소요 시간 측면에서 Original PatchCore를 앞서기도 하지만 그럼에도 부족한 정확도이다.

[문제점을 개선한 프로젝트](https://www.notion.so/Anomaly-Detection-FPC-03a4a34a4fb8426faa25e561ac133863)

---

# 논문

![한국인터넷정보학회 2020 춘계](Anomaly%20De%207ab1a/Untitled%201.png)

한국인터넷정보학회 2020 춘계

[내용을 Develop하여 해외 저널 도전 중.](https://www.notion.so/Anomaly-Detection-PatchCore-Develop-40b18238bce24810b56676ca50cda762)

---

📁Github

[GitHub - essential2189/PatchCore](https://github.com/essential2189/PatchCore)

📄Reference

[Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)