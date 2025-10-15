# 치과 X-ray 이미지 전처리 파이프라인

치과 파노라마 X-ray 이미지를 표준화된 형식으로 전처리하는 자동화 파이프라인입니다.

## 전체 워크플로우

### 1단계: 데이터셋 분석
python scripts/1_analyze_dataset.py

### 2단계: 참조 히스토그램 생성 (선택적)
python scripts/2_create_reference.py

### 3단계: 전처리 실행
python scripts/3_preprocess_all.py

### 추론: 새 이미지 전처리
python scripts/preprocess_inference.py --mode single --input test.png --output output.png

## 설정

모든 데이터를 통합 설정으로 표준화: config/unified_preprocessing_config.yaml
- 해상도: 2048x1024
- 종횡비: 2:1
- ROI 자동 검출

자세한 내용은 각 스크립트의 주석을 참고하세요.
