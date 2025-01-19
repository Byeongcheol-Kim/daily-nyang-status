# 고양이 분석 POC

고양이에 관련된 특성을 분석하는 POC 페이지입니다.

## 주요 기능

- 고양이 색상 추출
- 고양이 자세/행동 태깅

## 설치 방법

```bash
python3.12 -m venv .venv
source .venv/bin/activate
poetry install
```

## 패키지 업데이트

```bash
poetry export -f requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```
