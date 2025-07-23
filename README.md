# 250728snu
# ❄️ 환경 데이터 분석 및 의사 결정 도우미 (Streamlit + GPT-4o)

이 프로젝트는 Streamlit과 OpenAI의 GPT-4o API를 활용하여 사용자가 업로드한 환경 데이터를 분석하고, 데이터 기반의 의사 결정을 돕는 웹 애플리케이션입니다. 특히 기후 변화와 관련된 해빙(Sea Ice) 데이터와 같은 시계열 데이터 분석에 유용합니다.

## ✨ 주요 기능

- **CSV 파일 업로드:** 사용자가 자신의 환경 데이터를 CSV 형식으로 업로드할 수 있습니다.
- **데이터 미리보기:** 업로드된 데이터의 처음 몇 행과 기본 통계 정보를 확인합니다.
- **자연어 질문:** 데이터에 대해 궁금한 점을 자연어로 질문하면 GPT-4o가 분석합니다.
- **GPT-4o 기반 분석:** GPT-4o가 데이터의 구조와 내용을 이해하고, 질문에 대한 심층적인 분석 결과, 시각화 제안, 그리고 의사 결정 및 정책적 인사이트를 제공합니다.
- **자동 시각화:** 분석 결과를 보완하고 이해도를 높이기 위해 주요 수치 데이터의 시계열 그래프와 계절성 패턴을 자동으로 시각화합니다.

## 🚀 시작하는 방법

### 1. 전제 조건

- Python 3.8 이상
- OpenAI API 키 발급 (https://platform.openai.com/account/api-keys)

### 2. 프로젝트 클론 및 설정

```bash
git clone [https://github.com/YOUR_USERNAME/your-environmental-data-app.git](https://github.com/YOUR_USERNAME/your-environmental-data-app.git)
cd your-environmental-data-app
