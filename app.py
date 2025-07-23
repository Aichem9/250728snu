import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io

st.set_page_config(layout="wide")

st.title("❄️ 환경 데이터 분석 및 의사 결정 도우미")
st.markdown("---")

# 0. OpenAI API 키 입력 받기 (사이드바에 배치)
st.sidebar.header("API 키 설정")
openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요", type="password")

client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        st.sidebar.success("OpenAI API 키가 성공적으로 설정되었습니다!")
    except Exception as e:
        st.sidebar.error(f"API 키 설정 중 오류가 발생했습니다: {e}. 유효한 키인지 확인해주세요.")
else:
    st.sidebar.warning("OpenAI API 키를 입력해주세요.")
    
st.sidebar.markdown("---") # API 키 섹션과 다음 섹션 구분

# 1. 데이터셋 업로드 및 샘플 데이터 선택
st.sidebar.header("데이터셋 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드해주세요.", type=["csv"])

# 샘플 데이터셋 경로 (GitHub에 업로드하거나 사용자가 직접 다운로드하도록 안내)
sample_data_path = {
    "샘플: 북극 해빙 면적 데이터": "data/N_seaice_extent_daily_v3.0.csv",
}

st.sidebar.markdown("---")
st.sidebar.info("파일이 없으시면 아래 샘플 데이터셋을 선택하여 테스트할 수 있습니다.")
selected_sample = st.sidebar.selectbox("또는 샘플 데이터셋 선택", [""] + list(sample_data_path.keys()))

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"'{uploaded_file.name}' 파일 업로드 성공!")
    except Exception as e:
        st.sidebar.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
elif selected_sample:
    try:
        # GitHub에 배포 시 'data/' 경로에 파일이 있어야 함
        df = pd.read_csv(sample_data_path[selected_sample])
        st.sidebar.success(f"'{selected_sample}' 데이터셋 로드 성공!")
    except FileNotFoundError:
        st.sidebar.warning(f"샘플 파일을 찾을 수 없습니다. GitHub 저장소의 'data/' 폴더에 해당 파일이 있는지 확인하거나 직접 업로드해주세요.")
    except Exception as e:
        st.sidebar.error(f"샘플 데이터셋을 읽는 중 오류가 발생했습니다: {e}")

st.markdown("---")

# API 키와 데이터 프레임이 모두 준비되었을 때만 주 기능 활성화
if client and df is not None:
    st.subheader("📊 업로드된 데이터 미리보기")
    st.write(df.head())
    st.write(f"데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")

    # 날짜 컬럼 자동 감지 및 생성 시도
    date_col_found = False
    
    date_candidates = ['Date', 'date', 'DATE', 'Time', 'time', 'TIME']
    for col in date_candidates:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                st.info(f"'{col}' 컬럼을 날짜 형식으로 인식했습니다.")
                df['Date'] = df[col] # 'Date'라는 통일된 컬럼명 사용
                date_col_found = True
                break
            except Exception:
                pass 
    
    ymd_candidates = [('Year', 'Month', 'Day'), ('year', 'month', 'day'), ('YEAR', 'MONTH', 'DAY')]
    if not date_col_found:
        for y, m, d in ymd_candidates:
            if y in df.columns and m in df.columns:
                try:
                    if d in df.columns:
                        df['Date'] = pd.to_datetime(df[[y, m, d]])
                    else:
                        df['Date'] = pd.to_datetime(df[y].astype(str) + '-' + df[m].astype(str))
                    df = df.sort_values('Date')
                    st.info(f"'{y}', '{m}' (및 '{d}' 선택적) 컬럼을 사용하여 'Date' 컬럼을 생성했습니다.")
                    date_col_found = True
                    break
                except Exception:
                    pass
    
    if not date_col_found:
        st.warning("날짜/시간 정보를 포함하는 컬럼을 자동으로 감지하거나 생성할 수 없었습니다. 시계열 분석 및 시각화에 제한이 있을 수 있습니다.")

    st.subheader("❓ 데이터에 대한 질문 입력")
    user_question = st.text_area("업로드된 데이터에 대해 궁금한 점을 질문해주세요:",
                                 placeholder="예: '이 데이터셋에서 해빙 면적의 연간 평균 변화 추세는 어떻게 되나요?', '가장 큰 변화를 보인 기간은 언제인가요?', '이러한 환경 변화가 생태계에 미칠 잠재적 영향은 무엇인가요?'")

    if st.button("답변 생성"):
        if user_question:
            with st.spinner("GPT-4o가 데이터를 분석 중입니다..."):
                try:
                    data_head = df.head().to_markdown(index=False)
                    data_description = df.describe().to_markdown()
                    
                    buffer = io.StringIO()
