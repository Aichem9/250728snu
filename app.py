import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io
from matplotlib.dates import date2num

st.set_page_config(layout="wide")

st.title("❄️ 환경 데이터 분석 및 의사 결정 도움이")
st.markdown("---")

# 0. OpenAI API 키 입력 받기
st.sidebar.header("API 키 설정")
openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력해주세요", type="password")

client = None
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        st.sidebar.success("OpenAI API 키가 성공적으로 설정되었습니다!")
    except Exception as e:
        st.sidebar.error(f"API 키 설정 중 오류가 발생했습니다: {e}.")
else:
    st.sidebar.warning("OpenAI API 키를 입력해주세요.")

st.sidebar.markdown("---")

# 1. 데이터셀 업로드
st.sidebar.header("데이터셀 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드해주세요.", type=["csv"])

sample_data_path = {
    "샜항: 북규 해밀 면적 데이터": "data/N_seaice_extent_daily_v3.0.csv",
}

st.sidebar.markdown("---")
st.sidebar.info("파일이 없으면 산항 데이터셀을 선택해 테스트할 수 있습니다.")
selected_sample = st.sidebar.selectbox("또는 산항 데이터셀 선택", [""] + list(sample_data_path.keys()))

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"'{uploaded_file.name}' 파일 업로드 성공!")
    except Exception as e:
        st.sidebar.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
elif selected_sample:
    try:
        df = pd.read_csv(sample_data_path[selected_sample])
        st.sidebar.success(f"'{selected_sample}' 데이터셀 로드 성공!")
    except FileNotFoundError:
        st.sidebar.warning("산항 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.sidebar.error(f"산항 데이터셀을 읽는 중 오류가 발생했습니다: {e}")

st.markdown("---")

if client and df is not None:
    st.subheader("파일 미리보기")
    st.write(df.head())
    st.write(f"데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")

    date_col_found = False
    date_candidates = ['Date', 'date', 'DATE', 'Time', 'time', 'TIME']
    for col in date_candidates:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                st.info(f"'{col}' 열 데이터로 인식했습니다.")
                df['Date'] = df[col]
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
                    st.info(f"'{y}', '{m}' ({d} 선택적) 열 사용")
                    date_col_found = True
                    break
                except Exception:
                    pass

    if not date_col_found:
        st.warning("날짜/시간 정보를 자동으로 감지 또는 생성 할 수 없습니다.")

    st.subheader("질문 입력")
    user_question = st.text_area("질문을 입력해주세요:",
                                 placeholder="예: '어느 가지 수치가 최고인가?', 'Trend에 따라 어떻게 변화했나요?' 등")

    if st.button("답변 생성"):
        if user_question:
            with st.spinner("GPT-4o가 데이터를 분석 중입니다..."):
                try:
                    data_head = df.head().to_markdown(index=False)
                    data_description = df.describe().to_markdown()
                    buffer = io.StringIO()
                    df.info(buf=buffer, verbose=True, show_counts=True)
                    column_info_str = buffer.getvalue()

                    time_range = ""
                    if date_col_found and 'Date' in df.columns:
                        time_range = f"데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"

                    prompt = f"""
                    답변들:
                    1. **환경 데이터 분석 결과:**
                    2. **시각화 제안:**
                    3. **의사 결정 및 정책 인생:**

                    ---
                    데이터 요약:
                    {data_head}

                    ---
                    통계 요약:
                    {data_description}

                    ---
                    열 정보:
                    {column_info_str}

                    ---
                    {time_range}

                    ---
                    사용자의 질문: "{user_question}"
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful climate and environmental data analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )

                    gpt_response = response.choices[0].message.content
                    st.subheader("✨ GPT-4o의 분석 결과")
                    st.markdown(gpt_response)

                    st.markdown("---")
                    st.subheader("주요 시각화")

                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                    if date_col_found and 'Date' in df.columns and len(numeric_cols) > 0:
                        st.write("시간과 역사 수치 변화")

                        plot_col = None
                        for col in ['Extent', 'Area', 'CO2', 'Anomaly', 'Temperature']:
                            if col in numeric_cols:
                                plot_col = col
                                break

                        if plot_col is None:
                            plot_col = numeric_cols[0]

                        if plot_col:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            sns.lineplot(x='Date', y=plot_col, data=df, ax=ax, label=f'{plot_col} 값')

                            df['Date_ordinal'] = df['Date'].map(date2num)
                            sns.regplot(
                                x='Date_ordinal',
                                y=plot_col,
                                data=df,
                                scatter=False,
                                ax=ax,
                                label='추세선',
                                color='orange'
                            )
                            ax.set_xticks(ax.get_xticks())
                            ax.set_xticklabels(
                                [pd.to_datetime(num).strftime('%Y-%m-%d') for num in ax.get_xticks()],
                                rotation=45
                            )
                            ax.legend()
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"GPT-4o 등장 중 오류 발생: {e}")
