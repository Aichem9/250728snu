import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io

# OpenAI API 키 설정 (Streamlit Secrets를 사용하여 안전하게 관리하는 것이 가장 좋습니다)
# 예시: .streamlit/secrets.toml 파일에 OPENAI_API_KEY = "your_openai_api_key_here" 로 저장
# 또는 환경 변수로 설정: export OPENAI_API_KEY="your_openai_api_key_here"
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API 키가 설정되지 않았습니다. '.streamlit/secrets.toml' 파일에 'OPENAI_API_KEY'를 설정하거나 환경 변수로 설정해주세요.")
    st.stop() # API 키 없으면 앱 실행 중단

st.set_page_config(layout="wide")

st.title("❄️ 환경 데이터 분석 및 의사 결정 도우미")
st.markdown("---")

# 1. 데이터셋 업로드
st.sidebar.header("데이터셋 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드해주세요.", type=["csv"])

# 샘플 데이터셋 경로 (Kaggle에서 다운로드 후 프로젝트 폴더에 저장 가정)
# 사용자가 업로드하지 않았을 때의 대체 옵션
sample_data_path = {
    "샘플: 북극 해빙 면적 데이터": "data/N_seaice_extent_daily_v3.0.csv",
    # "샘플: 전지구 온도 이상치 데이터": "data/global_temp_anomaly.csv", # 추가 샘플
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
        # 실제 파일 경로가 존재해야 합니다. (Kaggle에서 다운로드하여 data 폴더에 넣었다고 가정)
        df = pd.read_csv(sample_data_path[selected_sample])
        st.sidebar.success(f"'{selected_sample}' 데이터셋 로드 성공!")
    except FileNotFoundError:
        st.sidebar.warning(f"'{sample_data_path[selected_sample]}' 파일을 찾을 수 없습니다. 해당 파일을 'data/' 폴더에 넣어주세요.")
    except Exception as e:
        st.sidebar.error(f"샘플 데이터셋을 읽는 중 오류가 발생했습니다: {e}")

st.markdown("---")

if df is not None:
    st.subheader("📊 업로드된 데이터 미리보기")
    st.write(df.head())
    st.write(f"데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")

    # 날짜 컬럼 자동 감지 및 생성 시도
    date_column_candidates = ['Date', 'date', 'DATE', 'Time', 'time', 'TIME']
    year_month_day_candidates = [('Year', 'Month', 'Day'), ('year', 'month', 'day'), ('YEAR', 'MONTH', 'DAY')]
    
    date_col_found = False
    for col in date_column_candidates:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                st.info(f"'{col}' 컬럼을 날짜 형식으로 인식했습니다.")
                date_col_found = True
                break
            except Exception:
                pass # 날짜 변환 실패 시 다음 후보 시도
    
    if not date_col_found:
        for y, m, d in year_month_day_candidates:
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
                    pass # 날짜 변환 실패 시 다음 후보 시도
    
    if not date_col_found:
        st.warning("날짜/시간 정보를 포함하는 컬럼을 자동으로 감지하거나 생성할 수 없었습니다. 시계열 분석 및 시각화에 제한이 있을 수 있습니다.")

    st.subheader("❓ 데이터에 대한 질문 입력")
    user_question = st.text_area("업로드된 데이터에 대해 궁금한 점을 질문해주세요:",
                                 placeholder="예: '이 데이터셋에서 해빙 면적의 연간 평균 변화 추세는 어떻게 되나요?', '가장 큰 변화를 보인 기간은 언제인가요?', '이러한 환경 변화가 생태계에 미칠 잠재적 영향은 무엇인가요?'")

    if st.button("답변 생성"):
        if user_question:
            with st.spinner("GPT-4o가 데이터를 분석 중입니다..."):
                try:
                    # GPT-4o에 전달할 데이터 요약 및 사용자 질문
                    data_head = df.head().to_markdown(index=False) # 데이터의 첫 5행 요약
                    data_description = df.describe().to_markdown() # 통계 요약
                    
                    # df.info()의 출력을 캡처하여 문자열로 변환
                    buffer = io.StringIO()
                    df.info(buf=buffer, verbose=True, show_counts=True)
                    column_info_str = buffer.getvalue()

                    time_range = ""
                    if date_col_found and 'Date' in df.columns:
                        time_range = f"데이터 기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
                    
                    prompt = f"""
                    당신은 기후 변화 및 환경 데이터 분석 전문가입니다. 주어진 환경 데이터에 대한 사용자의 질문에 답하고,
                    필요하다면 시각화를 위한 제안과 환경 변화에 대응하기 위한 의사 결정 또는 정책적 인사이트를 제공해주세요.
                    제공된 데이터는 CSV 파일에서 로드되었으며, 그 구조와 요약은 다음과 같습니다:

                    ---
                    데이터 요약 (첫 5행):
                    {data_head}

                    ---
                    데이터 통계 요약:
                    {data_description}

                    ---
                    컬럼 정보 (데이터 타입 및 Non-null 개수):
                    {column_info_str}

                    ---
                    {time_range}

                    ---
                    사용자의 질문: "{user_question}"

                    답변은 다음 형식으로 구성해주세요:
                    1. **환경 데이터 분석 결과:** 질문에 대한 직접적인 데이터 기반 답변 (예: 특정 기간 동안의 해빙 면적 감소율, 주요 추세).
                    2. **시각화 제안 (선택 사항):** 답변을 뒷받침하거나 더 깊이 이해하기 위한 시각화 아이디어 (예: '연간 해빙 면적 변화를 보여주는 꺾은선 그래프와 추세선', '월별 해빙 면적의 계절성 패턴').
                    3. **의사 결정 및 정책 인사이트:** 분석 결과를 바탕으로 기후 변화 대응, 환경 보호, 연구 방향 설정 등에 대해 사용자가 고려할 수 있는 구체적인 제안이나 통찰.
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o", # 또는 "gpt-3.5-turbo" 등 사용 가능한 모델
                        messages=[
                            {"role": "system", "content": "You are a helpful climate and environmental data analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7 # 창의성 조절
                    )
                    
                    gpt_response = response.choices[0].message.content
                    st.subheader("✨ GPT-4o의 분석 결과 및 의사 결정 지원")
                    st.markdown(gpt_response)

                    # 시각화 제안이 있다면 파싱하여 실제로 시각화하는 로직 추가
                    st.markdown("---")
                    st.subheader("📈 주요 시각화 (GPT 제안 기반)")
                    
                    # 시계열 데이터 시각화 시도 (날짜 컬럼 및 수치 컬럼 존재 시)
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if date_col_found and 'Date' in df.columns and len(numeric_cols) > 0:
                        st.write("시간 경과에 따른 주요 수치 데이터 변화 추이:")
                        
                        # 첫 번째 수치 컬럼을 기본으로 시각화
                        plot_col = numeric_cols[0] 
                        
                        # 해빙 데이터의 경우 'Extent' 또는 'Area'를 우선적으로 시각화
                        if 'Extent' in numeric_cols:
                            plot_col = 'Extent'
                        elif 'Area' in numeric_cols:
                            plot_col = 'Area'
                        elif 'CO2' in numeric_cols:
                            plot_col = 'CO2'
                        elif 'Anomaly' in numeric_cols:
                            plot_col = 'Anomaly'
                        elif 'Temperature' in numeric_cols:
                            plot_col = 'Temperature'

                        if plot_col:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            sns.lineplot(x='Date', y=plot_col, data=df, ax=ax, label=f'{plot_col} 값')
                            
                            # 추세선 추가 (선형 회귀)
                            # 날짜를 숫자로 변환하여 회귀 분석에 사용
                            sns.regplot(x=df['Date'].apply(lambda date: date.toordinal()), y=df[plot_col], ax=ax, scatter=False, color='red', line_kws={'linestyle': '--'}, label='추세선')

                            ax.set_title(f'시간 경과에 따른 {plot_col} 변화')
                            ax.set_xlabel('날짜')
                            ax.set_ylabel(plot_col)
                            plt.xticks(rotation=45)
                            plt.grid(True, linestyle='--', alpha=0.7)
                            plt.legend()
                            st.pyplot(fig)
                            st.caption(f"이 그래프는 {plot_col}이 시간 경과에 따라 어떻게 변화했는지 보여줍니다. 빨간 점선은 전체적인 추세를 나타냅니다.")

                            # 월별 평균 시각화 (계절성 파악)
                            if 'Month' in df.columns:
                                st.write(f"월별 평균 {plot_col} (계절성 패턴):")
                                df['Month_Name'] = df['Date'].dt.strftime('%b') # 'Jan', 'Feb'
                                # 월 순서 유지를 위해 reindex 사용
                                monthly_avg_per_month = df.groupby('Month_Name')[plot_col].mean().reindex(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                                
                                fig2, ax2 = plt.subplots(figsize=(10, 5))
                                sns.barplot(x=monthly_avg_per_month.index, y=monthly_avg_per_month.values, ax=ax2, palette='viridis')
                                ax2.set_title(f'월별 평균 {plot_col} (계절성)')
                                ax2.set_xlabel('월')
                                ax2.set_ylabel(f'평균 {plot_col}')
                                st.pyplot(fig2)
                                st.caption(f"이 그래프는 연간 {plot_col}의 계절적 변동을 보여줍니다.")
                        else:
                            st.info("시각화할 수 있는 적절한 수치 컬럼을 찾을 수 없습니다.")
                    else:
                        st.info("날짜/시간 컬럼과 수치 컬럼이 모두 존재해야 시계열 시각화를 생성할 수 있습니다.")

                except Exception as e:
                    st.error(f"GPT API 호출 중 오류가 발생했습니다: {e}")
        else:
            st.warning("질문을 입력해주세요!")

else:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터셋을 선택해주세요.")

st.markdown("---")
st.sidebar.markdown("이 앱은 업로드된 환경 데이터 분석을 통해 기후 변화에 대한 의사 결정을 돕기 위해 GPT-4o를 활용합니다.")
