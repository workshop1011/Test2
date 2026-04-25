import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import streamlit as st
import pandas as pd
import numpy as np

import main_2
import sub_2
import end_2

st.set_page_config(page_title="AI 투자 적격성 평가 프로토타입", layout="wide")

if 'ebm_results' not in st.session_state: st.session_state.ebm_results = {}
if 'llm_results' not in st.session_state: st.session_state.llm_results = {}
if 'trained_reg_model' not in st.session_state: st.session_state.trained_reg_model = None
if 'trained_clf_model' not in st.session_state: st.session_state.trained_clf_model = None

@st.cache_data
def load_and_merge_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir = os.path.join(base_dir, 'CSV')
        if not os.path.exists(os.path.join(csv_dir, 'KOSPI_IS.csv')):
            csv_dir = os.path.join(os.getcwd(), 'CSV')
            if not os.path.exists(os.path.join(csv_dir, 'KOSPI_IS.csv')): return None
            
        bs = pd.read_csv(os.path.join(csv_dir, 'KOSPI_BS.csv')) if os.path.exists(os.path.join(csv_dir, 'KOSPI_BS.csv')) else pd.DataFrame()
        is_df = pd.read_csv(os.path.join(csv_dir, 'KOSPI_IS.csv'))
        cf = pd.read_csv(os.path.join(csv_dir, 'KOSPI_CF.csv'))
        
        merge_keys = ['회사명', '회계년도']
        df = is_df.copy()
        if not bs.empty: df = pd.merge(bs, df, on=merge_keys, how='inner')
        df = pd.merge(df, cf, on=merge_keys, how='inner')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"데이터 병합 실패: {e}")
        return None

with st.sidebar:
    st.header("투자 심의 설정")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...").strip()
    macro_scenario = st.selectbox("거시경제 충격 테스트 시나리오", ["고금리 장기화 (이자부담 극대화)", "경기 침체 (매출채권 부실화)", "자금시장 경색 (차환 리스크)"])
    st.divider()
    
    raw_df = load_and_merge_data()
    
    if raw_df is not None and not raw_df.empty:
        selected_corp = st.selectbox("투자 심의 대상", raw_df['회사명'].unique())
        if 'current_corp' not in st.session_state or st.session_state.current_corp != selected_corp:
            st.session_state.current_corp = selected_corp
            st.session_state.ebm_results = {}
            st.session_state.llm_results = {}
            st.session_state.top_5_features = None

st.title("AI 투자 적격성 평가 파이프라인 프로토타입")
st.write("EBM 기반; 재무제표 예측(1페이지) -> LLM 기반; 부실 징후 2차 분석(2페이지) -> 다중 LLM 에이전트 투자 심의(3페이지)")

if raw_df is not None and not raw_df.empty:
    corp_data = raw_df[raw_df['회사명'] == selected_corp].sort_values('회계년도')
    st.session_state.current_corp_data = corp_data 

    tab1, tab2, tab3 = st.tabs(["1단계: 재무 건전성 및 OCF 예측", "2단계: 부도 리스크 재조사", "3단계: 가상 투자 심의 위원회"])

    with tab1:
        main_2.run_main(raw_df, corp_data, selected_corp)
    with tab2:
        sub_2.run_sub(selected_corp, api_key_input)
    with tab3:
        end_2.run_end(selected_corp, api_key_input, macro_scenario, corp_data, raw_df)
else:
    st.info("데이터 로딩 중이거나 사용할 수 있는 데이터가 없습니다.")