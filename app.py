import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import streamlit as st
import pandas as pd
import numpy as np

import main
import sub
import end

st.set_page_config(page_title="AI 퀀타멘탈 마스터 앱", layout="wide", page_icon="🏦")

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

def engineer_features(df):
    try:
        def get_col(keywords, exclude=None):
            if isinstance(keywords, str): keywords = [keywords]
            for kw in keywords:
                for col in df.columns:
                    clean_col = col.replace(" ", "")
                    if kw.replace(" ", "") in clean_col:
                        if exclude and exclude.replace(" ", "") in clean_col: continue
                        return col
            return None 

        c_ca = get_col(['유동자산'])
        c_cl = get_col(['유동부채'])
        c_ni = get_col(['당기순이익', '당기순손익', '순이익'])
        c_ta = get_col(['자산총계', '자산'], exclude='유동') 
        c_oi = get_col(['영업이익', '영업손익', '영업수익']) 
        c_rev = get_col(['매출액', '매출', '영업수익'])
        c_ocf = get_col(['영업활동현금흐름', '영업활동으로인한현금흐름', '현금흐름'])
        c_liab = get_col(['부채총계', '부채'], exclude='유동')
        c_eq = get_col(['자본총계', '자본'], exclude='자본금')

        def clean_num(val):
            if pd.isna(val): return np.nan
            v = str(val).strip().replace(',', '')
            if v in ['-', '', '0']: return 0.0
            try: return float(v)
            except: return np.nan

        for c in [c_ca, c_cl, c_ni, c_ta, c_oi, c_rev, c_ocf, c_liab, c_eq]:
            if c: df[c] = df[c].apply(clean_num)
        
        df['current_ratio'] = (df[c_ca] / df[c_cl]) * 100 if c_ca and c_cl else 100
        df['roa'] = (df[c_ni] / df[c_ta]) * 100 if c_ni and c_ta else 0
        df['operating_margin'] = np.where(df[c_rev] > 0, (df[c_oi] / df[c_rev]) * 100, 0)
        df['revenue_to_cash'] = np.where(df[c_rev] > 0, (df[c_ocf] / df[c_rev]), 0)
        df['debt_ratio'] = np.where(df[c_eq] > 0, (df[c_liab] / df[c_eq]) * 100, 500) if c_eq and c_liab else 100
        df['log_revenue'] = np.log1p(df[c_rev].clip(lower=0)) 
        df['log_assets'] = np.log1p(df[c_ta].clip(lower=0)) if c_ta else 0
        df['is_high_risk'] = ((df['debt_ratio'] > 200) | (df['current_ratio'] < 100)).astype(int)
        df['영업활동 현금(OCF)'] = df[c_ocf]
        df['이익의 질(OCF/NI)'] = np.where(df[c_ni] > 0, df[c_ocf] / df[c_ni], 0)
        
        # 💡 [핵심 해결] end.py가 찾을 수 있도록 한국어 이름표를 확실하게 달아줍니다.
        df['매출액'] = df[c_rev]
        df['영업이익'] = df[c_oi]
        df['영업이익률(%)'] = df['operating_margin']
        
        return df.dropna(subset=['current_ratio', 'operating_margin', 'log_revenue', 'debt_ratio', '영업활동 현금(OCF)'])
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        return None

with st.sidebar:
    st.header("⚙️ 마스터 설정")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...").strip()
    macro_scenario = st.selectbox("🌍 거시경제 시나리오 (스트레스 테스트용)", ["고금리 및 인플레이션 지속", "경기 침체(Recession)", "금리 인하 및 유동성 공급", "글로벌 공급망 위기"])
    st.divider()
    
    raw_df = load_and_merge_data()
    data = engineer_features(raw_df) if raw_df is not None else None
    
    if data is not None and not data.empty:
        selected_corp = st.selectbox("🎯 타겟 기업 선택", data['회사명'].unique())
        if 'current_corp' not in st.session_state or st.session_state.current_corp != selected_corp:
            st.session_state.current_corp = selected_corp
            st.session_state.ebm_results = {}
            st.session_state.llm_results = {}

st.title("🏦 AI 퀀타멘탈 마스터 파이프라인")
st.write("EBM 딥다이브 분석(Phase 1) ➔ LLM 주석/뉴스 정성 보정(Phase 2) ➔ 다중 에이전트 투자 심의(Phase 3)")

if data is not None and not data.empty:
    corp_data = data[data['회사명'] == selected_corp].sort_values('회계년도')
    tab1, tab2, tab3 = st.tabs(["📊 Phase 1: EBM 재무 해부 및 예측", "📝 Phase 2: 정성 보정 (LLM)", "⚖️ Phase 3: 투자 심의 (위원회)"])

    with tab1:
        main.run_main(data, corp_data, selected_corp)

    with tab2:
        sub.run_sub(selected_corp, api_key_input)

    with tab3:
        end.run_end(selected_corp, api_key_input, macro_scenario, corp_data, raw_df)
else:
    st.info("데이터 로딩 중이거나 사용할 수 있는 데이터가 없습니다.")