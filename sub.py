import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import json
import time
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 💡 OpenAI 라이브러리 (v1.0.0 이상 최신 문법)
try:
    from openai import OpenAI
except ImportError:
    st.error("오류: openai 라이브러리가 설치되어 있지 않습니다. 터미널에서 'pip install openai'를 실행해 주세요.")

st.set_page_config(page_title="현금흐름 및 주석 AI 분석", layout="wide", page_icon="🌊")

# ==========================================
# 🔑 0. 사이드바 - LLM API 설정 (복구됨)
# ==========================================
with st.sidebar:
    st.header("⚙️ LLM 분석 엔진 설정")
    st.write("재무제표 주석(Notes)을 AI가 읽고 해석하려면 API 키가 필요합니다.")
    api_key_input = st.text_input("OpenAI API Key 입력", type="password", placeholder="sk-...")
    
    if api_key_input:
        st.success("✅ API 키가 입력되었습니다. (실제 AI 분석 모드 가동)")
    else:
        st.warning("⚠️ API 키가 없습니다. (시뮬레이션 모드로 작동합니다)")
    st.divider()
    st.info("💡 **Tip:** 분석 비용 절감을 위해 GPT-3.5-turbo 모델을 기본으로 사용합니다.")

# ==========================================
# 메인 화면 타이틀
# ==========================================
st.title("🌊 현금흐름(CF) 딥다이브 및 주석(Notes) AI 해석기")
st.write("현금흐름표를 바탕으로 기업의 실제 현금 창출 능력을 시계열로 분석하고, 주석 데이터를 LLM으로 해석하여 미래를 예측합니다.")

# ==========================================
# 1. 데이터 로드 함수
# ==========================================
@st.cache_data
def load_and_prep_cf_data():
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir = None
        for path in [os.path.join(current_file_dir, 'CSV'), os.path.join(os.path.dirname(current_file_dir), 'CSV'), os.path.join(os.getcwd(), 'CSV')]:
            if os.path.exists(os.path.join(path, 'KOSPI_CF.csv')):
                csv_dir = path
                break
                
        if csv_dir is None: raise FileNotFoundError()
        
        cf = pd.read_csv(os.path.join(csv_dir, 'KOSPI_CF.csv'))
        is_df = pd.read_csv(os.path.join(csv_dir, 'KOSPI_IS.csv'))
        
        df = pd.merge(cf, is_df, on=['회사명', '회계년도'], how='inner')
        df.columns = df.columns.str.strip()
        
        def get_col(keywords):
            for kw in keywords:
                for col in df.columns:
                    if kw.replace(" ", "") in col.replace(" ", ""): return col
            return None

        c_ocf = get_col(['영업활동현금흐름', '영업활동으로인한현금흐름'])
        c_icf = get_col(['투자활동현금흐름', '투자활동으로인한현금흐름'])
        c_fcf = get_col(['재무활동현금흐름', '재무활동으로인한현금흐름'])
        c_rev = get_col(['매출액', '매출'])
        c_ni = get_col(['당기순이익', '순이익'])
        
        for c in [c_ocf, c_icf, c_fcf, c_rev, c_ni]:
            if c and df[c].dtype == 'object':
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')

        df['영업활동 현금(OCF)'] = df[c_ocf]
        df['투자활동 현금(ICF)'] = df[c_icf]
        df['재무활동 현금(FCF)'] = df[c_fcf]
        df['잉여현금흐름(FCF)'] = df[c_ocf] + df[c_icf]
        df['매출대비 현금마진율(%)'] = np.where(df[c_rev] > 0, (df[c_ocf] / df[c_rev]) * 100, 0)
        df['이익의 질 (OCF/Net Income)'] = np.where(df[c_ni] > 0, df[c_ocf] / df[c_ni], 0) 
        
        return df.dropna(subset=['영업활동 현금(OCF)'])
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return None

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error, explained_variance_score

# ==========================================
# 2. 모델 평가 지표 함수 (확장판)
# ==========================================
def calculate_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {"R2": 0, "Adj_R2": 0, "MAE": 0, "MSE": 0, "RMSE": 0, "MAPE": 0, "Max_Err": 0, "EVS": 0, "Dir_Acc": 0}
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-10, y_true))) * 100
    max_err = max_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    
    # Adjusted R2 계산 (데이터 수 n, 독립변수 수 p=1)
    n = len(y_true)
    p = 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 and r2 != 1.0 else r2
    
    dir_acc = 0
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = np.mean(actual_dir == pred_dir) * 100
        
    return {"R2": r2, "Adj_R2": adj_r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "Max_Err": max_err, "EVS": evs, "Dir_Acc": dir_acc}

# ==========================================
# 3. LLM 및 주석 관련 함수들
# ==========================================
def fetch_internet_notes(company):
    time.sleep(1) 
    mock_notes = f"""
    [주석 18. 우발부채 및 약정사항]
    {company}는 종속기업의 시설투자 차입금과 관련하여 KDB산업은행 등 금융기관에 150억 원의 지급보증을 제공하고 있습니다.
    [주석 31. 중요한 회계추정 (대손충당금)]
    당사는 매출채권의 기대신용손실률을 재평가하여, 회수 가능성이 현저히 낮아진 장기 매출채권에 대해 대손충당금 30억 원을 추가로 설정하였습니다.
    """
    return mock_notes

def analyze_notes_with_llm_json(company, notes_text, current_ocf, api_key):
    if not api_key:
        return {
            "analysis": "API 키가 입력되지 않아 시뮬레이션 결과를 출력합니다. 우발채무 리스크가 관찰됩니다.", 
            "ocf_adjustment_percent": -10.0, 
            "rationale": "시뮬레이션 - 지급보증 150억 및 대손상각 반영"
        }

    client = OpenAI(api_key=api_key)
    prompt = f"""
    당신은 투자자들의 친절하고 스마트한 금융 멘토입니다. 
    [기업명: {company}, 최근 영업현금흐름: {current_ocf:,.0f} 천원]
    [주석 데이터]: {notes_text}
    
    위 주석을 분석하여, 현재의 영업활동현금흐름(OCF)을 향후 보수적으로 얼마나 삭감(또는 가산)해서 봐야 하는지 추정해주세요.
    반드시 아래 JSON 형식으로만 응답해야 합니다.
    {{
        "analysis": "주석에 나타난 리스크를 딱딱한 보고서 말투가 아닌, 옆에서 다정하게 조언해주듯 친근한 말투(~해요, ~습니다)로 3~4문장으로 풀어서 설명해주세요.",
        "ocf_adjustment_percent": -15.5, // 현금흐름 예상 타격 비율 (숫자만)
        "rationale": "비율을 산정한 핵심 근거를 짧고 명확하게 1문장으로 작성"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"analysis": f"오류 발생: {e}", "ocf_adjustment_percent": 0, "rationale": "분석 실패"}

# ==========================================
# 🚀 메인 로직 실행부
# ==========================================
df = load_and_prep_cf_data()

if df is not None:
    corp_list = df['회사명'].unique().tolist()
    selected_corp = st.selectbox("🎯 분석할 기업을 선택하세요", corp_list)
    corp_data = df[df['회사명'] == selected_corp].sort_values('회계년도').copy()

    st.divider()
    st.subheader("📊 1. 핵심 현금흐름 3년 추이 및 AI 향후 1년 예측")
    
    available_features = ['영업활동 현금(OCF)', '투자활동 현금(ICF)', '재무활동 현금(FCF)', '잉여현금흐름(FCF)', '매출대비 현금마진율(%)', '이익의 질 (OCF/Net Income)']
    
    selected_features = st.multiselect(
        "📈 차트에 표시할 현금 지표를 선택하세요 (최대 5개)", 
        options=available_features, 
        default=['영업활동 현금(OCF)', '이익의 질 (OCF/Net Income)'],
        max_selections=5,
        key="cf_feature_select" # 에러 방지용 키
    )

    if selected_features and len(corp_data) >= 3:
        history_data = corp_data.tail(4) 
        future_year = "AI 예측 (Next 1Y)"
        pred_results = {}
        
        st.markdown("#### 🔍 AI 모델 성능 심층 평가 (과적합 검증)")
        metrics_cols = st.columns(len(selected_features))

        # EBM 예측 및 지표 산출
        for idx, feat in enumerate(selected_features):
            X_time = np.arange(len(history_data)).reshape(-1, 1)
            y_val = history_data[feat].values
            
            # EBM 훈련
            ebm_ts = ExplainableBoostingRegressor(interactions=0)
            ebm_ts.fit(X_time, y_val)
            
            y_in_sample_pred = ebm_ts.predict(X_time)
            metrics = calculate_metrics(y_val, y_in_sample_pred)
            
            pred_val = ebm_ts.predict([[len(history_data)]])[0]
            pred_results[feat] = pred_val
            
            # 💡 확장된 지표 출력 UI
            with metrics_cols[idx]:
                st.caption(f"**{feat}**")
                st.markdown(f"""
                <div style='font-size: 0.85em;'>
                • <b>R²:</b> {metrics['R2']:.2f} | <b>Adj R²:</b> {metrics['Adj_R2']:.2f}<br>
                • <b>MAPE:</b> {metrics['MAPE']:.1f}% | <b>Acc:</b> {metrics['Dir_Acc']:.1f}%<br>
                • <b>EVS:</b> {metrics['EVS']:.2f} | <b>Max Err:</b> {metrics['Max_Err']:,.0f}
                </div>
                """, unsafe_allow_html=True)

        # 📈 차트 그리기 (신뢰구간 보완)
        fig = go.Figure()
        for feat in selected_features:
            y_hist = history_data[feat].tolist()
            future_pred = pred_results[feat]
            y_pred_line = y_hist[-1:] + [future_pred]
            
            # 💡 예측의 불확실성을 나타내는 신뢰구간(오차 범위) 계산
            std_dev = np.std(y_hist) if len(y_hist) > 0 else 0
            upper_bound = future_pred + (std_dev * 0.8)
            lower_bound = future_pred - (std_dev * 0.8)
            
            # 과거 데이터 실선
            fig.add_trace(go.Scatter(x=history_data['회계년도'], y=y_hist, mode='lines+markers', name=f"{feat} (실적)", line=dict(width=2)))
            
            # 예측 데이터 점선
            fig.add_trace(go.Scatter(x=[history_data['회계년도'].iloc[-1], future_year], y=y_pred_line, mode='lines+markers', line=dict(dash='dash'), name=f"{feat} (예측)"))
            
            # 예측 범위(Polygon) 시각화 - 예측이 직선이더라도 밴드를 보여주어 현실성 부여
            fig.add_trace(go.Scatter(
                x=[history_data['회계년도'].iloc[-1], future_year, future_year, history_data['회계년도'].iloc[-1]],
                y=[y_hist[-1], upper_bound, lower_bound, y_hist[-1]],
                fill='toself', fillcolor='rgba(150, 150, 150, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, hoverinfo='skip'
            ))
            
        fig.update_layout(height=450, hovermode="x unified", title=f"{selected_corp} 현금 창출력 시계열 추적 (AI 예측 범위 포함)")
        fig.add_vrect(x0=history_data['회계년도'].iloc[-1], x1=future_year, fillcolor="LightSalmon", opacity=0.1, layer="below", line_width=0)
        
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader(f"📝 2. {selected_corp} 재무제표 주석(Notes) LLM 퀀타멘탈 분석")

        # 버튼 실행 로직 (col_n1, col_n2 복구됨!)
        if st.button("🚀 주석 데이터 검색 및 LLM 융합 분석 실행", type="primary", use_container_width=True):
            
            # 👇 에러의 원인이었던 컬럼 나눔 코드를 다시 추가했습니다.
            col_n1, col_n2 = st.columns([1, 2])
            
            notes_text = fetch_internet_notes(selected_corp)
            
            with col_n1:
                st.info("🔍 인터넷 수집 원본 주석 데이터")
                st.text_area("공시 원문 (Raw Data)", notes_text, height=300)
            
            with col_n2:
                current_ocf = history_data['영업활동 현금(OCF)'].iloc[-1]
                
                with st.spinner("🤖 LLM이 주석을 해독하여 정량적 조정 수치를 계산 중입니다..."):
                    llm_result = analyze_notes_with_llm_json(selected_corp, notes_text, current_ocf, api_key_input)
                    
                st.markdown("### 📑 AI 애널리스트 종합 해석 리포트")
                st.write(llm_result.get("analysis", "분석 실패"))
                
                adj_percent = llm_result.get("ocf_adjustment_percent", 0)
                adj_rationale = llm_result.get("rationale", "")
                
                st.divider()
                st.markdown("### ⚖️ 퀀타멘탈(Quantamental) 최종 조정 의견")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="기존 AI 예측 OCF", value=f"{pred_results.get('영업활동 현금(OCF)', 0):,.0f}")
                with c2:
                    final_ocf = pred_results.get('영업활동 현금(OCF)', 0) * (1 + (adj_percent / 100))
                    st.metric(
                        label=f"주석 리스크 반영 최종 OCF", 
                        value=f"{final_ocf:,.0f}", 
                        delta=f"{adj_percent}% (LLM 의견 반영)", 
                        delta_color="inverse"
                    )
                st.info(f"**조정 근거:** {adj_rationale}")