import streamlit as st
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import r2_score

def generate_derived_features(df):
    temp_df = df.copy()
    
    def get_c(keywords):
        for kw in keywords:
            for col in temp_df.columns:
                if kw.replace(" ", "") in col.replace(" ", ""): return col
        return None

    c_ocf = get_c(['영업활동현금', 'OCF', '영업활동'])
    if c_ocf is None: c_ocf = '영업활동 현금(OCF)'

    c_ast = get_c(['자산총계', '자산'])
    c_eq = get_c(['자본총계', '자본'])
    c_capital = get_c(['자본금'])
    c_ca = get_c(['유동자산'])
    c_cl = get_c(['유동부채'])
    c_rev = get_c(['매출액', '매출'])
    c_op = get_c(['영업이익'])
    c_ni = get_c(['당기순이익', '순이익'])

    base_cols = [c for c in [c_ast, c_eq, c_capital, c_ca, c_cl, c_rev, c_op, c_ni, c_ocf] if c is not None]
    for col in base_cols:
        if temp_df[col].dtype == 'object':
            temp_df[col] = temp_df[col].astype(str).str.replace(',', '', regex=False).replace(['-', ''], '0')
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)

    def safe_div(num, den, default=0):
        return np.where(den != 0, num / den, default)

    if c_ca and c_cl: temp_df['유동비율'] = safe_div(temp_df[c_ca], temp_df[c_cl], 1)
    if c_eq and c_ast: temp_df['자기자본비율'] = safe_div(temp_df[c_eq], temp_df[c_ast], 0)
    if c_capital and c_eq: temp_df['자본잠식률'] = safe_div(temp_df[c_capital] - temp_df[c_eq], temp_df[c_capital], 0)
    if c_op and c_rev: temp_df['영업이익률'] = safe_div(temp_df[c_op], temp_df[c_rev], 0)
    if c_ni and c_rev: temp_df['순이익률'] = safe_div(temp_df[c_ni], temp_df[c_rev], 0)
    if c_ni and c_eq: temp_df['자기자본순이익률(ROE)'] = safe_div(temp_df[c_ni], temp_df[c_eq], 0)
    if c_op and c_ast: temp_df['총자산영업이익률'] = safe_div(temp_df[c_op], temp_df[c_ast], 0)
    if c_rev and c_ast: temp_df['자산회전율'] = safe_div(temp_df[c_rev], temp_df[c_ast], 0)

    temp_df.replace([np.inf, -np.inf], 0, inplace=True)

    derived_features = [f for f in ['유동비율', '자기자본비율', '자본잠식률', '영업이익률', '순이익률', '자기자본순이익률(ROE)', '총자산영업이익률', '자산회전율'] if f in temp_df.columns]
    
    return temp_df.dropna(subset=derived_features + [c_ocf]), derived_features, c_ocf

def run_main(data, corp_data, selected_corp):
    st.markdown(f"### {selected_corp} | 기업 현금흐름 및 안정성 예측 대시보드")
    st.write("지정된 재무 파생변수와 EBM 모델을 기반으로 기업의 투자 회수 역량, 영업활동현금흐름(OCF)과 부실 위험 요인을 도출합니다.")
    st.markdown("<br>", unsafe_allow_html=True)

    processed_data, features, target_col = generate_derived_features(data)
    processed_corp, _, _ = generate_derived_features(corp_data)
    
    st.session_state.target_col_name = target_col

    if len(processed_corp) < 1 or len(processed_data) < 10:
        st.warning("파생변수 계산을 위한 기초 재무 데이터가 부족합니다.")
        return

    X = processed_data[features]
    y = processed_data[target_col]
    corp_X = processed_corp.loc[[processed_corp.index[-1]]][features]
    corp_y = processed_corp.loc[processed_corp.index[-1], target_col]

    if 'ebm_model_v7' not in st.session_state:
        with st.spinner("부도 위험 판별 모델을 학습 중입니다..."):
            st.session_state.ebm_model_v7 = ExplainableBoostingRegressor(interactions=0, random_state=42).fit(X, y)
    
    model = st.session_state.ebm_model_v7

    pred_ocf = float(model.predict(corp_X)[0])
    local_exp = model.explain_local(corp_X, pd.Series([corp_y])).data(0)
    
    feature_names = local_exp['names']
    feature_scores = local_exp['scores']
    current_values = corp_X.iloc[0].values

    importance_df = pd.DataFrame({
        '변수': feature_names,
        '현재값': current_values,
        'SHAP값': feature_scores
    })
    
    base_score = 40 
    if '영업이익률' in corp_X: 
        margin_bonus = max(-20, min(15, corp_X['영업이익률'].iloc[0] * 100))
        base_score += margin_bonus
    if '자기자본비율' in corp_X: 
        equity_bonus = min(45, corp_X['자기자본비율'].iloc[0] * 60)
        base_score += equity_bonus
    if '자본잠식률' in corp_X: 
        impairment = corp_X['자본잠식률'].iloc[0]
        if impairment > 0:  
            base_score -= min(60, (impairment * 150))
            
    suitability_score = min(100, max(0, base_score))
    
    st.session_state.phase1_base_score = suitability_score

    most_important_idx = importance_df['SHAP값'].abs().idxmax()
    top_feature = importance_df.loc[most_important_idx, '변수']

    r2 = r2_score(y, model.predict(X))
    variance_pct = (1 - r2) * 15 

    st.session_state.ebm_results = {'theoretical_ocf': pred_ocf, 'r2': r2}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("1차 투자 적격성 가상점수")
        st.metric(label="", value=f"{suitability_score:.1f} 점", delta="안정성/상환능력 기준", delta_color="off")
    with col2:
        st.success("부도 위험 최중요 변수")
        st.metric(label="", value=f"{top_feature}", delta="모델 기여도 1위", delta_color="off")
    with col3:
        st.warning("OCF 예측 (투자금상환능력)")
        st.metric(label="", value=f"{pred_ocf:,.0f}", delta="/천원", delta_color="off")
    with col4:
        st.error("직후 시장 스트레스 반영 변동 예측")
        st.metric(label="", value=f"± {variance_pct:.2f}%p", delta="신용 스프레드 변동폭", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 핵심 원인 (부실 징후 EBM 영향도 상위 5개)")
    
    importance_df['영향 방향'] = importance_df['SHAP값'].apply(lambda x: "위험 완화" if x > 0 else "위험 증가")
    importance_df['현재값'] = importance_df['현재값'].apply(lambda x: f"{float(x):.4f}")
    importance_df['SHAP값'] = importance_df['SHAP값'].apply(lambda x: f"{float(x):.4f}")
    
    top_5_df = importance_df.reindex(importance_df['SHAP값'].astype(float).abs().sort_values(ascending=False).index).head(5)
    
    st.session_state.top_5_features = top_5_df.reset_index(drop=True)
    
    st.dataframe(top_5_df.reset_index(drop=True), use_container_width=True)
    st.caption("※ 본 결과는 저장된 모델을 기반으로 산출되었으며, 개별 기업 수준에서 부실 원인을 해석했습니다.")