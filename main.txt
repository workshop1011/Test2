import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, roc_auc_score

feat_names_kor = {
    'log_revenue': '매출규모(Log)', 'debt_ratio': '부채비율(%)', 'current_ratio': '유동비율(%)',
    'operating_margin': '영업이익률(%)', 'roa': '총자산이익률(ROA)', 'revenue_to_cash': '현금창출력'
}

def run_main(data, corp_data, selected_corp):
    st.subheader("1. AI 모델 기반 타겟 기업 재무 해부")
    task_mode = st.radio("🧠 AI 분석 태스크를 선택하세요:", ["📈 자산 규모 예측 (회귀 모델)", "🚨 재무 부실 위험 감지 (분류 모델)"], horizontal=True)
    
    q_low, q_hi = data['log_assets'].quantile(0.01), data['log_assets'].quantile(0.99)
    filtered_data = data[(data['log_assets'] > q_low) & (data['log_assets'] < q_hi)].copy()
    
    if "회귀" in task_mode:
        features = ['current_ratio', 'roa', 'operating_margin', 'revenue_to_cash', 'log_revenue', 'debt_ratio']
        target = 'log_assets'
        X = filtered_data[features]; y = filtered_data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if st.session_state.trained_reg_model is None:
            with st.spinner("회귀 모델 훈련 중..."):
                st.session_state.trained_reg_model = ExplainableBoostingRegressor(interactions=15, random_state=42).fit(X_train, y_train)
        model = st.session_state.trained_reg_model
    else:
        features = ['roa', 'operating_margin', 'revenue_to_cash', 'log_revenue']
        target = 'is_high_risk'
        X = filtered_data[features]; y = filtered_data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if st.session_state.trained_clf_model is None:
            with st.spinner("분류 모델 훈련 중..."):
                st.session_state.trained_clf_model = ExplainableBoostingClassifier(interactions=10, random_state=42).fit(X_train, y_train)
        model = st.session_state.trained_clf_model

    st.success("✅ AI 모델 세팅 완료! (전체 KOSPI Test Set 기준)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    if "회귀" in task_mode:
        y_pred = model.predict(X_test)
        m_col1.metric("R² (결정계수)", f"{r2_score(y_test, y_pred):.3f}")
        m_col2.metric("MAE (평균 오차)", f"{mean_absolute_error(y_test, y_pred):.3f}")
        m_col3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
        m_col4.info("회귀(Regression) 모드 작동 중")
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1] 
        y_pred_custom = (y_pred_proba >= 0.50).astype(int)
        m_col1.metric("Accuracy (정확도)", f"{accuracy_score(y_test, y_pred_custom)*100:.1f} %")
        m_col2.metric("Recall (재현율)", f"{recall_score(y_test, y_pred_custom)*100:.1f} %")
        m_col3.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.3f}")
        m_col4.error("분류(Classification) 모드 작동 중")

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("#### 🌍 AI 판단 지표 중요도 (Global)")
        global_data = model.explain_global().data()
        kor_names = [feat_names_kor.get(n, n) for n in global_data['names']]
        imp_df = pd.DataFrame({'Feature': kor_names, 'Importance': global_data['scores']}).sort_values(by='Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Teal' if "회귀" in task_mode else 'Reds')
        fig_imp.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    with c_chart2:
        st.markdown(f"#### 🧠 {selected_corp} 맞춤형 AI 추론 (Local)")
        corp_idx = filtered_data[filtered_data['회사명'] == selected_corp].index[-1]
        corp_X = filtered_data.loc[[corp_idx]][features]
        exp_data = model.explain_local(corp_X, pd.Series([filtered_data.loc[corp_idx, target]])).data(0)
        base_score = exp_data['extra']['scores'][0] 
        f_scores = exp_data['scores'] 
        f_names = [feat_names_kor.get(n.split(' x ')[0], n.split(' x ')[0]) if ' x ' not in n else "시너지 효과" for n in exp_data['names']]
        
        fig_water = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute"] + ["relative"]*len(f_names) + ["total"],
            x=["Base"] + f_names + ["예측치"], textposition="outside",
            text=[f"{base_score:.1f}"] + [f"{s:+.1f}" for s in f_scores] + [f"{sum(f_scores)+base_score:.1f}"],
            y=[base_score] + list(f_scores) + [0], connector={"line":{"color":"#3f3f3f"}},
            decreasing={"marker":{"color":"#EF553B"}}, increasing={"marker":{"color":"#00CC96"}}, totals={"marker":{"color":"#636EFA"}}
        ))
        fig_water.update_layout(height=350, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_water, use_container_width=True)

    st.divider()
    st.subheader("2. 핵심 현금흐름(OCF) 시계열 예측 (Phase 2 연계용)")
    recent_ocf = corp_data.tail(4)
    if len(recent_ocf) >= 2:
        X_ts = np.arange(len(recent_ocf)).reshape(-1, 1)
        y_ts = recent_ocf['영업활동 현금(OCF)'].values
        if not st.session_state.ebm_results:
            try: ts_model = ExplainableBoostingRegressor(interactions=0, validation_size=0).fit(X_ts, y_ts)
            except: ts_model = LinearRegression().fit(X_ts, y_ts)
            future_ocf = float(ts_model.predict([[len(recent_ocf)]])[0])
            st.session_state.ebm_results = {'base_ocf': future_ocf, 'r2': r2_score(y_ts, ts_model.predict(X_ts))}
        
        st.info(f"👉 **{selected_corp}**의 1년 뒤 영업활동 현금흐름(OCF)은 **{st.session_state.ebm_results['base_ocf']:,.0f} 천원**으로 예측되었습니다. 이 수치는 Phase 2의 정성 분석 보정 기반이 됩니다.")
    else:
        st.warning("과거 현금흐름 데이터가 부족하여 시계열 예측을 건너뜁니다.")