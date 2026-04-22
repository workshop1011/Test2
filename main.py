import streamlit as st
import pandas as pd
import numpy as np
import os
# 💡 [수정] 분류 모델과 분류 평가 지표(Accuracy, Recall, ROC-AUC) 임포트 추가
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EBM 전문 분석 시스템", layout="wide", page_icon="🛡️")

st.title("🛡️ EBM 기업 재무 해부 대시보드")
st.write("AI 모델의 전체 학습 로직(Global)과 개별 기업에 대한 예측 과정(Local)을 투명하게 공개합니다.")

# ==========================================
# 1. 데이터 로드 및 병합 
# ==========================================
@st.cache_data
def load_and_merge_data():
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
        root_dir = os.path.dirname(current_file_dir) 
        working_dir = os.getcwd() 
        
        candidates = [
            os.path.join(current_file_dir, 'CSV'),       
            os.path.join(root_dir, 'CSV'),               
            os.path.join(working_dir, 'CSV'),            
            os.path.join(working_dir, 'pages', 'CSV')    
        ]
        
        csv_dir = None
        for path in candidates:
            if os.path.exists(os.path.join(path, 'KOSPI_BS.csv')):
                csv_dir = path
                break
                
        if csv_dir is None:
            raise FileNotFoundError("CSV 폴더나 파일을 찾을 수 없습니다.")
            
        bs = pd.read_csv(os.path.join(csv_dir, 'KOSPI_BS.csv'))
        is_df = pd.read_csv(os.path.join(csv_dir, 'KOSPI_IS.csv'))
        cf = pd.read_csv(os.path.join(csv_dir, 'KOSPI_CF.csv'))
        
        merge_keys = ['회사명', '회계년도']
        df = pd.merge(bs, is_df, on=merge_keys, how='inner')
        df = pd.merge(df, cf, on=merge_keys, how='inner')
        return df
    except Exception as e:
        st.error(f"데이터 병합 실패: {e}")
        return None

# ==========================================
# 2. 피처 엔지니어링 
# ==========================================
def engineer_features(df):
    try:
        df.columns = df.columns.str.strip()
        
        def get_col(keywords, exclude=None):
            if isinstance(keywords, str): keywords = [keywords]
            for kw in keywords:
                for col in df.columns:
                    clean_col = col.replace(" ", "")
                    if kw.replace(" ", "") in clean_col:
                        if exclude and exclude.replace(" ", "") in clean_col: continue
                        return col
            raise KeyError(f"[{', '.join(keywords)}]")

        c_ca = get_col(['유동자산'])
        c_cl = get_col(['유동부채'])
        c_ni = get_col(['당기순이익', '당기순손익', '순이익'])
        c_ta = get_col(['자산총계', '자산'], exclude='유동') 
        c_oi = get_col(['영업이익', '영업손익', '영업수익']) 
        c_rev = get_col(['매출액', '매출', '영업수익'])
        c_ocf = get_col(['영업활동현금흐름', '영업활동으로인한현금흐름', '현금흐름'])
        c_liab = get_col(['부채총계', '부채'], exclude='유동')
        c_eq = get_col(['자본총계', '자본'], exclude='자본금')

        target_cols = [c_ca, c_cl, c_ni, c_ta, c_oi, c_rev, c_ocf, c_liab, c_eq]
        for c in target_cols:
            if df[c].dtype == 'object':
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
        
        df['current_ratio'] = (df[c_ca] / df[c_cl]) * 100
        df['roa'] = (df[c_ni] / df[c_ta]) * 100
        df['operating_margin'] = (df[c_oi] / df[c_rev]) * 100
        df['revenue_to_cash'] = (df[c_ocf] / df[c_rev])
        df['debt_ratio'] = np.where(df[c_eq] > 0, (df[c_liab] / df[c_eq]) * 100, 500) 
        df['log_revenue'] = np.log1p(df[c_rev].clip(lower=0)) 
        df['log_assets'] = np.log1p(df[c_ta])
        
        # 💡 [신규] 분류 모델용 타겟 변수 생성 (고위험 기업: 부채비율 200% 초과 또는 유동비율 100% 미만)
        df['is_high_risk'] = ((df['debt_ratio'] > 200) | (df['current_ratio'] < 100)).astype(int)
        
        return df.dropna(subset=['current_ratio', 'operating_margin', 'log_revenue', 'log_assets', 'debt_ratio', 'is_high_risk'])
    except KeyError as e:
        st.error(f"❌ 전처리 오류: {e}")
        return None

feat_names_kor = {
    'log_revenue': '매출규모(Log)', 'debt_ratio': '부채비율(%)', 'current_ratio': '유동비율(%)',
    'operating_margin': '영업이익률(%)', 'roa': '총자산이익률(ROA)', 'revenue_to_cash': '현금창출력'
}

# ==========================================
# 3. 메인 실행 블록 (분석 모드 선택 및 성능 지표 출력)
# ==========================================
raw_df = load_and_merge_data()

if raw_df is not None:
    data = engineer_features(raw_df)
    
    if data is not None and not data.empty:
        st.divider()
        task_mode = st.radio(
            "🧠 AI 분석 태스크를 선택하세요:", 
            ["📈 자산 규모 예측 (회귀 모델)", "🚨 재무 부실 위험 감지 (분류 모델)"],
            horizontal=True
        )
        
        q_low = data['log_assets'].quantile(0.01)
        q_hi  = data['log_assets'].quantile(0.99)
        filtered_data = data[(data['log_assets'] > q_low) & (data['log_assets'] < q_hi)].copy()
        
        # 💡 [핵심] Streamlit 임시 기억 장치(Session State)에 모델 저장소 만들기
        if 'trained_reg_model' not in st.session_state:
            st.session_state['trained_reg_model'] = None
        if 'trained_clf_model' not in st.session_state:
            st.session_state['trained_clf_model'] = None

        # ==========================================
        # 💡 회귀/분류 로직 및 임시 저장소 활용
        # ==========================================
        if "회귀" in task_mode:
            features = ['current_ratio', 'roa', 'operating_margin', 'revenue_to_cash', 'log_revenue', 'debt_ratio']
            target = 'log_assets'
            X = filtered_data[features]
            y = filtered_data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 기억 장치에 회귀 모델이 없으면 새로 학습하고, 있으면 꺼내 씁니다.
            if st.session_state['trained_reg_model'] is None:
                with st.spinner("KOSPI 회귀 모델을 최초 1회 훈련 중입니다. 잠시만 기다려주세요..."):
                    model = ExplainableBoostingRegressor(interactions=15, random_state=42)
                    model.fit(X_train, y_train)
                    st.session_state['trained_reg_model'] = model # 학습 완료 후 저장
            else:
                model = st.session_state['trained_reg_model']
                
        else:
            features = ['roa', 'operating_margin', 'revenue_to_cash', 'log_revenue']
            target = 'is_high_risk'
            X = filtered_data[features]
            y = filtered_data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 기억 장치에 분류 모델이 없으면 새로 학습하고, 있으면 꺼내 씁니다.
            if st.session_state['trained_clf_model'] is None:
                with st.spinner("KOSPI 분류 모델을 최초 1회 훈련 중입니다. 잠시만 기다려주세요..."):
                    model = ExplainableBoostingClassifier(interactions=10, random_state=42)
                    model.fit(X_train, y_train)
                    st.session_state['trained_clf_model'] = model # 학습 완료 후 저장
            else:
                model = st.session_state['trained_clf_model']

        # ==========================================
        # 평가 성적표 출력
        # ==========================================
        st.success("✅ AI 모델 세팅 완료! (Test Set 기준 성적표)")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        if "회귀" in task_mode:
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            m_col1.metric("R² (결정계수)", f"{r2:.3f}")
            m_col2.metric("MAE (평균 오차)", f"{mae:.3f}")
            m_col3.metric("RMSE (평균 제곱근 오차)", f"{rmse:.3f}")
            m_col4.info("회귀(Regression) 모드 작동 중")
            
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1] 
            
            st.info("👇 **리스크 관리자의 시각:** 아래 슬라이더를 조절해도 이제 CPU가 요동치지 않습니다!")
            threshold = st.slider("🚨 위험 탐지 경고 기준선 (Threshold)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
            
            y_pred_custom = (y_pred_proba >= threshold).astype(int)
            
            acc = accuracy_score(y_test, y_pred_custom)
            rec = recall_score(y_test, y_pred_custom)
            auc = roc_auc_score(y_test, y_pred_proba) 
            
            m_col1.metric("Accuracy (정확도)", f"{acc*100:.1f} %")
            m_col2.metric("Recall (재현율)", f"{rec*100:.1f} %")
            m_col3.metric("ROC-AUC", f"{auc:.3f}")
            m_col4.error("분류(Classification) 모드 작동 중")

        st.divider()
    

        # ------------------------------------------
        # 탭 1: 모델 전체가 세상을 보는 방식 (Global)
        # ------------------------------------------
        st.subheader("1. AI가 판단한 재무 지표의 중요도 (Global Explanation)")
        
        ebm_global = model.explain_global()
        global_data = ebm_global.data()
        kor_names = [feat_names_kor.get(n, n) for n in global_data['names']]
        
        importance_df = pd.DataFrame({
            '지표(Feature)': kor_names,
            '영향도(Importance)': global_data['scores']
        }).sort_values(by='영향도(Importance)', ascending=True) 
        
        color_scale = 'Teal' if "회귀" in task_mode else 'Reds'
        fig_imp = px.bar(importance_df, x='영향도(Importance)', y='지표(Feature)', orientation='h', 
                         color='영향도(Importance)', color_continuous_scale=color_scale)
        fig_imp.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig_imp, use_container_width=True)

        # ------------------------------------------
        # 탭 2: 특정 기업 맞춤형 AI 뇌구조 해부 (Local Waterfall)
        # ------------------------------------------
        st.subheader("2. 개별 기업 맞춤형 AI 추론 해부 (Local Explanation)")
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            corp_list = filtered_data['회사명'].unique().tolist()
            selected_corp = st.selectbox("분석할 타겟 기업을 선택하세요", corp_list)
        
        corp_idx = filtered_data[filtered_data['회사명'] == selected_corp].index[-1]
        corp_X = filtered_data.loc[[corp_idx]][features]
        corp_y = filtered_data.loc[corp_idx, target]
        
        local_exp = model.explain_local(corp_X, pd.Series([corp_y]))
        exp_data = local_exp.data(0)
        
        base_score = exp_data['extra']['scores'][0] 
        feature_scores = exp_data['scores'] 
        feature_names = [feat_names_kor.get(n, n) for n in exp_data['names']]
        
        clean_names = [f"시너지: {feat_names_kor.get(n.split(' x ')[0], n.split(' x ')[0])}&{feat_names_kor.get(n.split(' x ')[1], n.split(' x ')[1])}" if ' x ' in n else n for n in feature_names]
        
        pred_val = model.predict(corp_X)[0]
        
        fig_water = go.Figure(go.Waterfall(
            name = "EBM 추론 과정", orientation = "v",
            measure = ["absolute"] + ["relative"] * len(clean_names) + ["total"], 
            x = ["기본 점수(Base)"] + clean_names + ["AI 최종 예측"],
            textposition = "outside",
            text = [f"{base_score:.2f}"] + [f"{s:+.2f}" for s in feature_scores] + [f"{sum(feature_scores)+base_score:.2f}"],
            y = [base_score] + list(feature_scores) + [0],
            connector = {"line":{"color":"rgb(63, 63, 63)", "width": 1}},
            decreasing = {"marker":{"color":"#EF553B"}}, increasing = {"marker":{"color":"#00CC96"}}, totals = {"marker":{"color":"#636EFA"}}
        ))
        
        fig_water.update_layout(height=500, showlegend=False, margin=dict(t=20, b=100))
        st.plotly_chart(fig_water, use_container_width=True)
        
        # 최종 결과 텍스트 (모드에 따라 다르게 출력)
        if "회귀" in task_mode:
            pred_real = np.expm1(pred_val)
            actual_real = np.expm1(corp_y)
            st.info(f"👉 **{selected_corp}의 자산규모 분석 결과:** 실제 공시 자산은 **{actual_real / 100000000:,.1f}억원**이며, AI 예측 자산은 **{pred_real / 100000000:,.1f}억원**입니다.")
        else:
            # 확률값 가져오기
            pred_prob = model.predict_proba(corp_X)[0][1] * 100
            risk_status = "🔴 고위험 (High Risk)" if pred_val == 1 else "🟢 정상 (Safe)"
            actual_status = "🔴 고위험" if corp_y == 1 else "🟢 정상"
            st.warning(f"👉 **{selected_corp}의 재무 위험 분석 결과:** 실제 상태는 **[{actual_status}]** 이며, AI는 **{pred_prob:.1f}%의 확률**로 이 기업을 **[{risk_status}]** 상태로 예측했습니다.")