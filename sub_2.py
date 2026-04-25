import streamlit as st
import json
import time
try:
    from openai import OpenAI
except ImportError:
    pass

FEATURE_INFO = {
    '유동비율': {'formula': '유동자산 / 유동부채', 'desc': '단기 채무 상환 능력을 나타내는 최우선 방어 지표'},
    '자기자본비율': {'formula': '자본총계 / 자산총계', 'desc': '외부 차입금 의존도를 낮추고 부실을 흡수할 수 있는 체력'},
    '자본잠식률': {'formula': '(자본금 - 자본총계) / 자본금', 'desc': '기업의 존속 능력을 위협하는 가장 치명적인 부도 경고 시그널'},
    '영업이익률': {'formula': '영업이익 / 매출액', 'desc': '본업을 통해 이자를 갚을 수 있는 현금을 창출하는 마진 능력'},
    '순이익률': {'formula': '당기순이익 / 매출액', 'desc': '이자 및 세금 차감 후 최종 잉여 자금 창출력'},
    '자기자본순이익률(ROE)': {'formula': '당기순이익 / 자본총계', 'desc': '투여된 자본 대비 최종 상환 재원 확보 효율'},
    '총자산영업이익률': {'formula': '영업이익 / 자산총계', 'desc': '보유 자산의 부실화 없이 이익을 내고 있는지 확인하는 지표'},
    '자산회전율': {'formula': '매출액 / 자산총계', 'desc': '보유 자산이 악성 재고로 묶여있지 않고 매출로 직결되는지 확인'}
}

def fetch_mock_data(company):
    notes = f"[주석] 매출채권 회수 기간이 전년 대비 45일 증가함. [약정] 종속회사 채무 보증 150억 존재."
    news = f"- [속보] {company}, 주요 발주처 대금 결제 지연으로 단기 유동성 압박 우려\n- [공시] 자산 재평가로 장부상 자산가치는 상승했으나 현금 유입은 무관함"
    return notes, news

def run_detective_analysis(company, theoretical, actual, gap, notes, news, top_features_str, api_key):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    당신은 신용평가기관의 '투자 부적격 전문 분석역'입니다.
    [기업명: {company}]
    - AI 예측 투자금 상환 재원(OCF): {theoretical:,.0f}
    - 실제 공시 OCF: {actual:,.0f}
    - 괴리(Gap): {gap:,.0f} ({ (gap/theoretical)*100:.1f}% 차이)

    [Phase 1 핵심 파생변수 상태 (부실 징후 모델 추출)]:
    {top_features_str}

    [정성 데이터]: 주석({notes}), 뉴스({news})

    미션: 
    1. AI가 찾아낸 핵심 파생변수의 약점과 주석/뉴스를 연결하여 '부도 가능성'과 '유동성 경색 위험'을 추리하세요.
    2. 성장성이나 호재는 무시하고, 철저히 투자금 회수 관점(Downside Risk)에서 장부의 착시를 지적하세요.
    3. 분석역의 시각에서 최종 신용 패널티 감점(0 ~ -30점)을 산정하세요.
    4. 해요체로 3~4문장으로 작성하세요.

    JSON 응답 형식:
    {{ 
        "detective_report": "부도 리스크에 초점을 맞춘 융합 분석 내용", 
        "penalty_score": -30 ~ 0 사이의 정수, 
        "gap_reason": "투자 원금 손실 원인이 될 수 있는 핵심 리스크 한 줄 요약" 
    }}
    """
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        return json.loads(res.choices[0].message.content)
    except: return None

def run_sub(selected_corp, api_key_input):
    st.subheader("2단계: 영업활동현금흐름(OCF) 괴리 분석 및 부실 정성 조사")
    
    if not st.session_state.ebm_results:
        st.warning("1페이지에서 대시보드 분석을 먼저 실행해 주세요.")
        return
        
    if 'top_5_features' not in st.session_state or st.session_state.top_5_features is None:
        st.error("1단계에서 최중요 변수 데이터가 전달되지 않았습니다. 분석을 다시 실행해 주세요.")
        return

    top_5_df = st.session_state.top_5_features

    st.markdown("### 1단계에서 제공받은, AI 지정 핵심 부실 경고 지표 (Top 5)")
    st.write("EBM 모델이 타겟 기업의 현금 유입에 가장 큰 타격을 주고 있다고 판단한 5가지 파생 변수 및 기타 재무정보의 상세 내역입니다.")
    
    cols = st.columns(5)
    top_features_context = ""
    
    for i, row in top_5_df.iterrows():
        feat_name = row['변수']
        feat_val = float(row['현재값'])
        info = FEATURE_INFO.get(feat_name, {'formula': 'N/A', 'desc': '설명 없음'})
        
        with cols[i % 5]:
            st.success(f"{i+1}. {feat_name}")
            st.metric(label="", value=f"{feat_val:.4f}")
            st.caption(f"수식: {info['formula']}")
            st.caption(f"{info['desc']}")
        
        top_features_context += f"- {feat_name}: 현재값 {feat_val:.4f} (의미: {info['desc']})\n"

    st.markdown("<br>", unsafe_allow_html=True)

    target_col_name = st.session_state.get('target_col_name', '영업활동 현금(OCF)')
    
    actual_ocf_raw = st.session_state.current_corp_data[target_col_name].iloc[-1]
    if isinstance(actual_ocf_raw, str):
        actual_ocf_raw = actual_ocf_raw.replace(',', '').replace('-', '0')
    actual_ocf = float(actual_ocf_raw)
    
    theoretical_ocf = st.session_state.ebm_results['theoretical_ocf']
    gap = actual_ocf - theoretical_ocf
    
    st.markdown("### 영업활동현금흐름(OCF) 실제 확보 괴리율")
    c1, c2, c3 = st.columns(3)
    c1.metric("이론적 기대 OCF", f"{theoretical_ocf:,.0f}")
    c2.metric("실제 공시 OCF", f"{actual_ocf:,.0f}")
    c3.metric("기대치 대비 현금 누수", f"{gap:,.0f}", f"{(gap/theoretical_ocf)*100:.1f}%" if theoretical_ocf != 0 else "0.0%", delta_color="inverse")

    st.divider()

    notes, news = fetch_mock_data(selected_corp)
    st.markdown("### 기업 관련 자연어 데이터 크롤링 (임시 데이터)")
    c_raw1, c_raw2 = st.columns(2)
    with c_raw1:
        st.info("전자공시 우발부채 및 주석 내용")
        st.write(notes)
    with c_raw2:
        st.info("최신 크롤링 신용 악재 뉴스")
        st.write(news)
    
    if st.button("OCF 괴리 및 자연어 분석을 추가하여 부도 리스크 추가 분석", type="primary"):
        with st.status("자연어 데이터를 분석, 장부의 이면과 자금시장 뉴스를 교차 검증 중입니다...", expanded=True) as status:
            st.write("1단계 분석 과정의 재무 방어력 취약점 스캔 완료...")
            time.sleep(0.5)
            st.write("전자공시 주석 내 우발부채 및 자회사 지급보증 위험 추출 중...")
            time.sleep(0.5)
            st.write("부도 시나리오 규명 및 최종 신용 감점 산출 중...")
            
            report = run_detective_analysis(selected_corp, theoretical_ocf, actual_ocf, gap, notes, news, top_features_context, api_key_input)
            
            if report:
                st.session_state.llm_results = report
                st.session_state.llm_results['gap'] = gap
                st.session_state.llm_results['actual_ocf'] = actual_ocf
                status.update(label="부실 징후 규명 및 최종 패널티 산출 완료!", state="complete", expanded=False)
            else:
                status.update(label="분석 실패. API 키를 확인해주세요.", state="error")

    if 'llm_results' in st.session_state and st.session_state.llm_results:
        res = st.session_state.llm_results
        st.error(f"부도 리스크 핵심 요인: {res.get('gap_reason', '')}")
        st.write(res.get('detective_report', ''))
        
        penalty = res.get('penalty_score', 0)
        st.metric("감점 % ", f"{penalty} %")