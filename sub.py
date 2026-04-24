import streamlit as st
import json
import time
try:
    from openai import OpenAI
except ImportError:
    pass 

def fetch_mock_data(company):
    notes = f"[주석 18. 우발부채] {company}는 금융기관에 150억 원의 지급보증을 제공.\n[주석 31. 회계추정] 장기 매출채권 대손충당금 30억 원 추가 설정."
    news = f"- [한국경제] {company}, 재무구조 개선 위해 유휴 부동산 매각 완료... 200억 확보\n- [블룸버그] {company} 소속 산업군, 하반기 수출 호조 전망"
    return notes, news

def run_sub_analysis(company, notes, news, base_ocf, api_key):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    prompt = f"""
    당신은 시니어 크레딧 애널리스트입니다. [기업명: {company}, ML예측 OCF: {base_ocf:,.0f} 천원]
    [주석]: {notes} | [뉴스]: {news}
    주석의 리스크와 뉴스의 호재를 교차 검증하여, ML이 예측한 OCF에 더하거나 뺄 최종 가중치(%)를 결정하세요. JSON 응답:
    {{ "analysis": "주석과 뉴스 교차 분석 결과 (친근한 해요체 3문장)", "adjustment_percent": 5.0, "rationale": "가중치 산정 근거" }}
    """
    try:
        return json.loads(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}).choices[0].message.content)
    except: return None

def run_sub(selected_corp, api_key_input):
    st.subheader("📝 주석 및 뉴스 기반 정성적 OCF 보정")
    if not st.session_state.ebm_results:
        st.warning("⚠️ Phase 1 탭 하단의 'OCF 시계열 예측'이 완료되어야 합니다.")
        return

    notes, news = fetch_mock_data(selected_corp)
    
    st.markdown("### 📡 수집된 Raw Data")
    c_raw1, c_raw2 = st.columns(2)
    with c_raw1:
        st.info("📑 **전자공시 주석 (Notes)**")
        st.write(notes)
    with c_raw2:
        st.info("📰 **최신 크롤링 뉴스 (News)**")
        st.write(news)
    
    if st.button("🚀 주석/뉴스 교차 검증 및 ML 보정 실행", type="primary"):
        with st.status("🤖 AI가 재무제표의 이면과 뉴스의 행간을 엮고 있습니다...", expanded=True) as status:
            st.write("1️⃣ 주석 데이터 리스크 해독 중...")
            time.sleep(0.5)
            st.write("2️⃣ 최신 뉴스 긍정/부정 동향 분석 중...")
            time.sleep(0.5)
            st.write("3️⃣ 주석의 계획과 뉴스의 팩트를 교차 검증하여 융합 중...")
            
            llm_res = run_sub_analysis(selected_corp, notes, news, st.session_state.ebm_results['base_ocf'], api_key_input)
            if llm_res:
                st.session_state.llm_results = llm_res
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
            else:
                status.update(label="⚠️ API 키를 확인해주세요.", state="error")
    
    if st.session_state.llm_results:
        lr = st.session_state.llm_results
        adj_ocf = st.session_state.ebm_results['base_ocf'] * (1 + (lr['adjustment_percent']/100))
        st.session_state.llm_results['adj_ocf'] = adj_ocf 
        
        st.markdown("---")
        st.markdown("### 🧠 AI 애널리스트 최종 종합 검증")
        st.success("💎 **[최종 종합] 주석-뉴스 교차 검증 인사이트**")
        st.write(lr['analysis'])
        
        st.divider()
        st.markdown("### ⚖️ 퀀타멘탈(Quantamental) 최종 조정 의견")
        c1, c2 = st.columns(2)
        c1.metric("Phase 1: EBM 기초 예측 OCF", f"{st.session_state.ebm_results['base_ocf']:,.0f}")
        c2.metric("Phase 2: LLM 리스크 반영 최종 OCF", f"{adj_ocf:,.0f}", f"{lr['adjustment_percent']}% (LLM 가중치)", delta_color="inverse")
        st.caption(f"**최종 산정 근거:** {lr['rationale']}")