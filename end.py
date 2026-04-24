import streamlit as st
import json
import pandas as pd
try:
    from openai import OpenAI
except ImportError:
    pass

def run_committee(company, data_summary_str, api_key, scenario):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    agents = [
        {"role": "리스크 관리자 (보수적)", "icon": "🛡️", "prompt": "1단계 EBM 수치를 불신하고, 우발채무와 거시경제 위협을 극대화하여 OCF를 깎아야 한다고 주장."},
        {"role": "성장 분석가 (공격적)", "icon": "🚀", "prompt": "뉴스 호재를 바탕으로 EBM 예측이 기업 모멘텀을 과소평가했다고 반박하며 긍정적 투자를 권장."},
        {"role": "계량 분석가 (퀀트)", "icon": "🔢", "prompt": "리스크/성장 분석가의 텍스트 해석을 비판하고, EBM의 신뢰도와 마진율만을 근거로 냉정하게 중심을 잡기."}
    ]
    debate_log = []
    for agent in agents:
        prompt = f"""
        투자 위원회 {agent['role']}. [기업: {company} | 거시경제: {scenario}]
        [데이터 요약]: {data_summary_str}
        미션: {agent['prompt']}\nJSON 응답: {{ "comment": "발언 내용(해요체 3문장)" }}
        """
        try:
            res = json.loads(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}).choices[0].message.content)
            debate_log.append({"role": agent['role'], "icon": agent['icon'], "content": res['comment']})
        except: pass

    prompt_chair = f"""
    의장으로서 {company}의 최종 의견을 결정하세요. [거시경제: {scenario}] | 토론: {debate_log}
    JSON 응답: {{ "decision": "Strong Buy / Buy / Hold / Sell", "synthesis": "최종 결론(4문장)", "score": 85, "stress_test_impact": "시나리오 영향 요약" }}
    """
    try:
        final = json.loads(client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_chair}], response_format={"type": "json_object"}).choices[0].message.content)
    except: final = {"decision": "Error", "synthesis": "실패", "score": 0, "stress_test_impact": ""}
    return {"debate": debate_log, "final": final}

def run_end(selected_corp, api_key_input, macro_scenario, corp_data, raw_df):
    st.subheader("⚖️ 거시경제 스트레스 테스트 및 4인 AI 투자 심의")
    if not st.session_state.llm_results:
        st.warning("⚠️ Phase 1과 Phase 2 분석이 모두 완료되어야 위원회가 열립니다.")
        return

    st.info("Phase 1의 **EBM 신뢰도**와 Phase 2의 **LLM 정성적 보정 수치**를 바탕으로, 각기 다른 페르소나의 AI가 난상 토론을 벌입니다.")
    
    if st.button("🎙️ 가상 투자 심의 컨퍼런스 개최", type="primary", use_container_width=True):
        safe_summary = {
            "Phase_1_Quant": {"EBM_Base_OCF": float(st.session_state.ebm_results['base_ocf']), "Model_R2": float(st.session_state.ebm_results['r2'])},
            "Phase_2_Qual": {"LLM_Adjustment_Percent": float(st.session_state.llm_results['adjustment_percent']), "Adjusted_OCF": float(st.session_state.llm_results['adj_ocf'])},
            "Current_Operating_Margin_Percent": float(corp_data['영업이익률(%)'].iloc[-1])
        }
        summary_str = json.dumps(safe_summary, ensure_ascii=False)
        
        with st.status("👨‍💼 위원들이 리포트와 거시경제 지표를 검토 중입니다...", expanded=True) as status:
            committee_res = run_committee(selected_corp, summary_str, api_key_input, macro_scenario)
            
            if committee_res:
                status.update(label="✅ 끝장 토론 종료! 판결이 준비되었습니다.", state="complete", expanded=False)
                
                st.markdown(f"### 🌍 거시경제 스트레스 환경: **{macro_scenario}**")
                st.caption(committee_res['final']['stress_test_impact'])
                st.divider()
                
                for chat in committee_res['debate']:
                    with st.chat_message(chat['role'], avatar=chat['icon']):
                        st.write(f"**{chat['role']}**")
                        st.write(chat['content'])
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.chat_message("의장", avatar="⚖️"):
                    st.error(f"### 📢 의장 최종 판결: {committee_res['final']['decision']}")
                    st.write(committee_res['final']['synthesis'])
                    st.metric("종합 매력도 점수", f"{committee_res['final'].get('score', 0)} / 100")
            else:
                status.update(label="⚠️ API 키를 확인해주세요.", state="error")

    st.divider()
    st.markdown("### 🔗 참조 데이터 및 리포트")
    c1, c2 = st.columns(2)
    with c1:
        st.info("📂 **손익계산서(IS) Raw Data 요약**")
        st.dataframe(corp_data[['회계년도', '매출액', '영업이익', '영업이익률(%)']] if len(corp_data) > 0 else pd.DataFrame(), hide_index=True)
    with c2:
        st.info("📑 **자본변동표**")
        st.warning("⚠️ **자본변동표 데이터 미포함**")
        st.write("해당 기업의 자본변동 데이터는 현재 시스템에 적재되지 않았습니다.")