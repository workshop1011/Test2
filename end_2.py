import streamlit as st
import json
import time
try:
    from openai import OpenAI
except ImportError:
    pass

def get_clean_val(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.replace(" ", "") in col.replace(" ", ""):
                val = df[col].iloc[-1]
                if isinstance(val, str): val = val.replace(',', '').replace('-', '0')
                try: return float(val)
                except: return 0.0
    return 0.0

def run_committee(company, summary_str, api_key, scenario):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    
    agents = [
        {"role": "부도 리스크 관리자", "prompt": "무조건 1순위는 '원금 회수'입니다. 이자보상배율이 낮거나 현금이 비는 현상을 파산의 징조로 보고 투자 불가(Reject)를 주장하세요."},
        {"role": "투자 심의역", "prompt": "회사의 현재 펀더멘탈과 담보 가치에 집중하세요. 리스크 관리자의 의견을 수용하되, 금리를 대폭 올려 조건부 승인(Conditional)이 가능한지 검토하세요."},
        {"role": "신용평가 퀀트", "prompt": "감정적인 해석을 배제하세요. Phase 1의 괴리율과 감점 점수를 통계적으로 분석하여 부도 확률(Probability of Default)을 숫자로만 제시하세요."}
    ]
    
    debate_history = []
    
    for round_num in range(1, 4):
        st.write(f"Round {round_num}: {'기조 발언' if round_num==1 else '신용 리스크 공방' if round_num==2 else '최종 심의 변론'} 진행 중...")
        
        for agent in agents:
            context = "\n".join([f"{d['role']}: {d['content']}" for d in debate_history[-3:]])
            prompt = f"""
            당신은 투자 심의 위원회의 {agent['role']}입니다. [스트레스 시나리오: {scenario}]
            [심의 대상 기업 데이터]: {summary_str}
            [현재 심의 맥락]: {context}
            
            미션 (Round {round_num}/3):
            - {agent['prompt']}
            - 다른 위원들의 의견 중 투자금 회수 관점에서 허점이 있다면 냉정하게 지적하세요.
            - 전문적이고 건조한 해요체로 3문장 이내로 작성하세요.
            
            JSON 응답 형식: {{ "comment": "발언 내용" }}
            """
            try:
                res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                comment = json.loads(res.choices[0].message.content)['comment']
                debate_history.append({"role": agent['role'], "content": comment, "round": round_num})
                time.sleep(0.3)
            except: pass

    prompt_chair = f"""
    당신은 투자 심의 위원회 의장입니다.
    [심의 로그]: {debate_history}
    [시나리오]: {scenario}
    
    미션:
    1. 3라운드의 논의를 종합하여 최종 투자 승인/거절 사유를 요약하세요. (4문장)
    2. 논의 과정을 반영하여 최종 신용 점수에 가감점(-20 ~ +10)을 산정하세요. (안정성 중시이므로 가점 한도가 낮음)
    3. 시나리오 발생 시 예상되는 투자금 회수 리스크를 요약하세요.
    
    JSON 응답 형식:
    {{ 
        "synthesis": "투자 심의 최종 결의안 (4문장)", 
        "committee_adjustment": -20 ~ 10, 
        "stress_test_impact": "시나리오 적용 시 상환 리스크 요약" 
    }}
    """
    try:
        final_res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_chair}], response_format={"type": "json_object"})
        return {"debate": debate_history, "final": json.loads(final_res.choices[0].message.content)}
    except: return None

def run_end(selected_corp, api_key_input, macro_scenario, corp_data, raw_df):
    st.subheader("3단계: 가상 투자 심의 위원회 및 최종 신용등급 산출")
    
    if not st.session_state.llm_results:
        st.warning("1페이지(대시보드) 및 2페이지(정성 조사)가 선행되어야 심의를 시작할 수 있습니다.")
        return

    theoretical = st.session_state.ebm_results['theoretical_ocf']
    actual = st.session_state.llm_results['actual_ocf']
    gap = st.session_state.llm_results['gap']
    gap_ratio = (gap / theoretical) * 100 if theoretical != 0 else 0
    penalty = st.session_state.llm_results.get('penalty_score', 0)

    net_income = get_clean_val(corp_data, ['당기순이익', '순이익'])
    op_income = get_clean_val(corp_data, ['영업이익'])
    int_expense = get_clean_val(corp_data, ['이자비용'])
    icr = (op_income / int_expense) if int_expense > 0 else float('inf')

    st.markdown("### 1. 전 단계에서 도출된 투자 심의 안건 종합 데이터")
    
    st.write("[OCF 건전성 분석]")
    sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
    sum_c1.metric("이론적 기대 OCF", f"{theoretical:,.0f}")
    sum_c2.metric("실제 공시 OCF", f"{actual:,.0f}")
    sum_c3.metric("OCF 괴리율", f"{gap:,.0f}", f"{gap_ratio:.1f}%", delta_color="inverse")
    sum_c4.metric("관련 산출 패널티", f"{penalty} %")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.write("[포괄손익계산서 주요 항목과, 이자 상환 능력 및 잉여 현금 창출력]")
    is_c1, is_c2, is_c3 = st.columns(3)
    with is_c1:
        st.info("최근 당기순이익")
        st.metric(label="", value=f"{net_income:,.0f}")
    with is_c2:
        st.success("최근 영업이익")
        st.metric(label="", value=f"{op_income:,.0f}")
    with is_c3:
        st.warning("이자보상배율 (최소 1 이상 필요)")
        icr_text = f"{icr:.2f} 배" if icr != float('inf') else "무차입/이자없음"
        st.metric(label="", value=icr_text)

    st.markdown("2. 2단계에서 추가 분석한 투자 심의 참조 데이터")
    with st.expander("부실 분석 리포트 (2단계)"):
        st.write(f"핵심 부실 요인: {st.session_state.llm_results.get('gap_reason', '')}")
        st.write(st.session_state.llm_results.get('detective_report', ''))
    
    with st.expander("손익계산서(IS) 핵심 요약 원본"):
        disp_cols = []
        for kw in ['회계년도', '매출', '영업이익', '순이익']:
            for c in corp_data.columns:
                if kw in c.replace(" ", ""):
                    if c not in disp_cols: disp_cols.append(c)
                    break
        
        if not disp_cols: 
            disp_cols = corp_data.columns.tolist()
            
        st.dataframe(corp_data[disp_cols], hide_index=True)
    
    st.divider()

    st.markdown("### 3. 가상 투자적격성 심의회")
    if st.button("3단계에 걸쳐 투자 심의 시작", type="primary", use_container_width=True):
        summary = {
            "company": selected_corp,
            "theoretical_ocf": theoretical,
            "actual_ocf": actual,
            "gap_ratio": gap_ratio,
            "net_income": net_income,
            "interest_coverage_ratio": icr,
            "detective_finding": st.session_state.llm_results.get('gap_reason', '')
        }
        
        with st.status("투자 적격성 심의 위원회가 진행중입니다. 등급 결정을 논의합니다...", expanded=True) as status:
            results = run_committee(selected_corp, json.dumps(summary, ensure_ascii=False), api_key_input, macro_scenario)
            
            if results:
                status.update(label="심의 종료! 최종 신용등급이 도출되었습니다.", state="complete", expanded=False)
                
                for r in range(1, 4):
                    st.markdown(f"#### Round {r} 논의 기록")
                    round_chats = [c for c in results['debate'] if c['round'] == r]
                    for chat in round_chats:
                        with st.chat_message(chat['role']):
                            st.write(f"**{chat['role']}**")
                            st.write(chat['content'])
                    st.write("")

                st.divider()
                
                with st.chat_message("위원회 최종 결론"):
                    base = st.session_state.get('phase1_base_score', 40)
                    committee_adj = results['final'].get('committee_adjustment', 0)
                    
                    final_score = max(0, min(100, base + penalty + committee_adj))
                    
                    if final_score >= 85:
                        auto_rating = "초우량 등급 [AAA ~ AA] : 초우량 투자, 적극 투자 권고"
                    elif final_score >= 65:
                        auto_rating = "투자 적격 [A ~ BBB] : 적격 등급, 대출 등지의 투자 정상 승인"
                    elif final_score >= 45:
                        auto_rating = "투기 등급 [BB] : 요주의 기업, 대출 심의에 집중하며, 투자 비중 축소"
                    elif final_score >= 25:
                        auto_rating = "투기 등급 [B ~ CCC] : 부실 징후, 대출 규모 감소 및 신규 투자 전면 중단"
                    else:
                        auto_rating = "투자 부적격 [C ~ D] : 부도 임박, 대출 불가 및 즉각적인 투자금 회수 절차 돌입"

                    st.error(f"### 최종 판결: {auto_rating}")
                    st.write(results['final']['synthesis'])

                    res_c1, res_c2 = st.columns(2)
                    res_c1.metric("최종 투자 적격성 점수", f"{final_score:.1f} / 100점")
                    res_c2.info(f"스트레스 시나리오 타격: {results['final'].get('stress_test_impact', 'N/A')}")
            else:
                status.update(label="심의 중 오류가 발생했습니다. API 키나 네트워크를 확인하세요.", state="error")