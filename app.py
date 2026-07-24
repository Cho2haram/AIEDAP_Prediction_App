import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import bcrypt
import json
from supabase import create_client
from datetime import datetime
import google.generativeai as genai

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(
    page_title="AI 디지털역량 유형 진단",
    page_icon="🎓",
    layout="centered"
)

# ============================================================
# Supabase 연결
# ============================================================
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ============================================================
# 모델 로드
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('rf_model_k3.pkl')

model = load_model()

# ============================================================
# 세션 초기화
# ============================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# ============================================================
# 유틸 함수
# ============================================================
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def login(email, password):
    res = supabase.table('users').select('*').eq('email', email).execute()
    if res.data:
        user = res.data[0]
        if check_password(password, user['password']):
            return user
    return None

def register(email, password, name, school, subject, age):
    # 이메일 중복 확인
    check = supabase.table('users').select('id').eq('email', email).execute()
    if check.data:
        return 'duplicate'
    try:
        hashed = hash_password(password)
        res = supabase.table('users').insert({
            'email': email,
            'password': hashed,
            'name': name,
            'school': school,
            'subject': subject,
            'age': age,
            'is_admin': False,
        }).execute()
        return res.data[0] if res.data else 'error'
    except Exception as e:
        st.error(f"상세 오류: {str(e)}")
        return 'error'

def save_result(user_id, cluster, type_name, lable, comp_scores, responses):
    vals = list(comp_scores.values())
    supabase.table('results').insert({
        'user_id': user_id,
        'cluster': int(cluster),
        'type_name': type_name,
        'lable': lable,
        'score_1': vals[0], 'score_2': vals[1],
        'score_3': vals[2], 'score_4': vals[3],
        'score_5': vals[4], 'score_6': vals[5],
        'score_7': vals[6], 'score_8': vals[7],
        'responses': json.dumps(responses),
    }).execute()

def get_my_results(user_id):
    res = supabase.table('results').select('*')\
        .eq('user_id', user_id)\
        .order('created_at', desc=True).execute()
    return res.data

def get_all_results():
    res = supabase.table('results').select('*, users(name, email, school)').execute()
    return res.data

# ============================================================
# 문항 정의
# ============================================================
subcomp_items = {
    '①기초-사회-교육이해': {
        'A1': 'AI·디지털의 기본적인 개념과 원리를 이해하고 설명할 수 있다.',
        'A2': 'AI·디지털 기술이 교육적으로 활용되는 사례를 설명할 수 있다.',
        'A3': 'AI·디지털 기반 시스템의 기본적인 데이터 수집 및 처리 과정을 설명할 수 있다.',
        'B1': 'AI·디지털 발전에 따른 사회적 변화(예: 생활방식, 교육 방식 등)를 설명할 수 있다.',
        'B2': 'AI·디지털 발전에 따라 새롭게 생겨날 직업과 사라질 직업을 설명할 수 있다.',
        'B3': 'AI·디지털이 개인, 공동체 및 사회에 미치는 긍정적인 영향과 부정적인 영향에 대해 설명할 수 있다.',
        'C1': 'AI·디지털 기술이 개인, 공동체 및 사회에 미치는 긍정적·부정적 영향을 설명할 수 있다.',
        'C2': 'AI·디지털이 교육에 미치는 영향을 설명할 수 있다.',
        'C3': 'AI·디지털 기술을 교육에서 활용하는 방법에 대해서 설명할 수 있다.',
    },
    '②AI윤리실천': {
        'D1': 'AI·디지털 활용에 대한 윤리적 쟁점을 이해하고 올바른 AI 디지털 윤리 실천의 중요성을 설명할 수 있다.',
        'D2': '인종, 성별, 알고리즘이나 개인정보 활용 문제 등 AI 디지털 활용에 따른 윤리적 문제를 다양한 관점에서 검토하고 해결 방안을 논의할 수 있다.',
        'D3': 'AI·디지털 활용에 따른 윤리적 문제를 해결하기 위한 실천 방안을 제시할 수 있다.',
    },
    '③교육과정-개별화설계': {
        'E1': 'AI·디지털을 활용하여 도달할 수 있는 교수-학습 목표의 내용체계 및 성취기준을 파악할 수 있다.',
        'E2': 'AI·디지털을 활용하여 학생의 학습 데이터를 분석하고, 맞춤형 학습 경험을 제공하기 위해 교육과정을 재구성할 수 있다.',
        'E3': 'AI·디지털을 활용하여 교과 지식 및 기능을 효과적으로 교육하기 위한 교수-학습 과정을 재구성할 수 있다.',
        'F1': 'AI·디지털을 활용하여 학습자의 개별 특성을 분석하고, 진단 평가 결과를 기반으로 학습 수준을 파악할 수 있다.',
        'F2': '학습자 개별 특성과 학습 수준을 바탕으로 AI·디지털 도구를 활용하여 맞춤형 학습 환경을 설계하고 운영할 수 있다.',
        'F3': '학습자 개별 특성과 학습 수준을 반영하여 AI·디지털 도구를 활용한 맞춤형 학습활동 및 과제를 설계하고 구성할 수 있다.',
    },
    '④평가설계-기술선정': {
        'G1': 'AI·디지털을 활용하여 학습자의 특성과 학습 과정을 분석할 수 있는 데이터를 선별할 수 있다.',
        'G2': 'AI·디지털을 활용하여 교수-학습과정에서 학습자의 특성과 학습 과정에 대한 데이터를 분석하여 평가 계획을 수립할 수 있다.',
        'G3': 'AI·디지털 기반 평가 계획을 토대로 학습자의 특성과 목표에 적합한 평가 방법과 도구를 선정하거나 개발할 수 있다.',
        'H1': '교육내용, 교수-학습, 평가 등에 적합한 AI·디지털 기술·데이터·서비스·콘텐츠를 탐색하고, 교육적 유용성과 효과성을 평가할 수 있다.',
        'H2': '평가 결과를 바탕으로 학습 목표와 학습자의 특성에 적합한 AI·디지털 기술·데이터·서비스·콘텐츠를 선정할 수 있다.',
        'H3': '선정한 AI·디지털 기술·데이터·서비스·콘텐츠를 교수-학습 목적에 맞게 수정·재구성하여 활용할 수 있다.',
    },
    '⑤매체활용': {
        'I1': '학습자의 수준을 고려하여 교수-학습활동에 적합한 AI·디지털 기반 교수-학습 매체를 탐색하고 교육적 유용성과 효과성을 맞춰 선정할 수 있다.',
        'I2': '학습자에게 AI·디지털 기반 교수-학습 매체 활용 방법을 이해하기 쉽게 지도할 수 있다.',
        'I3': '학습자에게 AI·디지털 기반 교수-학습 매체 활용 후 피드백을 수집하여 개선 방안을 마련할 수 있다.',
    },
    '⑥기술진단-데이터': {
        'J1': '교수-학습과정에서 발생하는 AI·디지털 기술적 문제의 원인을 파악할 수 있다.',
        'J2': '교수-학습 과정에서 발생하는 AI·디지털 기술 관련 장애를 동료 교사나 IT 전문가(디지털 튜터)와 협력하여 해결 방안을 마련할 수 있다.',
        'J3': '교수-학습과정에서 발생하는 AI·디지털 기술 관련 장애를 해결할 수 있다.',
        'K1': 'AI·디지털 기술과 데이터를 활용하여 학생들과 의견을 교환하고 공유할 수 있다.',
        'K2': 'AI·디지털 활용 시 생성되는 데이터의 의미를 파악하여 교수-학습 과정에 효과적으로 활용할 수 있는 방법을 설명할 수 있다.',
        'K3': 'AI·디지털 활용 시 생성되는 데이터를 교수-학습에 활용할 수 있는 형태로 재구성하여 교수-학습 과정에 적용할 수 있다.',
    },
    '⑦평가해석-피드백': {
        'L1': 'AI·디지털이 제공하는 평가 데이터(대시보드)에서 학생들의 학습 성과를 파악할 수 있다.',
        'L2': 'AI·디지털이 제공하는 평가 결과와 학습 데이터에서 나타난 학습자의 개별 특성을 종합적으로 분석하여 학습 상태를 평가할 수 있다.',
        'L3': 'AI·디지털이 제공하는 평가 결과와 해석 내용을 기반으로 교수-학습 과정을 재구성하고 수업에 반영할 수 있다.',
        'M1': 'AI·디지털 도구를 활용하여 분석한 학습 성과를 바탕으로 학생에게 학습과 관련한 적절한 피드백이 무엇인지 파악할 수 있다.',
        'M2': 'AI·디지털 도구를 활용하여 학습 성과와 성취도를 분석하고, 이를 기반으로 학생 맞춤형 학습 피드백을 제공할 수 있다.',
        'M3': 'AI·디지털 도구를 활용하여 분석한 학습결과를 바탕으로 교수-학습 과정을 개선할 수 있다.',
    },
    '⑧개인정보-저작권': {
        'N1': 'AI·디지털 활용 시 접근할 수 있는 개인 정보를 관련 정책과 법을 준수하고 정책에 따라 올바르게 관리할 수 있다.',
        'N2': 'AI·디지털 활용 시 개인정보 관리 과정에서 발생할 수 있는 문제를 진단하고 개선 방안을 제시할 수 있다.',
        'N3': '학습자들에게 AI·디지털 활용 시 개인정보 보호의 중요성과 구체적 보호 방안을 지도할 수 있다.',
        'O1': 'AI·디지털 활용과 관련하여 올바른 저작권 보호 의식을 갖출 수 있다.',
        'O2': 'AI·디지털 활용 시 창작물의 저작권을 침해하지 않는 방법을 설명할 수 있다.',
        'O3': '학습자들에게 AI·디지털 활용 시 이용자의 권리와 책임에 대해 교육할 수 있다.',
    },
}

all_items = [
    'A1','A2','A3','B1','B2','B3','C1','C2','C3',
    'D1','D2','D3',
    'E1','E2','E3','F1','F2','F3',
    'G1','G2','G3','H1','H2','H3',
    'I1','I2','I3',
    'J1','J2','J3','K1','K2','K3',
    'L1','L2','L3','M1','M2','M3',
    'N1','N2','N3','O1','O2','O3',
]

type_names = {0: '실천중심형', 1: '균형형', 2: '이해중심형'}

type_desc = {
    '실천중심형': '매체 활용, 평가 피드백 등 실천 역량이 상대적으로 강하나 AI 윤리 역량 개발이 필요한 유형입니다.',
    '균형형':     '8개 하위역량이 전반적으로 고른 수준을 보이는 유형입니다. 특정 역량의 심화 학습을 통해 전문성을 높일 수 있습니다.',
    '이해중심형': 'AI 윤리 및 개인정보·저작권 이해 역량이 상대적으로 강하나 교육과정 설계, 평가, 매체 활용 등 실천 역량 강화가 필요한 유형입니다.',
}

@st.cache_resource
def load_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemini-1.5-flash')  # 무료 티어

gemini = load_gemini()

def get_ai_recommendation(lable, result, my_total, 
                           grp_total, top3_names, bot3_names):
    prompt = f"""
당신은 교사 AI 디지털역량 연수 전문가입니다.
아래 교사의 진단 결과를 보고 맞춤형 연수를 추천해주세요.

[진단 결과]
- 생애주기: {lable}
- 역량 유형: {result}
- 나의 전체 평균: {my_total:.2f} / 5.00점
- 집단 평균: {grp_total:.2f} / 5.00점
- 상위 역량 (강점): {', '.join(top3_names)}
- 하위 역량 (개발 필요): {', '.join(bot3_names)}

[유형 설명]
- 실천중심형: 매체활용·평가피드백 강함, AI윤리 약함
- 균형형: 전 역량 고른 수준, 특정 심화 필요
- 이해중심형: AI윤리·개인정보 강함, 실천기술 약함

[요청사항]
위 결과를 바탕으로 이 교사에게 가장 필요한 연수를 3개 추천해주세요.
각 연수마다 아래 형식으로 작성해주세요.

1. [연수명]
   - 추천 이유: (이 교사의 진단 결과와 연결해서 구체적으로)
   - 기대 효과: (수강 후 어떤 역량이 향상되는지)

2. [연수명]
   ...

3. [연수명]
   ...

한국어로 작성하고, 실제로 존재할 법한 현실적인 연수명을 사용해주세요.
"""
    response = gemini.generate_content(prompt)
    return response.text
                               
recommendations = {
    '실천중심형': [
        ('AI 디지털 윤리 기초 연수', 'https://www.neti.go.kr'),
        ('AI 활용 개인정보 보호 실천 연수', 'https://www.neti.go.kr'),
        ('AI 디지털 윤리 사례 탐구 연수', 'https://www.neti.go.kr'),
    ],
    '균형형': [
        ('AI 디지털역량 종합 심화 연수', 'https://www.neti.go.kr'),
        ('AI 기반 교수학습 설계 고급 연수', 'https://www.neti.go.kr'),
    ],
    '이해중심형': [
        ('AI 기반 교육과정 재구성 실습 연수', 'https://www.neti.go.kr'),
        ('AI 평가설계 및 기술 선정 연수', 'https://www.neti.go.kr'),
        ('AI 디지털 매체 활용 실기 연수', 'https://www.neti.go.kr'),
        ('AI 기술 진단 및 데이터 활용 연수', 'https://www.neti.go.kr'),
    ],
}

score_cols = ['score_1','score_2','score_3','score_4',
              'score_5','score_6','score_7','score_8']
comp_names = list(subcomp_items.keys())

# ============================================================
# 페이지: 로그인
# ============================================================
def page_login():
    st.title("🎓 AI 디지털역량 유형 진단")
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔐 로그인", "📝 회원가입"])

    with tab1:
        st.subheader("로그인")
        email = st.text_input("이메일", key="login_email")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", use_container_width=True, type="primary"):
            user = login(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.session_state.page = 'admin' if user['is_admin'] else 'home'
                st.rerun()
            else:
                st.error("이메일 또는 비밀번호가 올바르지 않습니다.")

    with tab2:
        st.subheader("회원가입")
        r_name    = st.text_input("이름", key="reg_name")
        r_email   = st.text_input("이메일", key="reg_email")
        r_pw      = st.text_input("비밀번호", type="password", key="reg_pw")
        r_pw2     = st.text_input("비밀번호 확인", type="password", key="reg_pw2")
        col1, col2 = st.columns(2)
        with col1:
            r_school  = st.text_input("학교명", key="reg_school")
        with col2:
            r_subject = st.text_input("담당 과목", key="reg_subject")
        r_age = st.text_input("나이", key="reg_age")

        if st.button("회원가입", use_container_width=True):
            if not all([r_name, r_email, r_pw, r_pw2]):
                st.error("이름, 이메일, 비밀번호는 필수입니다.")
            elif r_pw != r_pw2:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                result = register(r_email, r_pw, r_name, r_school, r_subject, r_age)
                if result == 'duplicate':
                    st.error("이미 사용 중인 이메일입니다.")
                elif result == 'error':
                    st.error("회원가입 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                else:
                    st.success("✅ 회원가입 완료! 로그인 탭에서 로그인해주세요.")

# ============================================================
# 페이지: 홈 (진단 + 결과)
# ============================================================
def page_home():
    user = st.session_state.user

    # 사이드바
    with st.sidebar:
        st.markdown(f"👋 **{user['name']}** 선생님")
        st.markdown(f"📧 {user['email']}")
        st.markdown("---")
        if st.button("📋 내 진단 이력", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
        if st.button("🔓 로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()

    st.title("🎓 AI 디지털역량 유형 진단")
    st.markdown("45개 문항에 응답하시면 귀하의 **AI 디지털역량 유형**과 **맞춤 연수**를 추천해드립니다.")
    st.markdown("---")

    # 경력 단계
    st.subheader("📋 기본 정보")
    lable_map = {'입직기 (경력 0~3년)': '입직기', '성장기 (경력 4~10년)': '성장기',
                 '발전기 (경력 11~20년)': '발전기', '심화기 (경력 21년 이상)': '심화기'}
    lable_select = st.selectbox("교직 경력 단계를 선택하세요", list(lable_map.keys()))
    st.markdown("---")

    # 설문 문항
    st.subheader("📝 역량 진단 문항")
    st.markdown("""
| 점수 | 의미 |
|------|------|
| 1 | 전혀 그렇지 않다 |
| 2 | 그렇지 않다 |
| 3 | 보통이다 |
| 4 | 그렇다 |
| 5 | 매우 그렇다 |
""")
    st.markdown("---")

    responses = {}
    for comp_name, items in subcomp_items.items():
        st.markdown(f"### {comp_name}")
        for code, text in items.items():
            responses[code] = st.radio(
                f"**{code}.** {text}",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True,
                key=f"q_{code}"
            )
        st.markdown("")

    st.markdown("---")

    if st.button("✅ 결과 확인하기", use_container_width=True, type="primary"):
        X = np.array([[responses[item] for item in all_items]])
        cluster = model.predict(X)[0]
        result = type_names[cluster]
        proba = model.predict_proba(X)[0]

        comp_scores = {
            name: np.mean([responses[code] for code in items])
            for name, items in subcomp_items.items()
        }

        # DB 저장
        save_result(
            user_id=user['id'],
            cluster=cluster,
            type_name=result,
            lable=lable_map[lable_select],
            comp_scores=comp_scores,
            responses=responses,
        )

        # ── 집단 평균 계산 ──────────────────────────────
        my_lable = lable_map[lable_select]
        all_res  = supabase.table('results').select('*').execute().data
        all_df   = pd.DataFrame(all_res) if all_res else pd.DataFrame()

        score_cols_list = [f'score_{i+1}' for i in range(8)]

        if not all_df.empty and all(c in all_df.columns for c in score_cols_list):
            group_df  = all_df[all_df['lable'] == my_lable]
            total_avg = all_df[score_cols_list].mean().values
            group_avg = group_df[score_cols_list].mean().values if len(group_df) > 0 else total_avg
        else:
            total_avg = np.array(list(comp_scores.values()))
            group_avg = total_avg

        my_vals   = np.array(list(comp_scores.values()))
        my_total  = float(np.mean(my_vals))
        grp_total = float(np.mean(group_avg))
        all_total = float(np.mean(total_avg))

        # ── 수준 판정 ────────────────────────────────
        def get_level(score):
            if score >= 4.5:   return '우수', '#2ecc71'
            elif score >= 3.5: return '양호', '#3498db'
            elif score >= 2.5: return '보통', '#f39c12'
            else:              return '미흡', '#e74c3c'

        level_txt, level_color = get_level(my_total)

        # ════════════════════════════════════════════
        # 결과 출력
        # ════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🎯 진단 결과")
        st.markdown(f"**{user['name']}** 선생님의 진단 결과입니다.")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("나의 유형", result)
            st.caption(f"소속 확률: {proba[cluster]*100:.1f}%")
        with col2:
            st.info(type_desc[result])

        # ── 종합 점수 카드 ───────────────────────────
        st.markdown("---")
        st.subheader("📈 종합 점수")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("나의 점수", f"{my_total:.2f}",
                      delta=f"{my_total - all_total:+.2f} (전체 평균 대비)")
        with c2:
            st.metric(f"{my_lable} 평균", f"{grp_total:.2f}")
        with c3:
            st.metric("전체 평균", f"{all_total:.2f}")

        st.markdown(f"#### 수준 판정: <span style='color:{level_color}'>{level_txt}</span>",
                    unsafe_allow_html=True)

        # 종합 점수 비교 막대
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='내 점수',       x=['내 점수'],       y=[my_total],   marker_color='#4C72B0'))
        fig_bar.add_trace(go.Bar(name=f'{my_lable} 평균', x=[f'{my_lable} 평균'], y=[grp_total], marker_color='#2ecc71'))
        fig_bar.add_trace(go.Bar(name='전체 평균',     x=['전체 평균'],     y=[all_total],  marker_color='#f39c12'))
        fig_bar.update_layout(
            title='종합 점수 비교',
            yaxis=dict(range=[1, 5.5], title='점수 (5점 만점)'),
            height=350, plot_bgcolor='white', showlegend=False,
            bargap=0.4,
        )
        fig_bar.update_traces(text=[f'{my_total:.2f}', f'{grp_total:.2f}', f'{all_total:.2f}'],
                              textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── 레이더차트 ───────────────────────────────
        st.markdown("---")
        st.subheader("🕸️ 8개 하위역량 프로파일")
        st.caption(f"파란선: 내 점수 │ 초록선: {my_lable} 평균 │ 주황선: 전체 평균")

        labels_r = list(comp_scores.keys())
        labels_r_closed = labels_r + [labels_r[0]]
        my_r    = list(my_vals)   + [my_vals[0]]
        grp_r   = list(group_avg) + [group_avg[0]]
        total_r = list(total_avg) + [total_avg[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=my_r, theta=labels_r_closed,
            fill='toself', name='내 점수',
            line=dict(color='#4C72B0', width=2),
            fillcolor='rgba(76,114,176,0.15)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=grp_r, theta=labels_r_closed,
            fill='none', name=f'{my_lable} 평균',
            line=dict(color='#2ecc71', width=2, dash='dash'),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=total_r, theta=labels_r_closed,
            fill='none', name='전체 평균',
            line=dict(color='#f39c12', width=2, dash='dot'),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
            height=500, showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── 상위 3개 / 하위 3개 역량 ─────────────────
        st.markdown("---")
        st.subheader("🏆 나의 강점 & 개발 필요 역량")

        score_series = pd.Series(comp_scores)
        top3 = score_series.sort_values(ascending=False).head(3)
        bot3 = score_series.sort_values(ascending=True).head(3)

        col_t, col_b = st.columns(2)
        with col_t:
            st.markdown("#### ✅ 상위 역량 TOP 3")
            for i, (name, val) in enumerate(top3.items(), 1):
                diff = val - float(group_avg[list(comp_scores.keys()).index(name)])
                arrow = f"▲ {diff:.2f}" if diff >= 0 else f"▼ {abs(diff):.2f}"
                color = '#2ecc71' if diff >= 0 else '#e74c3c'
                st.markdown(
                    f"**{i}. {name}**  \n"
                    f"점수: `{val:.2f}` &nbsp; "
                    f"<span style='color:{color}'>{arrow} ({my_lable} 평균 대비)</span>",
                    unsafe_allow_html=True
                )
                st.progress(min((val - 1) / 4, 1.0))

        with col_b:
            st.markdown("#### 🔧 개발 필요 역량 TOP 3")
            for i, (name, val) in enumerate(bot3.items(), 1):
                diff = val - float(group_avg[list(comp_scores.keys()).index(name)])
                arrow = f"▲ {diff:.2f}" if diff >= 0 else f"▼ {abs(diff):.2f}"
                color = '#2ecc71' if diff >= 0 else '#e74c3c'
                st.markdown(
                    f"**{i}. {name}**  \n"
                    f"점수: `{val:.2f}` &nbsp; "
                    f"<span style='color:{color}'>{arrow} ({my_lable} 평균 대비)</span>",
                    unsafe_allow_html=True
                )
                st.progress(min((val - 1) / 4, 1.0))

        # 맞춤 연수 추천
        st.markdown("---")
        st.subheader("📚 AI 맞춤 연수 추천")
        
        col_rec1, col_rec2 = st.columns([3, 1])
        with col_rec1:
            st.markdown("진단 결과를 바탕으로 AI가 맞춤 연수를 추천합니다.")
        with col_rec2:
            run_ai = st.button("🤖 AI 추천 받기", type="primary")
        
        # 기존 고정 추천은 항상 표시
        st.markdown("**📋 기본 추천 연수**")
        for rec_name, rec_url in recommendations[result]:
            st.markdown(f"- [{rec_name}]({rec_url})")
        
        # AI 추천은 버튼 눌렀을 때만
        if run_ai:
            with st.spinner("AI가 맞춤 연수를 분석 중입니다..."):
                ai_rec = get_ai_recommendation(
                    lable=my_lable,
                    result=result,
                    my_total=my_total,
                    grp_total=grp_total,
                    top3_names=top3.index.tolist(),
                    bot3_names=bot3.index.tolist(),
                )
            st.markdown("**🤖 AI 맞춤 추천 연수**")
            st.markdown(ai_rec)

# ============================================================
# 페이지: 내 진단 이력
# ============================================================
def page_history():
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"👋 **{user['name']}** 선생님")
        st.markdown("---")
        if st.button("🏠 진단하기", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        if st.button("🔓 로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()

    st.title("📋 내 진단 이력")
    st.markdown("---")

    data = get_my_results(user['id'])
    if not data:
        st.info("아직 진단 결과가 없습니다. 진단을 먼저 진행해주세요.")
        return

    df = pd.DataFrame(data)
    df['진단일시'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    df.rename(columns={
        'type_name': '유형', 'lable': '경력단계',
        **{f'score_{i+1}': comp_names[i] for i in range(8)}
    }, inplace=True)

    # 이력 요약
    st.subheader(f"총 {len(df)}회 진단")
    st.dataframe(
        df[['진단일시', '경력단계', '유형'] + comp_names].round(2),
        use_container_width=True
    )

    # 유형 변화 그래프
    if len(df) >= 2:
        st.markdown("---")
        st.subheader("📈 역량 점수 변화")
        fig = go.Figure()
        for comp in comp_names:
            fig.add_trace(go.Scatter(
                x=df['진단일시'][::-1],
                y=df[comp][::-1],
                mode='lines+markers',
                name=comp,
            ))
        fig.add_hline(y=3.5, line_dash='dash', line_color='gray',
                      annotation_text='기준선 (3.5)')
        fig.update_layout(
            yaxis=dict(range=[1, 5.5], title='점수'),
            height=420,
            plot_bgcolor='white',
        )
        st.plotly_chart(fig, use_container_width=True)

        # 최신 vs 이전 비교
        st.markdown("---")
        st.subheader("🔍 최신 vs 이전 비교")
        latest = df.iloc[0]
        prev   = df.iloc[1]

        compare = pd.DataFrame({
            '역량': comp_names,
            f'이전 ({prev["진단일시"]})': [prev[c] for c in comp_names],
            f'최신 ({latest["진단일시"]})': [latest[c] for c in comp_names],
        })
        compare['변화'] = compare.iloc[:, 2] - compare.iloc[:, 1]
        compare['변화'] = compare['변화'].apply(
            lambda x: f'▲ {x:.2f}' if x > 0 else (f'▼ {abs(x):.2f}' if x < 0 else '-')
        )
        st.dataframe(compare.round(2), use_container_width=True)

# ============================================================
# 페이지: 관리자
# ============================================================
def page_admin():
    with st.sidebar:
        st.markdown("🛠️ **관리자 모드**")
        st.markdown("---")
        if st.button("🔓 로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()

    st.title("🛠️ 관리자 대시보드")
    st.markdown("---")

    data = get_all_results()
    if not data:
        st.info("아직 진단 데이터가 없습니다.")
        return

    rows = []
    for d in data:
        user_info = d.get('users') or {}
        rows.append({
            '진단일시': pd.to_datetime(d['created_at']).strftime('%Y-%m-%d %H:%M'),
            '이름': user_info.get('name', ''),
            '이메일': user_info.get('email', ''),
            '학교': user_info.get('school', ''),
            '경력단계': d['lable'],
            '유형': d['type_name'],
            **{comp_names[i]: d[f'score_{i+1}'] for i in range(8)},
        })

    df = pd.DataFrame(rows)

    # 전체 현황
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 진단 수", len(df))
    col2.metric("참여 인원", df['이메일'].nunique())
    col3.metric("가장 많은 유형", df['유형'].value_counts().idxmax())

    st.markdown("---")

    # 유형 분포
    st.subheader("📊 유형 분포")
    type_count = df['유형'].value_counts().reset_index()
    type_count.columns = ['유형', '인원']
    fig = go.Figure(go.Pie(
        labels=type_count['유형'],
        values=type_count['인원'],
        hole=0.4,
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # 경력단계별 유형 분포
    st.subheader("📊 경력단계별 유형 분포")
    ct = pd.crosstab(df['경력단계'], df['유형'])
    st.dataframe(ct, use_container_width=True)

    st.markdown("---")

    # 전체 데이터 테이블
    st.subheader("📋 전체 진단 데이터")
    st.dataframe(df, use_container_width=True)

    # 엑셀 다운로드
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="⬇️ CSV 다운로드",
        data=csv,
        file_name=f"aiedap_results_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )

# ============================================================
# 라우터
# ============================================================
if not st.session_state.logged_in:
    page_login()
else:
    if st.session_state.page == 'home':
        page_home()
    elif st.session_state.page == 'history':
        page_history()
    elif st.session_state.page == 'admin':
        page_admin()
    else:
        page_login()
