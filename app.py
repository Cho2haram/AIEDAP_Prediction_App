import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(
    page_title="AI 디지털역량 유형 진단",
    page_icon="🤖",
    layout="centered"
)

# ============================================================
# 모델 로드
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('rf_model_k3.pkl')

model = load_model()

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

# 연수 추천 목록 (이름, URL) - 실제 URL로 교체하세요
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

# ============================================================
# 화면 구성
# ============================================================
st.title("🤖 AI 디지털역량 유형 진단")
st.markdown("45개 문항에 응답하시면 귀하의 **AI 디지털역량 유형**과 **맞춤 연수**를 추천해드립니다.")
st.markdown("---")

# 기본 정보
st.subheader("📋 기본 정보")
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("이름")
    school = st.text_input("학교명")
with col2:
    age = st.text_input("나이")
    subject = st.text_input("담당 과목")

lable_map = {'입직기 (경력 0~3년)': 0, '성장기 (경력 4~10년)': 1,
             '발전기 (경력 11~20년)': 2, '심화기 (경력 21년 이상)': 3}
lable_select = st.selectbox("교직 경력 단계를 선택하세요", list(lable_map.keys()))
st.markdown("---")

# 설문 문항
st.subheader("📝 역량 진단 문항")
st.markdown("각 문항을 읽고 본인의 수준에 해당하는 점수를 선택해주세요.")
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
            key=code
        )
    st.markdown("")

st.markdown("---")

# ============================================================
# 결과 예측
# ============================================================
if st.button("✅ 결과 확인하기", use_container_width=True, type="primary"):

    X = np.array([[responses[item] for item in all_items]])
    cluster = model.predict(X)[0]
    result = type_names[cluster]
    proba = model.predict_proba(X)[0]

    st.markdown("---")
    st.subheader("🎯 진단 결과")

    if name:
        st.markdown(f"**{name}** 선생님의 진단 결과입니다.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("나의 유형", result)
        st.caption(f"소속 확률: {proba[cluster]*100:.1f}%")
    with col2:
        st.info(type_desc[result])

    st.markdown("---")

    # 역량 프로파일 시각화 (Plotly)
    st.subheader("📊 나의 역량 프로파일")

    comp_scores = {}
    for comp_name, items in subcomp_items.items():
        comp_scores[comp_name] = np.mean([responses[code] for code in items])

    labels = list(comp_scores.keys())
    values = list(comp_scores.values())
    colors = ['#e74c3c' if v < 3.5 else '#2ecc71' for v in values]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.2f}' for v in values],
        textposition='outside',
    ))
    fig.add_hline(
        y=3.5,
        line_dash='dash',
        line_color='gray',
        annotation_text='기준선 (3.5)',
        annotation_position='top right',
    )
    fig.update_layout(
        title='하위역량별 점수 프로파일',
        yaxis=dict(range=[1, 5.8], title='점수'),
        xaxis=dict(title=''),
        height=420,
        font=dict(family='Arial Unicode MS, sans-serif', size=12),
        plot_bgcolor='white',
        margin=dict(b=120),
    )
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 추천 연수 (링크 포함)
    st.subheader("📚 맞춤 연수 추천")
    st.markdown(f"**{result}** 에게 추천하는 연수 목록입니다.")
    for rec_name, rec_url in recommendations[result]:
        st.markdown(f"- [{rec_name}]({rec_url})")

    st.markdown("---")

    # 상세 점수 테이블
    with st.expander("📋 문항별 상세 점수 보기"):
        detail = []
        for comp_name, items in subcomp_items.items():
            for code, text in items.items():
                detail.append({
                    '역량': comp_name,
                    '문항코드': code,
                    '문항내용': text[:40] + '...' if len(text) > 40 else text,
                    '점수': responses[code],
                })
        st.dataframe(pd.DataFrame(detail), use_container_width=True)
