"""다문화 학생을 위한 AI 학습 도우미 (Streamlit + Gemini)."""

from __future__ import annotations

import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions


MODEL_NAME = "gemini-3-flash-preview"
SUBJECT_OPTIONS = ["한국어", "사회", "역사", "과학"]
LEVEL_OPTIONS = ["쉬움", "보통", "어려움"]


def generate_content(
    api_key: str,
    subject: str,
    level: str,
    question: str,
    easy_mode: bool,
) -> str:
    """사용자 입력을 기반으로 Gemini 응답을 생성한다."""
    if not api_key.strip():
        raise ValueError("API 키가 비어 있습니다.")
    if not question.strip():
        raise ValueError("질문이 비어 있습니다.")

    # API 키는 세션 메모리에서만 사용하며 파일/환경 변수로 저장하지 않는다.
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    easy_prompt = (
        "학생이 이해하기 쉬운 짧은 문장, 쉬운 단어, 단계별 설명을 사용해 주세요."
        if easy_mode
        else "학생 눈높이에 맞되 핵심 개념을 정확히 설명해 주세요."
    )

    prompt = f"""
너는 한국에 거주하는 다문화 초·중등 학생의 학습을 돕는 친절한 교사야.
아래 조건을 모두 지켜 답변해 줘.

[조건]
- 과목: {subject}
- 난이도: {level}
- 설명 스타일: {easy_prompt}
- 답변 언어: 한국어
- 형식: 1) 핵심 요약 2) 쉬운 설명 3) 예시 4) 한 줄 정리

[학생 질문]
{question.strip()}
""".strip()

    response = model.generate_content(prompt)
    text = getattr(response, "text", "")
    if not text:
        raise RuntimeError("모델 응답이 비어 있습니다.")
    return text


def init_session_state() -> None:
    """필요한 세션 상태를 초기화한다."""
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""


def main() -> None:
    st.set_page_config(page_title="다문화 학생 AI 학습 도우미", page_icon="📚", layout="wide")
    init_session_state()

    st.title("📚 다문화 학생 AI 학습 도우미")
    st.caption("한국에 거주하는 다문화 초·중등 학생을 위한 맞춤형 AI 설명 서비스")

    with st.container(border=True):
        st.subheader("🔐 1) Gemini API Key 입력")
        st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_api_key",
            help="API 키는 현재 세션 메모리에만 유지되며, 서버/파일에 저장되지 않습니다.",
            placeholder="AIza...",
        )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🧭 2) 학습 설정")
            subject = st.selectbox("과목 선택", SUBJECT_OPTIONS, index=0)
            level = st.radio("난이도 선택", LEVEL_OPTIONS, horizontal=True)
            easy_mode = st.checkbox("🧒 쉬운 한국어로 설명받기", value=True)

    with col2:
        with st.container(border=True):
            st.subheader("❓ 3) 질문 입력")
            question = st.text_area(
                "궁금한 내용을 입력하세요",
                height=170,
                placeholder="예: 광합성이 무엇인지 쉬운 말로 설명해 주세요.",
            )
            ask = st.button("✨ AI에게 질문하기", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 AI 답변")

    if ask:
        api_key = st.session_state.gemini_api_key.strip()

        # 테스트 요구사항: API 키 미입력 시 실행 차단
        if not api_key:
            st.warning("먼저 Gemini API Key를 입력해 주세요.")
            st.stop()

        # 테스트 요구사항: 질문 미입력 시 실행 차단
        if not question.strip():
            st.warning("질문을 입력한 뒤 다시 시도해 주세요.")
            st.stop()

        try:
            with st.spinner("AI가 학생 맞춤형 설명을 준비하고 있어요..."):
                answer = generate_content(api_key, subject, level, question, easy_mode)
            st.success("설명이 준비되었어요!")
            st.write(answer)
        except ValueError as err:
            st.error(f"입력값 오류: {err}")
        except google_exceptions.PermissionDenied:
            # 테스트 요구사항: 잘못된 API 키 입력 시 오류 메시지 출력
            st.error("API 키가 올바르지 않거나 권한이 없습니다. 키를 다시 확인해 주세요.")
        except google_exceptions.Unauthenticated:
            st.error("인증에 실패했습니다. API 키를 확인해 주세요.")
        except google_exceptions.ResourceExhausted:
            st.error("요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
        except google_exceptions.GoogleAPICallError as err:
            st.error(f"Gemini API 호출 중 오류가 발생했습니다: {err}")
        except Exception as err:  # 예기치 못한 오류에 대한 안전망
            st.error(f"알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. ({err})")

    with st.expander("ℹ️ 사용 안내", expanded=False):
        st.markdown(
            """
- API 키는 **현재 브라우저 세션 메모리에서만** 사용됩니다.
- 과목/난이도를 바꿔가며 같은 질문을 비교해 보세요.
- 쉬운 한국어 옵션을 켜면 더 단순한 어휘와 짧은 문장으로 설명합니다.
            """
        )


if __name__ == "__main__":
    main()
