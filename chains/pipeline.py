"""LCEL: context → diagnosis → rewrite 단일 파이프라인."""

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from chains.context_chain import build_context_chain, prep_context_input
from chains.diagnosis_chain import build_diagnosis_chain, prep_diagnosis_input
from chains.rewrite_chain import build_rewrite_chain, prep_rewrite_input


def build_chain_segments(llm: ChatOpenAI):
    """세 단계 Runnable (UI 단계별 스피너용).

    입력 키: purpose, output_format, improvement_goals, user_prompt
    context 단계 출력: context_profile dict
    diagnosis 단계: 이전 출력 + 동일 입력에서 diagnosis dict
    rewrite 단계: diagnosis까지 merge된 dict에서 rewrite dict
    """
    context_runnable = RunnableLambda(prep_context_input) | build_context_chain(llm)
    diagnosis_runnable = RunnableLambda(prep_diagnosis_input) | build_diagnosis_chain(llm)
    rewrite_runnable = RunnableLambda(prep_rewrite_input) | build_rewrite_chain(llm)
    return context_runnable, diagnosis_runnable, rewrite_runnable


def build_prompt_clinic_pipeline(llm: ChatOpenAI):
    """LCEL: context_prompt|llm|parser | diagnosis|llm|parser | rewrite|llm|parser.

    입력 키: purpose, output_format, improvement_goals, user_prompt
    출력: 위 키 + context_profile, diagnosis, rewrite
    """
    context_runnable, diagnosis_runnable, rewrite_runnable = build_chain_segments(llm)
    return (
        RunnablePassthrough.assign(context_profile=context_runnable)
        | RunnablePassthrough.assign(diagnosis=diagnosis_runnable)
        | RunnablePassthrough.assign(rewrite=rewrite_runnable)
    )
