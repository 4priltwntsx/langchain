from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class HallucinationChecker:
    """할루시네이션 검출을 담당하는 클래스"""
    
    def __init__(self, llm, document_processor):
        """
        할루시네이션 검출기 초기화
        Args:
            llm: 할루시네이션 검출에 사용할 LLM 인스턴스
            document_processor: 문서 포맷팅 등 문서 처리를 담당하는 인스턴스
        """
        self.llm = llm
        self.document_processor = document_processor
        
        # 할루시네이션 검출 프롬프트 템플릿
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a fact-checker. Verify if the following response aligns with the provided documents."),
                ("human", "Response: \n\n {response} \n\n Relevant documents: \n\n {documents}"),
            ]
        )
        self.hallucination_chain = self.hallucination_prompt | self.llm | StrOutputParser()
    
    def detect_hallucination(self, response: str, relevant_documents: List[Dict]) -> dict:
        """
        LLM 응답이 관련 문서와 일치하는지 검증합니다.
        Args:
            response (str): LLM이 생성한 응답.
            relevant_documents (list): 관련 문서 리스트.
        Returns:
            dict: 검증 결과와 신뢰도 점수.
        """
        # 문서가 비어있는 경우 처리
        if not relevant_documents:
            return {"result": "Unable to verify - no relevant documents found"}
        
        formatted_docs = self.document_processor.format_docs(relevant_documents)
        result = self.hallucination_chain.invoke({
            "response": response,
            "documents": formatted_docs,
        })
        return result