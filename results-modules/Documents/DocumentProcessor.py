from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from document_models import GradeDocuments


class DocumentProcessor:
    """문서 처리 및 관련성 평가를 담당하는 클래스"""
    
    def __init__(self, llm):
        """
        문서 처리기 초기화
        Args:
            llm: 문서 평가에 사용할 LLM 인스턴스
        """
        # 문서 관련성 평가를 위한 구조화된 출력 LLM
        self.structured_llm_grader = llm.with_structured_output(GradeDocuments)
        
        # 문서 평가 프롬프트 템플릿 생성
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        
        # 문서 관련성 평가기 생성
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader
    
    def format_docs(self, documents: List[Dict]) -> str:
        """문서를 포맷팅하여 문자열로 변환"""
        return "\n\n".join(
            [
                f'<document><content>{doc["content"]}</content><source>{doc["metadata"]["source"]}</source></document>'
                for doc in documents
            ]
        )
    
    def filter_relevant_documents(self, documents: List[Dict], user_question: str) -> List[Dict]:
        """
        검색된 문서 중에서 사용자 질문과 관련 있는 문서만 필터링합니다.
        Args:
            documents (List[Dict]): 검색된 문서 리스트
            user_question (str): 사용자 질문
        Returns:
            List[Dict]: 관련 있는 문서만 필터링된 리스트
        """
        relevant_documents = []
        for doc in documents:
            result = self.retrieval_grader.invoke({"document": doc["content"], "question": user_question})
            if result.binary_score == "yes":
                relevant_documents.append(doc)
        
        return relevant_documents