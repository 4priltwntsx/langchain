from typing import Dict, List

from langchain_openai import ChatOpenAI

from your_vector_db_client import VectorDBClient
from document_processor import DocumentProcessor
from hallucination_checker import HallucinationChecker
from response_generator import ResponseGenerator


class HallucinationDetector:
    """할루시네이션 검출 및 RAG 응답 생성을 위한 통합 클래스"""
    
    def __init__(self, model_name="gpt-4", temperature=0):
        """
        할루시네이션 검출기 초기화
        Args:
            model_name (str): 사용할 LLM 모델 이름
            temperature (float): LLM 응답 다양성 파라미터
        """
        # LLM 초기화
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # 벡터 데이터베이스 초기화
        self.vector_db = VectorDBClient()
        
        # 구성 요소 초기화
        self.document_processor = DocumentProcessor(self.llm)
        self.hallucination_checker = HallucinationChecker(self.llm, self.document_processor)
        self.response_generator = ResponseGenerator(self.llm, self.document_processor)
    
    def process_query(self, user_question: str, top_k: int = 5):
        """
        사용자 질문을 처리하여 응답을 생성하고 할루시네이션을 검출합니다.
        Args:
            user_question (str): 사용자 질문.
            top_k (int): 검색할 문서 수.
        Returns:
            tuple: (생성된 응답, 할루시네이션 검출 결과, 관련 문서 수)
        """
        # 벡터 데이터베이스에서 문서 검색
        retrieved_documents = self.vector_db.search(query=user_question, top_k=top_k)
        
        # 관련 문서 필터링
        relevant_documents = self.document_processor.filter_relevant_documents(retrieved_documents, user_question)
        
        # RAG 응답 생성
        response = self.response_generator.generate_response_with_cache(user_question, relevant_documents)
        
        # 할루시네이션 검출 실행
        hallucination_result = self.hallucination_checker.detect_hallucination(response, relevant_documents)
        
        return response, hallucination_result, len(relevant_documents)