from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

from result_cache import ResultCache


class ResponseGenerator:
    """RAG 응답 생성을 담당하는 클래스"""
    
    def __init__(self, llm, document_processor):
        """
        응답 생성기 초기화
        Args:
            llm: 응답 생성에 사용할 LLM 인스턴스
            document_processor: 문서 포맷팅 등 문서 처리를 담당하는 인스턴스
        """
        self.llm = llm
        self.document_processor = document_processor
        self.result_cache = ResultCache()
        
        # RAG 응답 생성 프롬프트
        self.response_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant generating answers based on relevant documents."),
                ("human", "Relevant documents: \n\n {documents} \n\n User question: {question}"),
            ]
        )
        self.response_chain = self.response_prompt | self.llm
    
    def generate_response_with_cache(self, question: str, documents: List[Dict]) -> str:
        """
        캐시를 활용하여 RAG 응답을 생성합니다.
        Args:
            question (str): 사용자 질문.
            documents (list): 검색된 문서 리스트.
        Returns:
            str: 생성된 응답 또는 관련 문서가 없을 경우 메시지.
        """
        # 캐시에서 결과 확인
        cached_response = self.result_cache.get(question)
        if cached_response:
            print("Returning cached response.")
            return cached_response
        
        # 문서가 비어있는 경우 처리
        if not documents:
            return "No relevant documents found to answer your question."
        
        # 캐시에 결과가 없으면 RAG 체인 실행
        formatted_docs = self.document_processor.format_docs(documents)
        response = self.response_chain.invoke({
            "documents": formatted_docs,
            "question": question,
        })
        
        # 결과를 캐시에 저장
        self.result_cache.set(question, response)
        return response