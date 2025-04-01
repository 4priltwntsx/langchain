import hashlib
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from your_vector_db_client import VectorDBClient

# 캐시 클래스 정의
class ResultCache:
    def __init__(self):
        self.cache: Dict[str, str] = {}

    def _hash_key(self, question: str) -> str:
        """질문을 해시하여 고유 키 생성"""
        return hashlib.sha256(question.encode()).hexdigest()

    def get(self, question: str) -> str:
        """캐시에서 결과 가져오기"""
        key = self._hash_key(question)
        return self.cache.get(key)

    def set(self, question: str, response: str):
        """캐시에 결과 저장"""
        key = self._hash_key(question)
        self.cache[key] = response

    def clear(self):
        """캐시 초기화"""
        self.cache.clear()

# 캐시 인스턴스 생성
result_cache = ResultCache()

# LLM 초기화
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 할루시네이션 검출 프롬프트 템플릿
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact-checker. Verify if the following response aligns with the provided documents."),
        ("human", "Response: \n\n {response} \n\n Relevant documents: \n\n {documents}"),
    ]
)
hallucination_chain = hallucination_prompt | llm | StrOutputParser()

# 벡터 데이터베이스 초기화
vector_db = VectorDBClient()

# 문서 포맷팅 함수
def format_docs(documents: List[Dict]) -> str:
    """문서를 포맷팅하여 문자열로 변환"""
    return "\n\n".join(
        [
            f'<document><content>{doc["content"]}</content><source>{doc["metadata"]["source"]}</source></document>'
            for doc in documents
        ]
    )

# RAG 응답 생성 함수
def generate_response_with_cache(question: str, documents: List[Dict]) -> str:
    """
    캐시를 활용하여 RAG 응답을 생성합니다.

    Args:
        question (str): 사용자 질문.
        documents (list): 검색된 문서 리스트.

    Returns:
        str: 생성된 응답.
    """
    # 캐시에서 결과 확인
    cached_response = result_cache.get(question)
    if cached_response:
        print("Returning cached response.")
        return cached_response

    # 캐시에 결과가 없으면 RAG 체인 실행
    formatted_docs = format_docs(documents)
    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant generating answers based on relevant documents."),
            ("human", "Relevant documents: \n\n {documents} \n\n User question: {question}"),
        ]
    )
    response_chain = response_prompt | llm
    response = response_chain.invoke({
        "documents": formatted_docs,
        "question": question,
    })

    # 결과를 캐시에 저장
    result_cache.set(question, response)
    return response

# 할루시네이션 검출 함수
def detect_hallucination(response: str, relevant_documents: List[Dict]) -> dict:
    """
    LLM 응답이 관련 문서와 일치하는지 검증합니다.

    Args:
        response (str): LLM이 생성한 응답.
        relevant_documents (list): 관련 문서 리스트.

    Returns:
        dict: 검증 결과와 신뢰도 점수.
    """
    formatted_docs = format_docs(relevant_documents)
    result = hallucination_chain.invoke({
        "response": response,
        "documents": formatted_docs,
    })
    return result

# 메인 워크플로우
def main():
    # 사용자 질문
    user_question = "What is the capital of France?"

    # 벡터 데이터베이스에서 문서 검색
    retrieved_documents = vector_db.search(query=user_question, top_k=5)

    # 관련 문서 필터링 (간단한 예제에서는 모든 문서를 사용)
    relevant_documents = retrieved_documents

    # RAG 응답 생성
    response = generate_response_with_cache(user_question, relevant_documents)

    # 할루시네이션 검출 실행
    hallucination_result = detect_hallucination(response, relevant_documents)

    # 결과 출력
    print("Generated Response:", response)
    print("Hallucination Detection Result:", hallucination_result)

if __name__ == "__main__":
    main()