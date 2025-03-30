from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from your_vector_db_client import VectorDBClient  # 벡터 DB 클라이언트 임포트

# LLM 초기화
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

# 할루시네이션 검출 프롬프트 템플릿
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact-checker. Verify if the following response aligns with the provided documents."),
        ("human", "Response: \n\n {response} \n\n Relevant documents: \n\n {documents}"),
    ]
)

# 할루시네이션 검출 체인 생성
hallucination_chain = hallucination_prompt | llm | StrOutputParser()

# 할루시네이션 검출 함수
def detect_hallucination(response, relevant_documents):
    """
    LLM 응답이 관련 문서와 일치하는지 검증합니다.
    
    Args:
        response (str): LLM이 생성한 응답.
        relevant_documents (list): 관련 문서 리스트.
    
    Returns:
        dict: 검증 결과와 신뢰도 점수.
    """
    # 문서 포맷팅
    formatted_docs = "\n\n".join(
        [
            f'<document><content>{doc["content"]}</content><source>{doc["metadata"]["source"]}</source></document>'
            for doc in relevant_documents
        ]
    )
    
    # 할루시네이션 검출 실행
    result = hallucination_chain.invoke({
        "response": response,
        "documents": formatted_docs,
    })
    
    return result