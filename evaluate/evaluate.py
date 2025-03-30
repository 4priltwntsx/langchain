from your_vector_db_client import VectorDBClient
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 문서 평가를 위한 데이터 모델 정의
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM 초기화 및 구조화된 출력 생성
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 문서 평가 프롬프트 템플릿 생성
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 문서 검색결과 평가기 생성
retrieval_grader = grade_prompt | structured_llm_grader

# 벡터 데이터베이스 초기화
vector_db = VectorDBClient()

# 사용자 질문
user_question = "What is the capital of France?"

# 벡터 데이터베이스에서 문서 검색
retrieved_documents = vector_db.search(query=user_question, top_k=5)

# 관련 문서 필터링
relevant_documents = []
for doc in retrieved_documents:
    result = retrieval_grader.invoke({"document": doc["content"], "question": user_question})
    if result["binary_score"] == "yes":
        relevant_documents.append(doc)

# 응답 생성
if relevant_documents:
    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant generating answers based on relevant documents."),
            ("human", "Relevant documents: \n\n {documents} \n\n User question: {question}"),
        ]
    )
    response_chain = response_prompt | llm
    final_response = response_chain.invoke({
        "documents": "\n".join([doc["content"] for doc in relevant_documents]),
        "question": user_question,
    })
    print("Generated Response:", final_response)
else:
    print("No relevant documents found.")