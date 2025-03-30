벡터 데이터베이스(Vector Database)를 활용하면, 문서 검색 단계에서 사용자 질문과 가장 관련성이 높은 문서를 검색한 후, 이를 평가하고 응답을 생성할 수 있습니다. 아래는 벡터 데이터베이스를 활용한 RAG 응답 생성 로직을 구현하는 방법입니다.

---

### 1. **벡터 데이터베이스에서 문서 검색**

벡터 데이터베이스(예: Pinecone, Weaviate, FAISS 등)를 사용하여 사용자 질문과 가장 유사한 문서를 검색합니다.

```python
from your_vector_db_client import VectorDBClient  # 벡터 DB 클라이언트 임포트

# 벡터 데이터베이스 초기화
vector_db = VectorDBClient()

# 사용자 질문
user_question = "What is the capital of France?"

# 벡터 데이터베이스에서 문서 검색
retrieved_documents = vector_db.search(query=user_question, top_k=5)  # top_k는 검색할 문서 수
```

`retrieved_documents`는 다음과 같은 형식으로 반환된다고 가정합니다:

```python
retrieved_documents = [
    {"content": "Paris is the capital of France.", "metadata": {"source": "Doc1"}},
    {"content": "Berlin is the capital of Germany.", "metadata": {"source": "Doc2"}},
]
```

---

### 2. **문서 평가**

검색된 문서를 `retrieval_grader`를 사용하여 평가합니다.

```python
# 관련 문서 필터링
relevant_documents = []
for doc in retrieved_documents:
    result = retrieval_grader.invoke({"document": doc["content"], "question": user_question})
    if result["binary_score"] == "yes":
        relevant_documents.append(doc)

# 필터링된 문서 출력
print("Relevant Documents:", relevant_documents)
```

---

### 3. **응답 생성**

평가된 문서를 기반으로 사용자 질문에 대한 최종 응답을 생성합니다.

```python
if relevant_documents:
    # 응답 생성용 프롬프트 템플릿
    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant generating answers based on relevant documents."),
            ("human", "Relevant documents: \n\n {documents} \n\n User question: {question}"),
        ]
    )

    # LLM 호출
    response_chain = response_prompt | llm
    final_response = response_chain.invoke({
        "documents": "\n".join([doc["content"] for doc in relevant_documents]),
        "question": user_question,
    })
    print("Generated Response:", final_response)
else:
    print("No relevant documents found.")
```

---

### 4. **전체 코드**

아래는 벡터 데이터베이스를 활용한 전체 RAG 응답 생성 로직입니다:

```python
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
```

---

### 주요 포인트

1. **벡터 데이터베이스 검색**:

   - 사용자 질문과 유사한 문서를 검색합니다.
   - `top_k`를 조정하여 검색할 문서 수를 설정합니다.

2. **문서 평가**:

   - `retrieval_grader`를 사용하여 검색된 문서의 관련성을 평가합니다.
   - 관련 문서만 필터링합니다.

3. **응답 생성**:
   - 평가된 문서를 기반으로 사용자 질문에 대한 최종 응답을 생성합니다.

---

### 결론

이 코드는 벡터 데이터베이스를 활용하여 검색된 문서를 평가하고, 관련 문서를 기반으로 응답을 생성하는 RAG 워크플로우를 구현합니다. 이를 통해 효율적이고 정확한 응답 생성을 수행할 수 있습니다.
