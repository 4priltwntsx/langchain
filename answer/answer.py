# 사용자 질문
user_question = "What is the capital of France?"

# 벡터 데이터베이스에서 문서 검색
retrieved_documents = vector_db.search(query=user_question, top_k=5)

# RAG 응답 생성
response = rag_chain.invoke({
    "documents": format_docs(retrieved_documents),
    "question": user_question,
})

# 할루시네이션 검출 실행
result = detect_hallucination(response, retrieved_documents)

# 결과 출력
print("Generated Response:", response)
print("Hallucination Detection Result:", result)