from hallucination_detector import HallucinationDetector


def main():
    """메인 실행 함수"""
    # 할루시네이션 검출기 인스턴스 생성
    detector = HallucinationDetector(model_name="gpt-4", temperature=0)
    
    # 사용자 질문
    user_question = "What is the ?"
    
    # 질문 처리 및 결과 출력
    response, hallucination_result, relevant_doc_count = detector.process_query(user_question)
    
    print(f"Found {relevant_doc_count} relevant documents.")
    print("Generated Response:", response)
    print("Hallucination Detection Result:", hallucination_result)


if __name__ == "__main__":
    main()