from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents.
    이 클래스는 결국 검색된 문서들이 주어진 질문에 대해 관련성이 있는지를 이진 형식으로 판단하는 데 사용될 수 있습니다. 객체를 생성할 때, binary_score 필드에 대해 "yes" 또는 "no"의 문자열 값을 제공하여 관련성을 표시하게 됩니다.
    """
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )