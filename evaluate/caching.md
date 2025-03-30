결과 캐싱 프로그램은 RAG 워크플로우에서 생성된 응답을 저장하고, 동일한 질문에 대해 반복적으로 요청이 들어올 경우 캐시된 결과를 반환하여 성능을 최적화하는 데 사용됩니다. 이를 통해 불필요한 계산을 줄이고 응답 속도를 개선할 수 있습니다.

---

### 결과 캐싱 프로그램 설계

#### 1. **캐싱의 기본 원리**

- **입력(질문)**: 사용자 질문을 키(key)로 사용.
- **출력(응답)**: RAG 체인에서 생성된 응답을 값(value)으로 저장.
- 동일한 질문이 들어오면 캐시에서 값을 반환.
- 새로운 질문이 들어오면 RAG 체인을 실행하고 결과를 캐시에 저장.

---

### 2. **결과 캐싱 프로그램 코드**

```python
import hashlib
from typing import Dict

# 간단한 메모리 기반 캐시
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

# 결과 캐싱을 포함한 RAG 응답 생성 함수
def generate_response_with_cache(question: str, documents):
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
    response = rag_chain.invoke({
        "documents": formatted_docs,
        "question": question,
    })

    # 결과를 캐시에 저장
    result_cache.set(question, response)
    return response
```

---

### 3. **사용 예제**

```python
# 사용자 질문
user_question = "What is the capital of France?"

# 벡터 데이터베이스에서 문서 검색 (예시)
retrieved_documents = vector_db.search(query=user_question, top_k=5)

# 캐싱을 활용한 RAG 응답 생성
response = generate_response_with_cache(user_question, retrieved_documents)

# 결과 출력
print("Generated Response:", response)
```

---

### 4. **결과 캐싱 프로그램의 주요 구성 요소**

#### (1) **`ResultCache` 클래스**

- **`_hash_key`**: 질문을 해시하여 고유 키를 생성합니다. 이를 통해 동일한 질문에 대해 항상 동일한 키를 사용합니다.
- **`get`**: 캐시에서 결과를 가져옵니다.
- **`set`**: 캐시에 결과를 저장합니다.
- **`clear`**: 캐시를 초기화합니다.

#### (2) **`generate_response_with_cache` 함수**

- 캐시에서 결과를 확인하고, 없으면 RAG 체인을 실행하여 응답을 생성합니다.
- 생성된 응답은 캐시에 저장하여 이후 동일한 질문에 대해 재사용합니다.

---

### 5. **확장 가능성**

#### (1) **파일 기반 캐싱**

- 메모리 기반 캐시는 프로그램 종료 시 데이터가 사라지므로, 파일 기반 캐싱(예: SQLite, Redis)을 사용하여 지속성을 추가할 수 있습니다.

```python
import shelve

class PersistentResultCache:
    def __init__(self, filename="cache.db"):
        self.db = shelve.open(filename)

    def get(self, question: str) -> str:
        key = hashlib.sha256(question.encode()).hexdigest()
        return self.db.get(key)

    def set(self, question: str, response: str):
        key = hashlib.sha256(question.encode()).hexdigest()
        self.db[key] = response

    def close(self):
        self.db.close()
```

#### (2) **TTL(Time-to-Live) 추가**

- 캐시된 결과에 유효 기간을 설정하여 오래된 데이터를 자동으로 제거할 수 있습니다.

#### (3) **캐시 적중률 모니터링**

- 캐시 적중률(캐시에서 결과를 반환한 비율)을 모니터링하여 캐시의 효율성을 평가할 수 있습니다.

---

### 6. **결과 캐싱의 이점**

1. **성능 향상**:
   - 동일한 질문에 대해 RAG 체인을 반복적으로 실행하지 않아도 되므로 응답 속도가 빨라집니다.
2. **비용 절감**:
   - LLM 호출 횟수를 줄여 비용을 절감할 수 있습니다.
3. **사용자 경험 개선**:
   - 빠른 응답을 통해 사용자 경험을 향상시킬 수 있습니다.

---

### 결론

결과 캐싱 프로그램은 RAG 워크플로우에서 성능 최적화와 비용 절감을 위한 필수적인 구성 요소입니다. 위 코드를 기반으로 메모리 기반 캐싱을 구현하고, 필요에 따라 파일 기반 캐싱이나 TTL 기능을 추가하여 확장할 수 있습니다.
