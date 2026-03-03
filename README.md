> [!NOTE]
> 본 문서는 Claude로 작성되었습니다.

# EcoVector Android

온디바이스 하이브리드 벡터 검색 SDK for Android.

벡터 유사도 검색(FAISS + HNSW)과 전문 검색(BM25 + Kiwi 형태소 분석)을 결합하여,
클라우드 없이 기기 내에서 의미론적 + 키워드 기반 하이브리드 문서 검색을 수행합니다.

| | |
|---|---|
| **Group** | `io.graphai` |
| **Artifact** | `ecovector` |
| **Version** | `0.1.0` |
| **Min SDK** | 24 (Android 7.0) |
| **ABI** | `arm64-v8a` |

## 주요 특징

- **하이브리드 검색** — Vector, BM25, Ensemble(RRF/RSF) 세 가지 검색 전략
- **온디바이스** — 모든 처리가 기기 내에서 완결, 네트워크 불필요
- **한국어 최적화** — Kiwi 형태소 분석기로 한국어 토큰화
- **네이티브 성능** — C++17 코어 + Kotlin API (JNI)
- **토큰 인식 청킹** — 모델 토큰 경계를 존중하는 문서 분할
- **PDF 지원** — PdfBox-Android 기반 텍스트 추출

## 기술 스택

### 네이티브 (C++17)

| 용도 | 라이브러리 |
|---|---|
| K-means 클러스터링 + 벡터 인덱스 | **FAISS** |
| 근사 최근접 이웃 탐색 | **HNSW** |
| 임베딩 모델 추론 (KoSimCSE-BERT QInt8) | **ONNX Runtime** |
| 한국어 형태소 분석 | **Kiwi** |
| 청크/임베딩 저장 | **ObjectBox** |
| 보조 데이터 관리 | **SQLite** |
| HuggingFace 토크나이저 | **tokenizers-cpp** |

### Kotlin

| 용도 | 라이브러리 |
|---|---|
| 비동기 문서 처리 | **Coroutines** |
| PDF 텍스트 추출 | **PdfBox-Android** |

### 빌드

- Gradle 8.x + Kotlin DSL
- CMake 3.22.1 + NDK (arm64-v8a)

## 모듈 구조

```
ecovector-android/
├── ecovector/                  # 메인 라이브러리 (io.graphai:ecovector)
│   ├── src/main/java/          #   Kotlin API
│   └── src/main/cpp/           #   C++ 네이티브 코어
│       ├── jni/                #     JNI 바인딩
│       ├── src/                #     핵심 구현
│       │   ├── retriever/      #       검색기 (Vector, BM25, Ensemble)
│       │   ├── eco_vector/     #       FAISS 벡터 인덱스
│       │   ├── bm25/           #       BM25 전문 검색
│       │   ├── chunker/        #       TokenAwareChunker
│       │   ├── embedder/       #       ONNX 임베딩
│       │   ├── tokenizer/      #       토크나이저
│       │   ├── kiwi/           #       한국어 형태소 분석
│       │   └── object_box/     #       DB 레이어
│       └── third_party/        #     프리빌트 네이티브 라이브러리
└── ecovector-benchmark/        # 벤치마크 앱 (Jetpack Compose UI)
```

## 검색 파이프라인

### 인덱싱

```
문서 추가 → TextCleaner → TokenAwareChunker → ONNX 임베딩 + Kiwi 토큰화 → ObjectBox 저장
                                                         ↓
                                              buildIndex() → FAISS 클러스터링 + HNSW 구축
                                                           → BM25 역인덱스 구축
```

### 검색

```
쿼리 → 임베딩/토큰화 → Retriever.retrieve()
                        ├── VectorRetriever  : FAISS nprobe → HNSW efSearch → top-K
                        ├── BM25Retriever    : Kiwi 토큰 → IDF 시드 → BM25 스코어링
                        └── EnsembleRetriever: 위 결과를 RRF 또는 RSF로 퓨전
```

## Quick Start

```kotlin
// 1. 초기화
val eco = EcoVectorStore.create(context)

// 2. 문서 추가
eco.addDocument("문서 내용...", "제목")
eco.addDocuments(
    listOf("내용1" to "제목1", "내용2" to "제목2"),
    ChunkParams(maxTokens = 216, overlapTokens = 128)
)

// 3. 인덱스 빌드
eco.buildIndex()

// 4. 검색기 생성
val vector = eco.createVectorRetriever(
    VectorRetrieverParams(efSearch = 64, nprobe = 4, topK = 50)
)
val bm25 = eco.createBM25Retriever(
    BM25RetrieverParams(k1 = 0.9f, topK = 50)
)
val ensemble = eco.createEnsembleRetriever(
    components = listOf(vector weighted 0.7f, bm25 weighted 0.3f),
    topK = 10
)

// 5. 검색
val results: List<SearchResult> = ensemble.retrieve("검색 쿼리")
results.forEach { println("${it.score}: ${it.content}") }

// 6. 정리
eco.close()
```

### PDF 문서 추가

```kotlin
val pdfSource = PdfDocument(
    inputStreamProvider = { context.assets.open("sample.pdf") },
    title = "샘플 PDF"
)
eco.addDocument(pdfSource)  // suspend 함수
```

## API 레퍼런스

### EcoVectorStore

| 메서드 | 설명 |
|---|---|
| `create(context, config?)` | 인스턴스 생성 및 초기화 (싱글톤) |
| `getInstance()` | 기존 인스턴스 반환 |
| `addDocument(content, title?, chunkParams?)` | 단일 문서 추가 |
| `addDocument(source)` | DocumentSource로 문서 추가 (suspend) |
| `addDocuments(documents, chunkParams?, progressCallback?)` | 배치 문서 추가 |
| `buildIndex(centroidCount?)` | Vector + BM25 인덱스 빌드 |
| `buildVectorIndex(centroidCount?)` | Vector 인덱스만 빌드 |
| `buildVectorIndex(params)` | 커스텀 파라미터로 Vector 인덱스 빌드 |
| `buildBM25Index()` | BM25 인덱스만 빌드 |
| `isIndexReady()` | 인덱스 빌드 완료 여부 |
| `removeDocument(id)` | 문서 및 청크 삭제 |
| `removeChunk(id)` | 청크 삭제 |
| `removeAll()` | 전체 삭제 |
| `tokenize(text)` | HuggingFace 토큰 ID 반환 |
| `embed(text)` | 임베딩 벡터 반환 |
| `tokenizeKiwi(text)` | Kiwi 형태소 해시 반환 |
| `documentCount` / `chunkCount` | 저장된 문서/청크 수 |
| `close()` | 네이티브 리소스 해제 |

### Retriever

모든 검색기는 `Retriever` 인터페이스를 구현합니다.

```kotlin
interface Retriever : AutoCloseable {
    fun retrieve(query: String): List<SearchResult>
    fun retrieve(bundle: QueryBundle): List<SearchResult>
    val name: String
    val isReady: Boolean
}
```

| 검색기 | 설명 | 핵심 파라미터 |
|---|---|---|
| `VectorRetriever` | FAISS + HNSW 벡터 유사도 검색 | `efSearch`, `nprobe`, `topK` |
| `BM25Retriever` | Kiwi 기반 전문 검색 | `k1`, `b`, `topK`, RM3 확장 |
| `EnsembleRetriever` | 다중 검색기 결과 퓨전 | `fusionMethod`, `rrfK`, `topK` |

### 설정 클래스

#### EcoVectorConfig

```kotlin
EcoVectorConfig(
    modelAssetDir = "models/KoSimCSE-bert-QInt8",  // 임베딩 모델 경로
    dbName = "ecovector-db",                        // ObjectBox DB 이름
    kiwiModelAssetDir = "kiwi_model"                // Kiwi 모델 경로
)
```

#### ChunkParams

```kotlin
ChunkParams(
    maxTokens = 216,      // 청크당 최대 토큰 수
    overlapTokens = 128   // 청크 간 오버랩 토큰 수
)
```

#### VectorRetrieverParams

```kotlin
VectorRetrieverParams(
    efSearch = 20,   // HNSW 탐색 범위 (↑ = 정확도↑, 속도↓)
    nprobe = 4,      // FAISS 클러스터 탐색 수 (↑ = 정확도↑, 속도↓)
    topK = 11        // 반환 결과 수
)
```

#### BM25RetrieverParams

```kotlin
BM25RetrieverParams(
    k1 = 0.9f,                // TF 포화 계수
    b = 0.25f,                // 문서 길이 정규화
    topK = 15,                // 반환 결과 수
    idfThreshold = 0.6f,      // 시드 토큰 IDF 필터 비율
    maxSeedTerms = 5,         // 최대 시드 토큰 수
    candidateMultiplier = 10, // 후보 = topK × multiplier
    minCandidates = 50,       // 최소 전체 스코어링 후보 수
    minScore = 0.0f,          // BM25 점수 하한
    // RM3 쿼리 확장 (기본 비활성)
    rm3Enabled = false,
    rm3FbDocs = 10,           // 피드백 문서 수
    rm3FbTerms = 20,          // 확장 용어 수
    rm3OrigWeight = 0.6f,     // 원본 쿼리 가중치 (λ)
    rm3MinDf = 2              // 최소 문서 빈도 필터
)
```

#### EcoVectorIndexParams

```kotlin
EcoVectorIndexParams(
    nCluster = 0,           // K-means 클러스터 수 (0 = auto)
    hnswM = 16,             // HNSW 노드당 연결 수
    efConstruction = 100,   // HNSW 빌드 품질
    maxTrainSamples = 0     // FAISS 학습 샘플 수 (0 = auto)
)
```

#### FusionMethod

| 값 | 설명 |
|---|---|
| `RRF` | Reciprocal Rank Fusion — 순위 기반, 점수 분포 무관하게 안정적 |
| `RSF` | Relative Score Fusion — 점수 min-max 정규화 후 가중합 |

### 데이터 클래스

```kotlin
data class SearchResult(
    val documentId: Long,   // 원본 문서 ID
    val chunkId: Long,      // 매칭된 청크 ID
    val content: String,    // 청크 텍스트
    val score: Float        // 관련성 점수 (높을수록 관련성↑)
)

data class QueryBundle(
    val rawText: String? = null,        // 원본 쿼리
    val embedding: FloatArray? = null,  // 사전 계산된 임베딩
    val kiwiTokens: IntArray? = null    // 사전 계산된 Kiwi 토큰
)
```

## 빌드

### 필요 환경

- Android Studio Ladybug 이상
- NDK (CMake 3.22.1 포함)
- JDK 11+

### 빌드 명령

```bash
# 라이브러리 빌드
./gradlew :ecovector:assembleRelease

# AAR 생성 경로
# ecovector/build/outputs/aar/ecovector-release.aar

# 벤치마크 앱 빌드
./gradlew :ecovector-benchmark:assembleDebug
```

### Maven 로컬 퍼블리시

```bash
./gradlew :ecovector:publishReleasePublicationToMavenLocal
```

사용 측 `build.gradle.kts`:

```kotlin
dependencies {
    implementation("io.graphai:ecovector:0.1.0")
}
```
