package io.graphai.ecovector

/**
 * Unified JNI interface for all vector database operations.
 * Handles the complete pipeline: chunking, tokenization, embedding, storage, and search.
 */
object NativeEcoVectorStore {
    init {
        // Load libc++_shared.so BEFORE libecovector.so to prevent libobjectbox-jni.so
        // (which statically links libc++ and exports __cxa_throw etc.) from shadowing
        // the shared C++ runtime symbols in the global symbol table.
        System.loadLibrary("c++_shared")
        System.loadLibrary("ecovector")
    }

    // --- Lifecycle ---
    external fun initialize(
        dbPath: String,
        tokenizerPath: String,
        modelPath: String,
        kiwiModelDir: String
    ): Boolean

    external fun close()

    /** Set a prefix string to prepend to each chunk after splitting. */
    external fun setChunkPrefix(prefix: String)
    external fun clearChunkPrefix()

    // --- Document Management ---
    /** Add a document. Auto chunks, tokenizes, embeds, and stores. Returns document ID. */
    external fun addDocument(text: String, title: String): Long

    /** Batch add documents. Returns array of document IDs. */
    external fun addDocuments(texts: Array<String>, titles: Array<String>): LongArray

    /** Batch add documents with external IDs. Returns array of document IDs. */
    external fun addDocumentsWithIds(texts: Array<String>, titles: Array<String>, externalIds: Array<String>): LongArray

    /** Batch add documents with external IDs + metadata. Returns array of document IDs. */
    external fun addDocumentsWithMetadata(
        texts: Array<String>, titles: Array<String>, externalIds: Array<String>,
        createdAts: LongArray, senders: Array<String>, sourceTypes: ShortArray = ShortArray(0)
    ): LongArray

    /** Add a document with custom chunk parameters. Returns document ID. */
    external fun addDocumentWithChunkParams(
        text: String, title: String,
        maxTokens: Int, overlapTokens: Int
    ): Long

    /** Batch add documents with custom chunk parameters. Returns number of documents added. */
    external fun addDocumentsWithChunkParams(
        texts: Array<String>, titles: Array<String>,
        maxTokens: Int, overlapTokens: Int
    ): Int

    /** Remove all documents and chunks. */
    external fun removeAll()

    /** Remove a single document and its chunks. Returns true on success. */
    external fun removeDocument(id: Long): Boolean

    /** Remove a single chunk. Returns true on success. */
    external fun removeChunk(id: Long): Boolean

    /** Remove documents that have no chunks (crash recovery). Returns JSON array of removed externalIds. */
    external fun removeOrphanDocuments(): String

    // --- Tokenization & Embedding ---
    /** Tokenize text to HuggingFace token IDs. */
    external fun tokenize(text: String): IntArray

    /** Embed text to float vector (ONNX inference). */
    external fun embed(text: String): FloatArray

    /** Tokenize text to Kiwi morpheme hashes (for BM25). */
    external fun tokenizeKiwi(text: String): IntArray

    // --- Index Management ---
    external fun buildIndex(centroidCount: Int): Boolean
    external fun buildVectorIndex(centroidCount: Int, hnswM: Int = 0, efConstruction: Int = 0, maxTrainSamples: Int = 0): Boolean
    external fun buildBM25Index(): Boolean
    external fun isIndexReady(): Boolean

    /**
     * Re-tokenize all stored chunks/queries with the current Kiwi dictionary
     * and rebuild BM25 index. Embeddings remain unchanged.
     * Use after updating user_dict.tsv / synonyms.tsv without full reindex.
     */
    external fun reTokenizeAll(): Boolean

    // --- Statistics ---
    external fun getDocumentCount(): Int
    external fun getChunkCount(): Int

    // --- Data Inspection ---
    external fun getDocumentsJson(offset: Int, limit: Int): String
    external fun getChunksJson(offset: Int, limit: Int): String

    /** Get a single document by ID. Returns JSON object string. */
    external fun getDocumentJson(id: Long): String

    /** Get all chunks belonging to a document. Returns JSON array string. */
    external fun getChunksByDocumentJson(docId: Long): String

    /** Get a single chunk by ID. Returns JSON object string. */
    external fun getChunkJson(id: Long): String

    // --- Lightweight ID-only access (for incremental loading) ---
    external fun getDocumentExternalIdsJson(): String

    // --- Data Import (pre-computed vectors, bypasses chunk/tokenize/embed pipeline) ---
    external fun importDocument(
        externalId: String, description: String, content: String,
        createdAt: Long, sourceType: Int, sender: String
    ): Long

    external fun importChunk(
        documentId: Long, chunkIndex: Int, content: String,
        tokenIds: IntArray, embedding: FloatArray,
        kiwiTokens: IntArray,
        createdAt: Long, sourceType: Int, sender: String
    ): Long

    /** Bulk update created_at on chunks by document external ID. Returns number of chunks updated. */
    external fun bulkUpdateCreatedAt(docExternalIds: Array<String>, createdAts: LongArray): Int

    /** Bulk update source_type on chunks by document external ID. Only updates mismatched chunks. */
    external fun bulkUpdateSourceType(docExternalIds: Array<String>, sourceTypes: ShortArray): Int

    // --- Pipeline Stage Methods ---
    /** Stage: 문서 저장만 (청크/임베딩/토큰화 없이) */
    external fun addDocumentsOnly(
        texts: Array<String>, titles: Array<String>, externalIds: Array<String>,
        createdAts: LongArray, senders: Array<String>, sourceTypes: IntArray
    ): Int

    /** Stage: 전체 문서 청크 분할 (기존 청크 삭제 후 재생성) */
    external fun chunkAllDocuments(): Int

    /** Stage: 청크 임베딩 (forceAll=false: 미임베딩만, true: 전체) */
    external fun embedChunks(forceAll: Boolean): Int

    /** Stage: 청크 Kiwi 토큰화 */
    external fun tokenizeChunks(): Int

    /** Stage: 청크(id+content)를 SQLite로 export (데스크탑 임베딩용) */
    external fun exportChunksToSQLite(sqlitePath: String): Int

    /** Stage: SQLite에서 사전 계산된 임베딩을 읽어 chunk vector 업데이트 */
    external fun importEmbeddingsFromSQLite(sqlitePath: String): Int
}
