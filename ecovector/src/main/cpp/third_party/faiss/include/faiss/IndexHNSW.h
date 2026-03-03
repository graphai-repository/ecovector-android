/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>
#include <mutex>
#include <queue>
#include <omp.h>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>


namespace faiss {

    struct IndexHNSW;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

    struct IndexHNSW : Index {
        typedef HNSW::storage_idx_t storage_idx_t;

        // the link structure
        HNSW hnsw;

        // the sequential storage
        bool own_fields = false;
        Index* storage = nullptr;

        // When set to false, level 0 in the knn graph is not initialized.
        // This option is used by GpuIndexCagra::copyTo(IndexHNSWCagra*)
        // as level 0 knn graph is copied over from the index built by
        // GpuIndexCagra.
        bool init_level0 = true;

        // Delete 관련 멤버 변수
        std::vector<bool> is_deleted;    // 삭제된 노드 표시를 위한 벡터
        std::vector<omp_lock_t> locks;   // 노드별 락을 위한 벡터
        const float alpha = 1.1f;        // neighbor selection에서 사용할 alpha 값
        size_t M_ = 32;         // 기본 이웃 수
        size_t maxM_ = 32;      // 기본 최대 이웃 수
        size_t maxM0_ = 32;     // 레벨 0에서의 최대 이웃 수

        // insert 시에는 2배까지 허용
        size_t insert_maxM_ = maxM_ * 2;     // 32
        size_t insert_maxM0_ = maxM0_ * 2;   // 64


        // Random level generator
        std::default_random_engine level_generator_;

        /** Get random level for node insertion
         * @param reverse_size reverse_size parameter for level generation
         * @return random level
         */
        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

        // When set to true, all neighbors in level 0 are filled up
        // to the maximum size allowed (2 * M). This option is used by
        // IndexHHNSWCagra to create a full base layer graph that is
        // used when GpuIndexCagra::copyFrom(IndexHNSWCagra*) is invoked.
        bool keep_max_size_level0 = false;

        explicit IndexHNSW(int d = 0, int M = 32, MetricType metric = METRIC_L2);
        explicit IndexHNSW(Index* storage, int M = 32);

        ~IndexHNSW() override;

        void add(idx_t n, const float* x) override;

        /// Trains the storage if needed
        void train(idx_t n, const float* x) override;

        /// entry point for search
        void search(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr) const override;

        void range_search(
                idx_t n,
                const float* x,
                float radius,
                RangeSearchResult* result,
                const SearchParameters* params = nullptr) const override;

        void reconstruct(idx_t key, float* recons) const override;

        void reset() override;

        // Delete & Insert 관련 함수들
        void directDelete_clean_nolock(storage_idx_t label);
        void directDelete_clean(storage_idx_t label);
        void directDelete_force(storage_idx_t label);
        void directDelete_knn_prune_nolock(storage_idx_t label);
        void directDelete_knn_prune(storage_idx_t label);
        void directDelete_knn_force(storage_idx_t label);
        void directDelete_only(storage_idx_t label);
        void directDelete(storage_idx_t label);
        void reconnectNeighborsDirectDelete(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors);
        void directDelete_knn(storage_idx_t label);
        void reconnectNeighborsDirectDelete_knn(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors);
        void reconnectNeighborsDirectDelete_knn_force(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors, int level);
        void reconnectNeighborsDirectDelete_knn_prune_nolock(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors, int level);
        void reconnectNeighborsDirectDelete_knn_prune(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors, int level);
        void removePhysicalNode(storage_idx_t node_id);
        void markDelete(storage_idx_t label);
        void unmarkDelete(storage_idx_t label);
        void reconnectNeighbors(storage_idx_t node_id);
        void insertPoint_knn_twoway_prune_nolock(storage_idx_t label);
        void insertPoint_knn_prune_nolock(storage_idx_t label);
        void insertPoint_knn_prune(storage_idx_t label);
        void insertPoint_knn_force(storage_idx_t label);
        void insertPoint_knn(storage_idx_t label);
        void insertPoint(storage_idx_t label);
        storage_idx_t selectRandomId();
        void getNeighborsByHeuristic2_direct_delete(
                std::priority_queue<std::pair<float, storage_idx_t>>& candidates,
                size_t M);
        void reconnectNeighborsByHeuristic_delete(
                storage_idx_t current_node,
                const std::vector<storage_idx_t>& potential_neighbors,
                size_t M);
        void getNeighborsByHeuristic2_insert(
                std::priority_queue<std::pair<float, storage_idx_t>>& candidates,
                size_t M);
        void shrink_level_0_neighbors(int size);
//  Test 관련 함수들

        /** Perform search only on level 0, given the starting points for
         * each vertex.
         *
         * @param search_type 1:perform one search per nprobe, 2: enqueue
         *                    all entry points
         */
        void search_level_0(
                idx_t n,
                const float* x,
                idx_t k,
                const storage_idx_t* nearest,
                const float* nearest_d,
                float* distances,
                idx_t* labels,
                int nprobe = 1,
                int search_type = 1,
                const SearchParameters* params = nullptr) const;

        /// alternative graph building
        void init_level_0_from_knngraph(int k, const float* D, const idx_t* I);

        /// alternative graph building
        void init_level_0_from_entry_points(
                int npt,
                const storage_idx_t* points,
                const storage_idx_t* nearests);

        // reorder links from nearest to farthest
        void reorder_links();

        void link_singletons();

        void permute_entries(const idx_t* perm);

        DistanceComputer* get_distance_computer() const override;

        void resizeIndex(size_t new_max_elements);
    };

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

    struct IndexHNSWFlat : IndexHNSW {
        IndexHNSWFlat();
        IndexHNSWFlat(int d, int M, MetricType metric = METRIC_L2);
    };

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
    struct IndexHNSWPQ : IndexHNSW {
        IndexHNSWPQ();
        IndexHNSWPQ(
                int d,
                int pq_m,
                int M,
                int pq_nbits = 8,
                MetricType metric = METRIC_L2);
        void train(idx_t n, const float* x) override;
    };

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
    struct IndexHNSWSQ : IndexHNSW {
        IndexHNSWSQ();
        IndexHNSWSQ(
                int d,
                ScalarQuantizer::QuantizerType qtype,
                int M,
                MetricType metric = METRIC_L2);
    };

/** 2-level code structure with fast random access
 */
    struct IndexHNSW2Level : IndexHNSW {
        IndexHNSW2Level();
        IndexHNSW2Level(Index* quantizer, size_t nlist, int m_pq, int M);

        void flip_to_ivf();

        /// entry point for search
        void search(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr) const override;
    };

    struct IndexHNSWCagra : IndexHNSW {
        IndexHNSWCagra();
        IndexHNSWCagra(int d, int M, MetricType metric = METRIC_L2);

        /// When set to true, the index is immutable.
        /// This option is used to copy the knn graph from GpuIndexCagra
        /// to the base level of IndexHNSWCagra without adding upper levels.
        /// Doing so enables to search the HNSW index, but removes the
        /// ability to add vectors.
        bool base_level_only = false;

        /// When `base_level_only` is set to `True`, the search function
        /// searches only the base level knn graph of the HNSW index.
        /// This parameter selects the entry point by randomly selecting
        /// some points and using the best one.
        int num_base_level_search_entrypoints = 32;

        void add(idx_t n, const float* x) override;

        /// entry point for search
        void search(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr) const override;
    };

} // namespace faiss