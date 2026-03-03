/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

    using MinimaxHeap = HNSW::MinimaxHeap;
    using storage_idx_t = HNSW::storage_idx_t;
    using NodeDistFarther = HNSW::NodeDistFarther;

    HNSWStats hnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

    namespace {

        DistanceComputer* storage_distance_computer(const Index* storage) {
            if (is_similarity_metric(storage->metric_type)) {
                return new NegativeDistanceComputer(storage->get_distance_computer());
            } else {
                return storage->get_distance_computer();
            }
        }

        void hnsw_add_vertices(
                IndexHNSW& index_hnsw,
                size_t n0,
                size_t n,
                const float* x,
                bool verbose,
                bool preset_levels = false) {
            size_t d = index_hnsw.d;
            HNSW& hnsw = index_hnsw.hnsw;
            size_t ntotal = n0 + n;
            double t0 = getmillisecs();
            if (verbose) {
                printf("hnsw_add_vertices: adding %zd elements on top of %zd "
                       "(preset_levels=%d)\n",
                       n,
                       n0,
                       int(preset_levels));
            }

            if (n == 0) {
                return;
            }

            int max_level = hnsw.prepare_level_tab(n, preset_levels);

            if (verbose) {
                printf("  max_level = %d\n", max_level);
            }

            std::vector<omp_lock_t> locks(ntotal);
            for (int i = 0; i < ntotal; i++)
                omp_init_lock(&locks[i]);

            // add vectors from highest to lowest level
            std::vector<int> hist;
            std::vector<int> order(n);

            { // make buckets with vectors of the same level
                // build histogram
                for (int i = 0; i < n; i++) {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    while (pt_level >= hist.size())
                        hist.push_back(0);
                    hist[pt_level]++;
                }

                // accumulate
                std::vector<int> offsets(hist.size() + 1, 0);
                for (int i = 0; i < hist.size() - 1; i++) {
                    offsets[i + 1] = offsets[i] + hist[i];
                }

                // bucket sort
                for (int i = 0; i < n; i++) {
                    storage_idx_t pt_id = i + n0;
                    int pt_level = hnsw.levels[pt_id] - 1;
                    order[offsets[pt_level]++] = pt_id;
                }
            }

            idx_t check_period = InterruptCallback::get_period_hint(
                    max_level * index_hnsw.d * hnsw.efConstruction);

            { // perform add
                RandomGenerator rng2(789);

                int i1 = n;

                for (int pt_level = hist.size() - 1;
                     pt_level >= !index_hnsw.init_level0;
                     pt_level--) {
                    int i0 = i1 - hist[pt_level];

                    if (verbose) {
                        printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
                    }

                    // random permutation to get rid of dataset order bias
                    for (int j = i0; j < i1; j++)
                        std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

                    bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
                    {
                        VisitedTable vt(ntotal);

                        std::unique_ptr<DistanceComputer> dis(
                                storage_distance_computer(index_hnsw.storage));
                        int prev_display =
                                verbose && omp_get_thread_num() == 0 ? 0 : -1;
                        size_t counter = 0;

#pragma omp for schedule(static)
                        for (int i = i0; i < i1; i++) {
                            storage_idx_t pt_id = order[i];
                            dis->set_query(x + (pt_id - n0) * d);

                            if (interrupt) {
                                continue;
                            }

                            hnsw.add_with_locks(
                                    *dis,
                                    pt_level,
                                    pt_id,
                                    locks,
                                    vt,
                                    index_hnsw.keep_max_size_level0 && (pt_level == 0));

                            if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                                prev_display = i - i0;
                                printf("  %d / %d\r", i - i0, i1 - i0);
                                fflush(stdout);
                            }
                            if (counter % check_period == 0) {
                                if (InterruptCallback::is_interrupted()) {
                                    interrupt = true;
                                }
                            }
                            counter++;
                        }
                    }
                    if (interrupt) {
                        FAISS_THROW_MSG("computation interrupted");
                    }
                    i1 = i0;
                }
                if (index_hnsw.init_level0) {
                    FAISS_ASSERT(i1 == 0);
                } else {
                    FAISS_ASSERT((i1 - hist[0]) == 0);
                }
            }
            if (verbose) {
                printf("Done in %.3f ms\n", getmillisecs() - t0);
            }

            for (int i = 0; i < ntotal; i++) {
                omp_destroy_lock(&locks[i]);
            }
        }

    } // anonymous namespace

    /**************************************************************
 * IndexHNSW implementation
 **************************************************************/





    IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
            : Index(d, metric), hnsw(M) {
        is_deleted.resize(0);
        locks.resize(0);

        // Insert용 변수들 초기화
        M_ = M;
        maxM_ = M;
        maxM0_ = M * 2;
        insert_maxM_ = maxM_ * 2;
        insert_maxM0_ = maxM0_ * 2;

    }

    IndexHNSW::IndexHNSW(Index* storage, int M)
            : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {
        is_deleted.resize(0);
        locks.resize(0);

        // Insert용 변수들 초기화
        M_ = M;
        maxM_ = M;
        maxM0_ = M * 2;
        insert_maxM_ = maxM_ * 2;
        insert_maxM0_ = maxM0_ * 2;

    }

    IndexHNSW::~IndexHNSW() {
        if (own_fields) {
            delete storage;
        }

        for (size_t i = 0; i < locks.size(); i++) {
            omp_destroy_lock(&locks[i]);
        }
    }

    void IndexHNSW::train(idx_t n, const float* x) {
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
        storage->train(n, x);
        is_trained = true;
    }
    namespace {

        template <class BlockResultHandler>
        void hnsw_search(
                const IndexHNSW* index,
                idx_t n,
                const float* x,
                BlockResultHandler& bres,
                const SearchParameters* params_in) {
            FAISS_THROW_IF_NOT_MSG(
                    index->storage,
                    "No storage index, please use IndexHNSWFlat (or variants) "
                    "instead of IndexHNSW directly");
            const SearchParametersHNSW* params = nullptr;
            const HNSW& hnsw = index->hnsw;

            int efSearch = hnsw.efSearch;
            if (params_in) {
                params = dynamic_cast<const SearchParametersHNSW*>(params_in);
                FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
                efSearch = params->efSearch;
            }
            size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

            idx_t check_period = InterruptCallback::get_period_hint(
                    hnsw.max_level * index->d * efSearch);

            for (idx_t i0 = 0; i0 < n; i0 += check_period) {
                idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
                {
                    VisitedTable vt(index->ntotal);
                    typename BlockResultHandler::SingleResultHandler res(bres);

                    std::unique_ptr<DistanceComputer> dis(
                            storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
                    for (idx_t i = i0; i < i1; i++) {
                        res.begin(i);
                        dis->set_query(x + i * index->d);

                        HNSWStats stats = hnsw.search(*dis, res, vt, params);
                        n1 += stats.n1;
                        n2 += stats.n2;
                        ndis += stats.ndis;
                        nhops += stats.nhops;
                        res.end();
                    }
                }
                InterruptCallback::check();
            }

            hnsw_stats.combine({n1, n2, ndis, nhops});
        }

    } // anonymous namespace



    void IndexHNSW::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in) const {
        FAISS_THROW_IF_NOT(k > 0);

        using RH = HeapBlockResultHandler<HNSW::C>;
        RH bres(n, distances, labels, k);

        hnsw_search(this, n, x, bres, params_in);

        if (is_similarity_metric(this->metric_type)) {
            for (size_t i = 0; i < k * n; i++) {
                distances[i] = -distances[i];
            }
        }
    }

    void IndexHNSW::range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params) const {
        using RH = RangeSearchBlockResultHandler<HNSW::C>;
        RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

        hnsw_search(this, n, x, bres, params);

        if (is_similarity_metric(this->metric_type)) {
            // we need to revert the negated distances
            for (size_t i = 0; i < result->lims[result->nq]; i++) {
                result->distances[i] = -result->distances[i];
            }
        }
    }


    void IndexHNSW::add(idx_t n, const float* x) {
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
        FAISS_THROW_IF_NOT(is_trained);

        int n0 = ntotal;
        storage->add(n, x);
        ntotal = storage->ntotal;

        size_t old_size = is_deleted.size();
        is_deleted.resize(ntotal, false);

        locks.resize(ntotal);
        for (size_t i = old_size; i < ntotal; i++) {
            omp_init_lock(&locks[i]);
        }

        hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
    }

    void IndexHNSW::reset() {
        hnsw.reset();
        storage->reset();
        ntotal = 0;

        for (size_t i = 0; i < locks.size(); i++) {
            omp_destroy_lock(&locks[i]);
        }
        locks.clear();
        locks.resize(0);

        is_deleted.clear();
        is_deleted.resize(0);
    }

    void IndexHNSW::reconstruct(idx_t key, float* recons) const {
        storage->reconstruct(key, recons);
    }

    void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
        {
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

#pragma omp for
            for (idx_t i = 0; i < ntotal; i++) {
                size_t begin, end;
                hnsw.neighbor_range(i, 0, &begin, &end);

                std::priority_queue<NodeDistFarther> initial_list;

                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    initial_list.emplace(dis->symmetric_dis(i, v1), v1);
                }

                std::vector<NodeDistFarther> shrunk_list;
                HNSW::shrink_neighbor_list(
                        *dis, initial_list, shrunk_list, new_size);

                for (size_t j = begin; j < end; j++) {
                    if (j - begin < shrunk_list.size())
                        hnsw.neighbors[j] = shrunk_list[j - begin].id;
                    else
                        hnsw.neighbors[j] = -1;
                }
            }
        }
    }
//PTH START
    void IndexHNSW::getNeighborsByHeuristic2_direct_delete(
            std::priority_queue<std::pair<float, storage_idx_t>>& candidates,
            size_t M) {
        if (candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<float, storage_idx_t>> queue_closest;
        std::vector<std::pair<float, storage_idx_t>> return_list;

        while (!candidates.empty()) {
            queue_closest.emplace(-candidates.top().first, candidates.top().second);
            candidates.pop();
        }

        while (!queue_closest.empty()) {
            if (return_list.size() >= M) break;

            auto current = queue_closest.top();
            float dist_to_query = -current.first;
            queue_closest.pop();
            bool good = true;

            for (const auto& selected : return_list) {
                std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
                float dist = dis->symmetric_dis(current.second, selected.second);

                if (alpha * dist < dist_to_query) {
                    good = false;
                    break;
                }
            }

            if (good) {
                return_list.push_back(current);
            }
        }

        for (const auto& neighbor : return_list) {
            candidates.emplace(-neighbor.first, neighbor.second);
        }
    }

    void IndexHNSW::reconnectNeighborsByHeuristic_delete(
            storage_idx_t current_node,
            const std::vector<storage_idx_t>& potential_neighbors,
            size_t M) {

        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

        // 현재 노드의 기존 이웃들 확인
        size_t begin, end;
        hnsw.neighbor_range(current_node, 0, &begin, &end);
        std::vector<storage_idx_t> existing_neighbors;
        for (size_t i = begin; i < end; i++) {
            if (hnsw.neighbors[i] >= 0) {
                existing_neighbors.push_back(hnsw.neighbors[i]);
            }
        }

        // 연결할 이웃 후보들 저장
        std::vector<std::pair<float, storage_idx_t>> candidates;

        // 각 potential neighbor에 대해 검사
        for (auto neighbor : potential_neighbors) {
            if (neighbor == current_node) continue;

            float dist_to_neighbor = dis->symmetric_dis(current_node, neighbor);
            bool should_connect = true;

            // 기존 이웃들과 비교
            for (auto existing : existing_neighbors) {
                float existing_dist = dis->symmetric_dis(current_node, existing);
                if (alpha * existing_dist < dist_to_neighbor) {
                    should_connect = false;
                    break;
                }
            }

            if (should_connect) {
                candidates.emplace_back(dist_to_neighbor, neighbor);
            }
        }

        // 거리 순으로 정렬
        std::sort(candidates.begin(), candidates.end());

        // 새로운 이웃 연결 설정 (최대 M개)
        size_t idx = begin;
        for (size_t i = 0; i < std::min(M, candidates.size()); i++) {
            hnsw.neighbors[idx++] = candidates[i].second;
        }

        // 남은 슬롯 초기화
        while (idx < end) {
            hnsw.neighbors[idx++] = -1;
        }
    }

// Insert용 휴리스틱
    void IndexHNSW::getNeighborsByHeuristic2_insert(
            std::priority_queue<std::pair<float, storage_idx_t>>& candidates,
            size_t M) {
        if (candidates.size() < M) {
            return;
        }

        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
        std::priority_queue<std::pair<float, storage_idx_t>> queue_closest;
        std::vector<std::pair<float, storage_idx_t>> return_list;

        // 거리순으로 정렬
        while (!candidates.empty()) {
            queue_closest.emplace(-candidates.top().first, candidates.top().second);
            candidates.pop();
        }

        // 첫 번째 가장 가까운 이웃은 무조건 선택
        if (!queue_closest.empty()) {
            return_list.push_back(queue_closest.top());
            queue_closest.pop();
        }

        // 나머지 이웃 선택
        while (!queue_closest.empty() && return_list.size() < M) {
            auto current = queue_closest.top();
            float dist_to_query = -current.first;
            queue_closest.pop();
            bool good = true;

            // 이미 선택된 이웃들과 비교
            for (const auto& selected : return_list) {
                float dist = dis->symmetric_dis(current.second, selected.second);
                if (alpha * dist < dist_to_query) {
                    good = false;
                    break;
                }
            }

            if (good) {
                return_list.push_back(current);
            }
        }

        // 결과 반환
        candidates = std::priority_queue<std::pair<float, storage_idx_t>>();
        for (const auto& neighbor : return_list) {
            candidates.emplace(-neighbor.first, neighbor.second);
        }
    }



    void IndexHNSW::directDelete_force(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_force");
        }

        omp_set_lock(&locks[label]);

        try {
            // 모든 레벨에서 작업 수행 (Top Layer 제외)
            int max_level = hnsw.levels[label];
            for (int level = 0; level < max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);

                // 현재 레벨의 모든 이웃들과의 연결 제거
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0) break;

                    // 이웃 노드의 락 획득
                    omp_set_lock(&locks[neighbor_id]);

                    // 이웃 노드에서 현재 노드로의 연결 제거
                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] == label) {
                            // 연결 제거 및 리스트 정리
                            for (size_t k = j + 1; k < nb_end; k++) {
                                if (hnsw.neighbors[k] < 0) break;
                                hnsw.neighbors[k-1] = hnsw.neighbors[k];
                            }
                            hnsw.neighbors[nb_end - 1] = -1;
                            break;
                        }
                    }

                    // 이웃 노드의 락 해제
                    omp_unset_lock(&locks[neighbor_id]);

                    // 현재 노드의 이웃 연결 제거
                    hnsw.neighbors[i] = -1;
                }
            }

            // 노드 물리적 제거
            removePhysicalNode(label);
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::directDelete_knn_force(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn_force");
        }

        omp_set_lock(&locks[label]);

        try {
            // 모든 레벨에서 작업 수행 (Top Layer 제외)
            int max_level = hnsw.levels[label];
            for (int level = 0; level < max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);
                std::vector<storage_idx_t> old_neighbors;

                // 현재 레벨의 이웃들 저장
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0) break;
                    old_neighbors.push_back(neighbor_id);
                }

                // 이웃들 간의 연결 재구성
                reconnectNeighborsDirectDelete_knn_force(label, old_neighbors, level);
            }

            // 노드 물리적 제거
            removePhysicalNode(label);
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::reconnectNeighborsDirectDelete_knn_force(
            storage_idx_t node_id,
            std::vector<storage_idx_t>& old_neighbors,
            int level) {

        const size_t k = 10;
        hnsw.efSearch = 10;

        for (size_t i = 0; i < old_neighbors.size(); i++) {
            storage_idx_t current = old_neighbors[i];
            omp_set_lock(&locks[current]);

            std::vector<float> current_vec(d);
            storage->reconstruct(current, current_vec.data());

            size_t nb_begin, nb_end;
            hnsw.neighbor_range(current, level, &nb_begin, &nb_end);

            // 삭제할 노드와의 연결 제거
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] == node_id) {
                    for (size_t k = j + 1; k < nb_end; k++) {
                        if (hnsw.neighbors[k] < 0) break;
                        hnsw.neighbors[k-1] = hnsw.neighbors[k];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                    break;
                }
            }

            // kNN 검색 수행
            std::vector<float> distances(k);
            std::vector<idx_t> neighbors(k);

            // search 함수 사용하되 현재 레벨에서만 검색
            int tmp_max_level = hnsw.max_level;
            hnsw.max_level = level;  // 임시로 max_level을 현재 레벨로 설정
            search(1, current_vec.data(), k, distances.data(), neighbors.data());
            hnsw.max_level = tmp_max_level;  // max_level 복원

            // 현재 레벨에서 새로운 연결 설정
            size_t idx = nb_begin;
            for (size_t j = 0; j < k; j++) {
                idx_t new_neighbor = neighbors[j];
                if (new_neighbor == current || new_neighbor == node_id ||
                    new_neighbor < 0 || is_deleted[new_neighbor]) {
                    continue;
                }

                hnsw.neighbors[idx++] = new_neighbor;

                omp_set_lock(&locks[new_neighbor]);
                size_t other_begin, other_end;
                hnsw.neighbor_range(new_neighbor, level, &other_begin, &other_end);

                for (size_t l = other_begin; l < other_end; l++) {
                    if (hnsw.neighbors[l] < 0) {
                        hnsw.neighbors[l] = current;
                        break;
                    }
                }
                omp_unset_lock(&locks[new_neighbor]);

                if (idx >= nb_end) break;
            }

            while (idx < nb_end) {
                hnsw.neighbors[idx++] = -1;
            }

            omp_unset_lock(&locks[current]);
        }
    }

    void IndexHNSW::insertPoint_knn_twoway_prune_nolock(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn_twoway_prune_nolock");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        // 1. 레벨 할당
        int insert_level = hnsw.levels[label];
        if (insert_level <= 0) {
            insert_level = getRandomLevel(1.0 / log(1.0 * insert_maxM_));
            hnsw.levels[label] = insert_level;
        }

        // 2. 진입점 찾기
        storage_idx_t currObj = hnsw.entry_point;
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
        dis->set_query(point_vec.data());
        float curr_dist = dis->symmetric_dis(currObj, label);

        // 상위 레벨부터 시작하여 좋은 진입점 찾기
        for (int level = hnsw.max_level; level > insert_level; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t begin, end;
                hnsw.neighbor_range(currObj, level, &begin, &end);

                for (size_t idx = begin; idx < end; idx++) {
                    storage_idx_t neighbor = hnsw.neighbors[idx];
                    if (neighbor < 0 || is_deleted[neighbor]) continue;

                    float dist = dis->symmetric_dis(neighbor, label);
                    if (dist < curr_dist) {
                        curr_dist = dist;
                        currObj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 3. 각 레벨에서 이웃 연결
        for (int level = std::min(insert_level, hnsw.max_level); level >= 0; level--) {
            std::vector<std::pair<float, storage_idx_t>> candidates;
            const size_t ef_construction = std::max(size_t(insert_maxM_), size_t(40));

            // 진입점을 첫 번째 후보로 추가
            candidates.emplace_back(curr_dist, currObj);
            std::vector<bool> visited(ntotal, false);
            visited[currObj] = true;

            // 후보 확장
            std::priority_queue<std::pair<float, storage_idx_t>> top_candidates;
            size_t candidate_count = 0;

            while (candidate_count < ef_construction && !candidates.empty()) {
                auto min_it = std::min_element(candidates.begin(), candidates.end(),
                                               [](const auto& a, const auto& b) { return a.first < b.first; });

                float dist_to_query = min_it->first;
                storage_idx_t current_node = min_it->second;
                candidates.erase(min_it);

                size_t begin, end;
                hnsw.neighbor_range(current_node, level, &begin, &end);

                for (size_t idx = begin; idx < end; idx++) {
                    storage_idx_t neighbor = hnsw.neighbors[idx];
                    if (neighbor < 0 || is_deleted[neighbor] || visited[neighbor]) continue;

                    visited[neighbor] = true;
                    float dist = dis->symmetric_dis(neighbor, label);
                    candidates.emplace_back(dist, neighbor);
                    top_candidates.emplace(-dist, neighbor);
                    candidate_count++;
                }
            }

            // RobustPrune 적용
            size_t Mcurmax = (level == 0) ? insert_maxM0_ : insert_maxM_;
            std::vector<std::pair<float, storage_idx_t>> pruned_candidates;
            std::vector<storage_idx_t> final_neighbors;

            // 진입점은 무조건 첫 번째로 추가
            pruned_candidates.emplace_back(curr_dist, currObj);
            final_neighbors.push_back(currObj);

            while (!top_candidates.empty() && pruned_candidates.size() < Mcurmax) {
                auto current = std::make_pair(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();

                bool should_add = true;
                // pruning 조건 체크
                for (const auto& selected : pruned_candidates) {
                    float dist_between = dis->symmetric_dis(current.second, selected.second);
                    if (alpha * dist_between <= current.first) {
                        should_add = false;
                        break;
                    }
                }

                if (should_add) {
                    pruned_candidates.push_back(current);
                    final_neighbors.push_back(current.second);
                }
            }

            // 이웃 연결 설정 (A->B)
            size_t begin, end;
            hnsw.neighbor_range(label, level, &begin, &end);
            size_t idx = begin;
            for (auto neighbor_id : final_neighbors) {
                hnsw.neighbors[idx++] = neighbor_id;

                // B->A 연결 시도
                size_t nb_begin, nb_end;
                hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                // 1. 기존 이웃들과 새로운 노드의 거리 계산
                std::vector<std::pair<float, storage_idx_t>> existing_neighbors;
                float max_dist = -1;
                storage_idx_t max_idx_neighbor = -1;
                size_t max_idx_slot = nb_begin;
                bool has_space = false;

                for (size_t j = nb_begin; j < nb_end; j++) {
                    storage_idx_t existing_neighbor = hnsw.neighbors[j];
                    if (existing_neighbor < 0) {
                        has_space = true;
                        max_idx_slot = j;
                        break;
                    }
                    if (!is_deleted[existing_neighbor]) {
                        float dist = dis->symmetric_dis(neighbor_id, existing_neighbor);
                        if (dist > max_dist) {
                            max_dist = dist;
                            max_idx_neighbor = existing_neighbor;
                            max_idx_slot = j;
                        }
                    }
                }

                // 2. 새로운 노드와의 거리 계산
                float new_dist = dis->symmetric_dis(neighbor_id, label);

                // 3. RobustPrune 조건으로 B->A 연결 결정
                bool should_add = has_space || new_dist < max_dist;
                if (!has_space && should_add) {
                    // 추가 pruning 체크
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        storage_idx_t existing = hnsw.neighbors[j];
                        if (existing >= 0 && !is_deleted[existing] && existing != max_idx_neighbor) {
                            float dist_between = dis->symmetric_dis(label, existing);
                            if (alpha * dist_between <= new_dist) {
                                should_add = false;
                                break;
                            }
                        }
                    }
                }

                // 4. 조건을 만족하면 연결
                if (should_add) {
                    hnsw.neighbors[max_idx_slot] = label;
                }
            }

            // 남은 슬롯 초기화
            while (idx < end) {
                hnsw.neighbors[idx++] = -1;
            }
        }

        is_deleted[label] = false;

        if (insert_level > hnsw.max_level) {
            hnsw.max_level = insert_level;
            hnsw.entry_point = label;
        }
    }


    void IndexHNSW::insertPoint_knn_prune_nolock(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn_prune_nolock");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        // 1. 레벨 할당
        int insert_level = hnsw.levels[label];
        if (insert_level <= 0) {
            insert_level = getRandomLevel(1.0 / log(1.0 * insert_maxM_));
            hnsw.levels[label] = insert_level;
        }

        // 2. 진입점 찾기
        storage_idx_t currObj = hnsw.entry_point;
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
        dis->set_query(point_vec.data());
        float curr_dist = dis->symmetric_dis(currObj, label);

        // 상위 레벨부터 시작하여 좋은 진입점 찾기
        for (int level = hnsw.max_level; level > insert_level; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t begin, end;
                hnsw.neighbor_range(currObj, level, &begin, &end);

                for (size_t idx = begin; idx < end; idx++) {
                    storage_idx_t neighbor = hnsw.neighbors[idx];
                    if (neighbor < 0 || is_deleted[neighbor]) continue;

                    float dist = dis->symmetric_dis(neighbor, label);
                    if (dist < curr_dist) {
                        curr_dist = dist;
                        currObj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 3. 각 레벨에서 이웃 연결
        for (int level = std::min(insert_level, hnsw.max_level); level >= 0; level--) {
            std::vector<std::pair<float, storage_idx_t>> candidates;
            const size_t ef_construction = std::max(size_t(insert_maxM_), size_t(40));

            // 진입점을 첫 번째 후보로 추가
            candidates.emplace_back(curr_dist, currObj);
            std::vector<bool> visited(ntotal, false);
            visited[currObj] = true;

            // 후보 확장
            std::priority_queue<std::pair<float, storage_idx_t>> top_candidates;
            size_t candidate_count = 0;

            while (candidate_count < ef_construction && !candidates.empty()) {
                auto min_it = std::min_element(candidates.begin(), candidates.end(),
                                               [](const auto& a, const auto& b) { return a.first < b.first; });

                float dist_to_query = min_it->first;
                storage_idx_t current_node = min_it->second;
                candidates.erase(min_it);

                size_t begin, end;
                hnsw.neighbor_range(current_node, level, &begin, &end);

                for (size_t idx = begin; idx < end; idx++) {
                    storage_idx_t neighbor = hnsw.neighbors[idx];
                    if (neighbor < 0 || is_deleted[neighbor] || visited[neighbor]) continue;

                    visited[neighbor] = true;
                    float dist = dis->symmetric_dis(neighbor, label);
                    candidates.emplace_back(dist, neighbor);
                    top_candidates.emplace(-dist, neighbor);
                    candidate_count++;
                }
            }

            // RobustPrune 적용
            size_t Mcurmax = (level == 0) ? insert_maxM0_ : insert_maxM_;
            std::vector<std::pair<float, storage_idx_t>> pruned_candidates;
            std::vector<storage_idx_t> final_neighbors;

            // 진입점은 무조건 첫 번째로 추가
            pruned_candidates.emplace_back(curr_dist, currObj);
            final_neighbors.push_back(currObj);

            while (!top_candidates.empty() && pruned_candidates.size() < Mcurmax) {
                auto current = std::make_pair(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();

                bool should_add = true;
                // pruning 조건 체크
                for (const auto& selected : pruned_candidates) {
                    float dist_between = dis->symmetric_dis(current.second, selected.second);
                    if (alpha * dist_between <= current.first) {
                        should_add = false;
                        break;
                    }
                }

                if (should_add) {
                    pruned_candidates.push_back(current);
                    final_neighbors.push_back(current.second);
                }
            }

            // 이웃 연결 설정
            size_t begin, end;
            hnsw.neighbor_range(label, level, &begin, &end);

            // 현재 노드에서 선택된 이웃들로 연결
            size_t idx = begin;
            for (auto neighbor_id : final_neighbors) {
                hnsw.neighbors[idx++] = neighbor_id;

                // 양방향 연결 설정
                size_t nb_begin, nb_end;
                hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                // 상대 노드의 이웃 목록에 현재 노드 추가 (여유 공간 있으면)
                bool connection_added = false;
                for (size_t j = nb_begin; j < nb_end; j++) {
                    if (hnsw.neighbors[j] < 0) {
                        hnsw.neighbors[j] = label;
                        connection_added = true;
                        break;
                    }
                }

                if (!connection_added) {
                    // 공간이 없으면 가장 먼 이웃을 대체
                    float max_dist = -1;
                    size_t max_idx = nb_begin;
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] >= 0) {
                            float dist = dis->symmetric_dis(neighbor_id, hnsw.neighbors[j]);
                            if (dist > max_dist) {
                                max_dist = dist;
                                max_idx = j;
                            }
                        }
                    }

                    float new_dist = dis->symmetric_dis(neighbor_id, label);
                    if (new_dist < max_dist) {
                        hnsw.neighbors[max_idx] = label;
                    }
                }
            }

            // 남은 슬롯 초기화
            while (idx < end) {
                hnsw.neighbors[idx++] = -1;
            }
        }

        is_deleted[label] = false;

        if (insert_level > hnsw.max_level) {
            hnsw.max_level = insert_level;
            hnsw.entry_point = label;
        }
    }

    void IndexHNSW::insertPoint_knn_prune(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn_prune");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        omp_set_lock(&locks[label]);

        try {
            // 1. 레벨 할당
            int insert_level = hnsw.levels[label];
            if (insert_level <= 0) {
                insert_level = getRandomLevel(1.0 / log(1.0 * insert_maxM_));
                hnsw.levels[label] = insert_level;
            }

            // 2. 진입점 찾기
            storage_idx_t currObj = hnsw.entry_point;
            std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
            dis->set_query(point_vec.data());
            float curr_dist = dis->symmetric_dis(currObj, label);

            // 상위 레벨부터 시작하여 좋은 진입점 찾기
            for (int level = hnsw.max_level; level > insert_level; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    size_t begin, end;
                    hnsw.neighbor_range(currObj, level, &begin, &end);

                    for (size_t idx = begin; idx < end; idx++) {
                        storage_idx_t neighbor = hnsw.neighbors[idx];
                        if (neighbor < 0 || is_deleted[neighbor]) continue;

                        float dist = dis->symmetric_dis(neighbor, label);
                        if (dist < curr_dist) {
                            curr_dist = dist;
                            currObj = neighbor;
                            changed = true;
                        }
                    }
                }
            }

            // 3. 각 레벨에서 이웃 연결
            for (int level = std::min(insert_level, hnsw.max_level); level >= 0; level--) {
                std::vector<std::pair<float, storage_idx_t>> candidates;
                const size_t ef_construction = std::max(size_t(insert_maxM_), size_t(40));

                // 진입점을 첫 번째 후보로 추가
                candidates.emplace_back(curr_dist, currObj);
                std::vector<bool> visited(ntotal, false);
                visited[currObj] = true;

                // 후보 확장
                std::priority_queue<std::pair<float, storage_idx_t>> top_candidates;
                size_t candidate_count = 0;

                while (candidate_count < ef_construction && !candidates.empty()) {
                    auto min_it = std::min_element(candidates.begin(), candidates.end(),
                                                   [](const auto& a, const auto& b) { return a.first < b.first; });

                    float dist_to_query = min_it->first;
                    storage_idx_t current_node = min_it->second;
                    candidates.erase(min_it);

                    size_t begin, end;
                    hnsw.neighbor_range(current_node, level, &begin, &end);

                    for (size_t idx = begin; idx < end; idx++) {
                        storage_idx_t neighbor = hnsw.neighbors[idx];
                        if (neighbor < 0 || is_deleted[neighbor] || visited[neighbor]) continue;

                        visited[neighbor] = true;
                        float dist = dis->symmetric_dis(neighbor, label);
                        candidates.emplace_back(dist, neighbor);
                        top_candidates.emplace(-dist, neighbor);
                        candidate_count++;
                    }
                }

                // RobustPrune 적용
                size_t Mcurmax = (level == 0) ? insert_maxM0_ : insert_maxM_;
                std::vector<std::pair<float, storage_idx_t>> pruned_candidates;
                std::vector<storage_idx_t> final_neighbors;

                // 진입점은 무조건 첫 번째로 추가
                pruned_candidates.emplace_back(curr_dist, currObj);
                final_neighbors.push_back(currObj);

                while (!top_candidates.empty() && pruned_candidates.size() < Mcurmax) {
                    auto current = std::make_pair(-top_candidates.top().first, top_candidates.top().second);
                    top_candidates.pop();

                    bool should_add = true;
                    // pruning 조건 체크
                    for (const auto& selected : pruned_candidates) {
                        float dist_between = dis->symmetric_dis(current.second, selected.second);
                        if (alpha * dist_between <= current.first) {
                            should_add = false;
                            break;
                        }
                    }

                    if (should_add) {
                        pruned_candidates.push_back(current);
                        final_neighbors.push_back(current.second);
                    }
                }

                // 이웃 연결 설정
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);

                // 현재 노드에서 선택된 이웃들로 연결
                size_t idx = begin;
                for (auto neighbor_id : final_neighbors) {
                    hnsw.neighbors[idx++] = neighbor_id;

                    // 양방향 연결 설정
                    omp_set_lock(&locks[neighbor_id]);
                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                    // 상대 노드의 이웃 목록에 현재 노드 추가 (여유 공간 있으면)
                    bool connection_added = false;
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] < 0) {
                            hnsw.neighbors[j] = label;
                            connection_added = true;
                            break;
                        }
                    }

                    if (!connection_added) {
                        // 공간이 없으면 가장 먼 이웃을 대체
                        float max_dist = -1;
                        size_t max_idx = nb_begin;
                        for (size_t j = nb_begin; j < nb_end; j++) {
                            if (hnsw.neighbors[j] >= 0) {
                                float dist = dis->symmetric_dis(neighbor_id, hnsw.neighbors[j]);
                                if (dist > max_dist) {
                                    max_dist = dist;
                                    max_idx = j;
                                }
                            }
                        }

                        float new_dist = dis->symmetric_dis(neighbor_id, label);
                        if (new_dist < max_dist) {
                            hnsw.neighbors[max_idx] = label;
                        }
                    }
                    omp_unset_lock(&locks[neighbor_id]);
                }

                // 남은 슬롯 초기화
                while (idx < end) {
                    hnsw.neighbors[idx++] = -1;
                }
            }

            is_deleted[label] = false;

            if (insert_level > hnsw.max_level) {
                hnsw.max_level = insert_level;
                hnsw.entry_point = label;
            }

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }


    void IndexHNSW::insertPoint_knn_force(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn_force");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        const size_t k = 10;
        hnsw.efSearch = 10;

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        omp_set_lock(&locks[label]);

        try {
            int max_level = hnsw.levels[label];
            for (int level = 0; level < max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);

                // kNN 검색 수행
                std::vector<float> distances(k);
                std::vector<idx_t> neighbors(k);

                // search 함수 사용하되 현재 레벨에서만 검색
                int tmp_max_level = hnsw.max_level;
                hnsw.max_level = level;  // 임시로 max_level을 현재 레벨로 설정
                search(1, point_vec.data(), k, distances.data(), neighbors.data());
                hnsw.max_level = tmp_max_level;  // max_level 복원

                size_t idx = begin;
                for (size_t i = 0; i < k; i++) {
                    idx_t new_neighbor = neighbors[i];
                    if (new_neighbor == label || new_neighbor < 0 || is_deleted[new_neighbor]) {
                        continue;
                    }

                    hnsw.neighbors[idx++] = new_neighbor;

                    omp_set_lock(&locks[new_neighbor]);
                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(new_neighbor, level, &nb_begin, &nb_end);

                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] < 0) {
                            hnsw.neighbors[j] = label;
                            break;
                        }
                    }
                    omp_unset_lock(&locks[new_neighbor]);

                    if (idx >= end) break;
                }

                while (idx < end) {
                    hnsw.neighbors[idx++] = -1;
                }
            }

            is_deleted[label] = false;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::directDelete_only(storage_idx_t label) {
        // 1. 유효성 검사
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_only");
        }

        // 2. 이미 삭제된 노드인지 확인
        if (is_deleted[label]) {
            throw std::runtime_error("The requested to delete element is already deleted");
        }

        // 3. 노드 락 획득
        omp_set_lock(&locks[label]);

        try {
            // 4. 모든 레벨에서 처리 (Top Layer 제외)
            int max_level = hnsw.levels[label];
            for (int level = 0; level < max_level; level++) {
                // 현재 레벨의 이웃 정보 가져오기
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);

                // 이웃들과의 양방향 연결 제거
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0) break;

                    // 이웃 노드 락 획득
                    omp_set_lock(&locks[neighbor_id]);

                    // 이웃 노드의 연결 목록에서 현재 노드 제거
                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] == label) {
                            // 연결 제거 및 리스트 정리
                            for (size_t k = j + 1; k < nb_end; k++) {
                                if (hnsw.neighbors[k] < 0) break;
                                hnsw.neighbors[k-1] = hnsw.neighbors[k];
                            }
                            hnsw.neighbors[nb_end - 1] = -1;
                            break;
                        }
                    }

                    omp_unset_lock(&locks[neighbor_id]);
                }

                // 현재 레벨의 이웃 목록 초기화
                for (size_t i = begin; i < end; i++) {
                    hnsw.neighbors[i] = -1;
                }
            }

            // 5. level 정보 초기화
            hnsw.levels[label] = 0;

            // 6. 삭제 표시
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            // 예외 발생 시 락 해제 보장
            omp_unset_lock(&locks[label]);
            throw;
        }

        // 7. 락 해제
        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::directDelete_clean_nolock(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_clean_nolock");
        }

        if (is_deleted[label]) {
            throw std::runtime_error("The requested to delete element is already deleted");
        }

        // Top Level 처리
        if (label == hnsw.entry_point) {
            // 새로운 entry_point 찾기
            storage_idx_t new_entry_point = -1;
            int new_max_level = -1;

            // 현재 entry_point의 최상위 레벨에서 새로운 entry_point 찾기
            size_t begin, end;
            hnsw.neighbor_range(label, hnsw.max_level, &begin, &end);

            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor_id = hnsw.neighbors[i];
                if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;

                if (new_entry_point == -1 || hnsw.levels[neighbor_id] > new_max_level) {
                    new_entry_point = neighbor_id;
                    new_max_level = hnsw.levels[neighbor_id];
                }
            }

            // 최상위 레벨에 노드가 없으면 하위 레벨에서 찾기
            if (new_entry_point == -1) {
                for (int level = hnsw.max_level - 1; level >= 0; level--) {
                    for (storage_idx_t i = 0; i < ntotal; i++) {
                        if (i != label && !is_deleted[i] && hnsw.levels[i] == level) {
                            new_entry_point = i;
                            new_max_level = level;
                            break;
                        }
                    }
                    if (new_entry_point != -1) break;
                }
            }

            // entry_point 및 max_level 업데이트
            if (new_entry_point != -1) {
                hnsw.entry_point = new_entry_point;
                hnsw.max_level = new_max_level;
            } else {
                // 그래프가 비어있게 되는 경우
                hnsw.entry_point = -1;
                hnsw.max_level = 0;
            }
        }
            // Top Level이 아닌 경우에도 max_level 조정이 필요할 수 있음
        else if (hnsw.levels[label] == hnsw.max_level) {
            // 현재 max_level에 다른 노드가 있는지 확인
            bool other_node_exists = false;
            for (storage_idx_t i = 0; i < ntotal; i++) {
                if (i != label && !is_deleted[i] && hnsw.levels[i] == hnsw.max_level) {
                    other_node_exists = true;
                    break;
                }
            }

            // 다른 노드가 없으면 max_level 감소
            if (!other_node_exists) {
                int new_max_level = hnsw.max_level - 1;
                while (new_max_level >= 0) {
                    bool level_has_nodes = false;
                    for (storage_idx_t i = 0; i < ntotal; i++) {
                        if (i != label && !is_deleted[i] && hnsw.levels[i] == new_max_level) {
                            level_has_nodes = true;
                            break;
                        }
                    }
                    if (level_has_nodes) break;
                    new_max_level--;
                }
                hnsw.max_level = std::max(0, new_max_level);
            }
        }

        // 모든 레벨에서 연결 제거
        int max_level = hnsw.levels[label];
        for (int level = 0; level <= max_level; level++) {
            size_t begin, end;
            hnsw.neighbor_range(label, level, &begin, &end);

            // 현재 레벨의 이웃들과의 연결 제거
            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor_id = hnsw.neighbors[i];
                if (neighbor_id < 0) continue;

                // 이웃 노드의 연결 제거
                size_t nb_begin, nb_end;
                hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                // 이웃 노드의 연결 목록에서 현재 노드 제거
                for (size_t j = nb_begin; j < nb_end; j++) {
                    if (hnsw.neighbors[j] == label) {
                        // 연결 제거 및 리스트 정리
                        for (size_t k = j + 1; k < nb_end; k++) {
                            if (hnsw.neighbors[k] < 0) break;
                            hnsw.neighbors[k-1] = hnsw.neighbors[k];
                        }
                        hnsw.neighbors[nb_end - 1] = -1;
                        break;
                    }
                }
            }

            // 현재 노드의 이웃 목록 초기화
            for (size_t i = begin; i < end; i++) {
                hnsw.neighbors[i] = -1;
            }
        }

        // level 정보 초기화
        hnsw.levels[label] = 0;

        // 삭제 표시
        is_deleted[label] = true;
    }

    void IndexHNSW::directDelete_clean(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_clean");
        }

        if (is_deleted[label]) {
            throw std::runtime_error("The requested to delete element is already deleted");
        }

        omp_set_lock(&locks[label]);

        try {
            // Top Level 처리
            if (label == hnsw.entry_point) {
                // 새로운 entry_point 찾기
                storage_idx_t new_entry_point = -1;
                int new_max_level = -1;

                // 현재 entry_point의 최상위 레벨에서 새로운 entry_point 찾기
                size_t begin, end;
                hnsw.neighbor_range(label, hnsw.max_level, &begin, &end);

                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;

                    if (new_entry_point == -1 || hnsw.levels[neighbor_id] > new_max_level) {
                        new_entry_point = neighbor_id;
                        new_max_level = hnsw.levels[neighbor_id];
                    }
                }

                // 최상위 레벨에 노드가 없으면 하위 레벨에서 찾기
                if (new_entry_point == -1) {
                    for (int level = hnsw.max_level - 1; level >= 0; level--) {
                        for (storage_idx_t i = 0; i < ntotal; i++) {
                            if (i != label && !is_deleted[i] && hnsw.levels[i] == level) {
                                new_entry_point = i;
                                new_max_level = level;
                                break;
                            }
                        }
                        if (new_entry_point != -1) break;
                    }
                }

                // entry_point 및 max_level 업데이트
                if (new_entry_point != -1) {
                    hnsw.entry_point = new_entry_point;
                    hnsw.max_level = new_max_level;
                } else {
                    // 그래프가 비어있게 되는 경우
                    hnsw.entry_point = -1;
                    hnsw.max_level = 0;
                }
            }
                // Top Level이 아닌 경우에도 max_level 조정이 필요할 수 있음
            else if (hnsw.levels[label] == hnsw.max_level) {
                // 현재 max_level에 다른 노드가 있는지 확인
                bool other_node_exists = false;
                for (storage_idx_t i = 0; i < ntotal; i++) {
                    if (i != label && !is_deleted[i] && hnsw.levels[i] == hnsw.max_level) {
                        other_node_exists = true;
                        break;
                    }
                }

                // 다른 노드가 없으면 max_level 감소
                if (!other_node_exists) {
                    int new_max_level = hnsw.max_level - 1;
                    while (new_max_level >= 0) {
                        bool level_has_nodes = false;
                        for (storage_idx_t i = 0; i < ntotal; i++) {
                            if (i != label && !is_deleted[i] && hnsw.levels[i] == new_max_level) {
                                level_has_nodes = true;
                                break;
                            }
                        }
                        if (level_has_nodes) break;
                        new_max_level--;
                    }
                    hnsw.max_level = std::max(0, new_max_level);
                }
            }

            // 모든 레벨에서 연결 제거
            int max_level = hnsw.levels[label];
            for (int level = 0; level <= max_level; level++) {  // <= 로 변경하여 최상위 레벨도 처리
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);

                // 현재 레벨의 이웃들과의 연결 제거
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0) continue;

                    // 이웃 노드의 연결 제거
                    omp_set_lock(&locks[neighbor_id]);

                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(neighbor_id, level, &nb_begin, &nb_end);

                    // 이웃 노드의 연결 목록에서 현재 노드 제거
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] == label) {
                            // 연결 제거 및 리스트 정리
                            for (size_t k = j + 1; k < nb_end; k++) {
                                if (hnsw.neighbors[k] < 0) break;
                                hnsw.neighbors[k-1] = hnsw.neighbors[k];
                            }
                            hnsw.neighbors[nb_end - 1] = -1;
                            break;
                        }
                    }

                    omp_unset_lock(&locks[neighbor_id]);
                }

                // 현재 노드의 이웃 목록 초기화
                for (size_t i = begin; i < end; i++) {
                    hnsw.neighbors[i] = -1;
                }
            }

            // level 정보 초기화
            hnsw.levels[label] = 0;

            // 삭제 표시
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::directDelete_knn_prune_nolock(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn_prune_nolock");
        }

        // Top Level 처리
        if (label == hnsw.entry_point) {
            // 새로운 entry_point 찾기
            storage_idx_t new_entry_point = -1;
            int new_max_level = -1;

            // 현재 entry_point의 최상위 레벨에서 새로운 entry_point 찾기
            size_t begin, end;
            hnsw.neighbor_range(label, hnsw.max_level, &begin, &end);

            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor_id = hnsw.neighbors[i];
                if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;

                if (new_entry_point == -1 || hnsw.levels[neighbor_id] > new_max_level) {
                    new_entry_point = neighbor_id;
                    new_max_level = hnsw.levels[neighbor_id];
                }
            }

            // 최상위 레벨에 노드가 없으면 하위 레벨에서 찾기
            if (new_entry_point == -1) {
                for (int level = hnsw.max_level - 1; level >= 0; level--) {
                    for (storage_idx_t i = 0; i < ntotal; i++) {
                        if (i != label && !is_deleted[i] && hnsw.levels[i] == level) {
                            new_entry_point = i;
                            new_max_level = level;
                            break;
                        }
                    }
                    if (new_entry_point != -1) break;
                }
            }

            // entry_point 및 max_level 업데이트
            if (new_entry_point != -1) {
                hnsw.entry_point = new_entry_point;
                hnsw.max_level = new_max_level;
            } else {
                // 그래프가 비어있게 되는 경우
                hnsw.entry_point = -1;
                hnsw.max_level = 0;
            }
        }
            // Top Level이 아닌 경우에도 max_level 조정이 필요할 수 있음
        else if (hnsw.levels[label] == hnsw.max_level) {
            // 현재 max_level에 다른 노드가 있는지 확인
            bool other_node_exists = false;
            for (storage_idx_t i = 0; i < ntotal; i++) {
                if (i != label && !is_deleted[i] && hnsw.levels[i] == hnsw.max_level) {
                    other_node_exists = true;
                    break;
                }
            }

            // 다른 노드가 없으면 max_level 감소
            if (!other_node_exists) {
                int new_max_level = hnsw.max_level - 1;
                while (new_max_level >= 0) {
                    bool level_has_nodes = false;
                    for (storage_idx_t i = 0; i < ntotal; i++) {
                        if (i != label && !is_deleted[i] && hnsw.levels[i] == new_max_level) {
                            level_has_nodes = true;
                            break;
                        }
                    }
                    if (level_has_nodes) break;
                    new_max_level--;
                }
                hnsw.max_level = std::max(0, new_max_level);
            }
        }

        // 모든 레벨에서 작업 수행 (Top Layer 포함)
        int max_level = hnsw.levels[label];
        for (int level = 0; level <= max_level; level++) {
            size_t begin, end;
            hnsw.neighbor_range(label, level, &begin, &end);
            std::vector<storage_idx_t> old_neighbors;

            // 현재 레벨의 이웃들 저장
            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor_id = hnsw.neighbors[i];
                if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                old_neighbors.push_back(neighbor_id);
            }

            // 이웃들 간의 연결 재구성
            reconnectNeighborsDirectDelete_knn_prune_nolock(label, old_neighbors, level);
        }

        // 노드 물리적 제거
        removePhysicalNode(label);
        is_deleted[label] = true;
    }




    void IndexHNSW::reconnectNeighborsDirectDelete_knn_prune_nolock(
            storage_idx_t node_id,
            std::vector<storage_idx_t>& old_neighbors,
            int level) {

        const size_t k = insert_maxM_;  // kNN search에서 찾을 이웃 수
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

        for (size_t i = 0; i < old_neighbors.size(); i++) {
            storage_idx_t current = old_neighbors[i];
            std::vector<float> current_vec(d);
            storage->reconstruct(current, current_vec.data());

            size_t nb_begin, nb_end;
            hnsw.neighbor_range(current, level, &nb_begin, &nb_end);

            // 삭제할 노드와의 연결 제거
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] == node_id) {
                    for (size_t k = j + 1; k < nb_end; k++) {
                        if (hnsw.neighbors[k] < 0) break;
                        hnsw.neighbors[k-1] = hnsw.neighbors[k];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                    break;
                }
            }

            // 후보 집합 구성: kNN 결과 + 기존 이웃들
            std::vector<std::pair<float, storage_idx_t>> candidates;

            // 1. kNN 검색 수행
            std::vector<float> distances(k);
            std::vector<idx_t> neighbors(k);

            // 현재 레벨에서만 검색하도록 임시로 max_level 조정
            int tmp_max_level = hnsw.max_level;
            hnsw.max_level = level;
            search(1, current_vec.data(), k, distances.data(), neighbors.data());
            hnsw.max_level = tmp_max_level;

            // kNN 결과를 후보에 추가
            for (size_t j = 0; j < k; j++) {
                idx_t neighbor_id = neighbors[j];
                if (neighbor_id == current || neighbor_id == node_id ||
                    neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                candidates.emplace_back(distances[j], neighbor_id);
            }

            // 2. 기존 이웃들도 후보에 추가
            for (size_t j = nb_begin; j < nb_end; j++) {
                storage_idx_t existing = hnsw.neighbors[j];
                if (existing == node_id || existing < 0 || is_deleted[existing]) continue;

                // 이미 후보에 있는지 확인
                bool already_added = false;
                for (const auto& candidate : candidates) {
                    if (candidate.second == existing) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added) {
                    float dist = dis->symmetric_dis(current, existing);
                    candidates.emplace_back(dist, existing);
                }
            }

            // 3. RobustPrune으로 최종 이웃 선택
            std::vector<std::pair<float, storage_idx_t>> pruned_candidates;
            std::vector<storage_idx_t> final_neighbors;

            // 거리 기준 정렬
            std::sort(candidates.begin(), candidates.end());

            // 가장 가까운 이웃은 무조건 추가 (검색 가능성 보장)
            if (!candidates.empty()) {
                pruned_candidates.push_back(candidates[0]);
                final_neighbors.push_back(candidates[0].second);

                // 나머지 후보들에 대해 RobustPrune 적용
                for (size_t j = 1; j < candidates.size() && pruned_candidates.size() < insert_maxM_; j++) {
                    const auto& current_candidate = candidates[j];
                    bool should_add = true;

                    // pruning 조건 체크
                    for (const auto& selected : pruned_candidates) {
                        float dist_between = dis->symmetric_dis(current_candidate.second, selected.second);
                        if (alpha * dist_between <= current_candidate.first) {
                            should_add = false;
                            break;
                        }
                    }

                    if (should_add) {
                        pruned_candidates.push_back(current_candidate);
                        final_neighbors.push_back(current_candidate.second);
                    }
                }
            }

            // 4. 최종 이웃 연결 설정
            size_t idx = nb_begin;
            for (auto neighbor_id : final_neighbors) {
                hnsw.neighbors[idx++] = neighbor_id;

                // 양방향 연결 설정
                size_t other_begin, other_end;
                hnsw.neighbor_range(neighbor_id, level, &other_begin, &other_end);

                bool connection_added = false;
                for (size_t j = other_begin; j < other_end; j++) {
                    if (hnsw.neighbors[j] < 0) {
                        hnsw.neighbors[j] = current;
                        connection_added = true;
                        break;
                    }
                }

                if (!connection_added) {
                    // 가장 먼 이웃 찾아서 대체 여부 결정
                    float max_dist = -1;
                    size_t max_idx = other_begin;
                    for (size_t j = other_begin; j < other_end; j++) {
                        if (hnsw.neighbors[j] >= 0) {
                            float dist = dis->symmetric_dis(neighbor_id, hnsw.neighbors[j]);
                            if (dist > max_dist) {
                                max_dist = dist;
                                max_idx = j;
                            }
                        }
                    }

                    float new_dist = dis->symmetric_dis(neighbor_id, current);
                    if (new_dist < max_dist) {
                        hnsw.neighbors[max_idx] = current;
                    }
                }
            }

            // 남은 슬롯 초기화
            while (idx < nb_end) {
                hnsw.neighbors[idx++] = -1;
            }
        }
    }

    void IndexHNSW::directDelete_knn_prune(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn_prune");
        }

        omp_set_lock(&locks[label]);

        try {
            // Top Level 처리
            if (label == hnsw.entry_point) {
                // 새로운 entry_point 찾기
                storage_idx_t new_entry_point = -1;
                int new_max_level = -1;

                // 현재 entry_point의 최상위 레벨에서 새로운 entry_point 찾기
                size_t begin, end;
                hnsw.neighbor_range(label, hnsw.max_level, &begin, &end);

                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;

                    if (new_entry_point == -1 || hnsw.levels[neighbor_id] > new_max_level) {
                        new_entry_point = neighbor_id;
                        new_max_level = hnsw.levels[neighbor_id];
                    }
                }

                // 최상위 레벨에 노드가 없으면 하위 레벨에서 찾기
                if (new_entry_point == -1) {
                    for (int level = hnsw.max_level - 1; level >= 0; level--) {
                        for (storage_idx_t i = 0; i < ntotal; i++) {
                            if (i != label && !is_deleted[i] && hnsw.levels[i] == level) {
                                new_entry_point = i;
                                new_max_level = level;
                                break;
                            }
                        }
                        if (new_entry_point != -1) break;
                    }
                }

                // entry_point 및 max_level 업데이트
                if (new_entry_point != -1) {
                    hnsw.entry_point = new_entry_point;
                    hnsw.max_level = new_max_level;
                } else {
                    // 그래프가 비어있게 되는 경우
                    hnsw.entry_point = -1;
                    hnsw.max_level = 0;
                }
            }
                // Top Level이 아닌 경우에도 max_level 조정이 필요할 수 있음
            else if (hnsw.levels[label] == hnsw.max_level) {
                // 현재 max_level에 다른 노드가 있는지 확인
                bool other_node_exists = false;
                for (storage_idx_t i = 0; i < ntotal; i++) {
                    if (i != label && !is_deleted[i] && hnsw.levels[i] == hnsw.max_level) {
                        other_node_exists = true;
                        break;
                    }
                }

                // 다른 노드가 없으면 max_level 감소
                if (!other_node_exists) {
                    int new_max_level = hnsw.max_level - 1;
                    while (new_max_level >= 0) {
                        bool level_has_nodes = false;
                        for (storage_idx_t i = 0; i < ntotal; i++) {
                            if (i != label && !is_deleted[i] && hnsw.levels[i] == new_max_level) {
                                level_has_nodes = true;
                                break;
                            }
                        }
                        if (level_has_nodes) break;
                        new_max_level--;
                    }
                    hnsw.max_level = std::max(0, new_max_level);
                }
            }

            // 모든 레벨에서 작업 수행 (Top Layer 포함)
            int max_level = hnsw.levels[label];
            for (int level = 0; level <= max_level; level++) {  // <= 로 변경하여 최상위 레벨도 처리
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);
                std::vector<storage_idx_t> old_neighbors;

                // 현재 레벨의 이웃들 저장
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                    old_neighbors.push_back(neighbor_id);
                }

                // 이웃들 간의 연결 재구성
                reconnectNeighborsDirectDelete_knn_prune(label, old_neighbors, level);
            }

            // 노드 물리적 제거
            removePhysicalNode(label);
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    /* 기존에 잘 돌던 함수
    void IndexHNSW::directDelete_knn_prune(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn_prune");
        }

        omp_set_lock(&locks[label]);

        try {
            // 모든 레벨에서 작업 수행 (Top Layer 제외)
            int max_level = hnsw.levels[label];
            for (int level = 0; level < max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);
                std::vector<storage_idx_t> old_neighbors;

                // 현재 레벨의 이웃들 저장
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                    old_neighbors.push_back(neighbor_id);
                }

                // 이웃들 간의 연결 재구성
                reconnectNeighborsDirectDelete_knn_prune(label, old_neighbors, level);
            }

            // 노드 물리적 제거
            removePhysicalNode(label);
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }
*/

    void IndexHNSW::reconnectNeighborsDirectDelete_knn_prune(
            storage_idx_t node_id,
            std::vector<storage_idx_t>& old_neighbors,
            int level) {

        const size_t k = insert_maxM_;  // kNN search에서 찾을 이웃 수
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

        for (size_t i = 0; i < old_neighbors.size(); i++) {
            storage_idx_t current = old_neighbors[i];
            omp_set_lock(&locks[current]);

            std::vector<float> current_vec(d);
            storage->reconstruct(current, current_vec.data());

            size_t nb_begin, nb_end;
            hnsw.neighbor_range(current, level, &nb_begin, &nb_end);

            // 삭제할 노드와의 연결 제거
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] == node_id) {
                    for (size_t k = j + 1; k < nb_end; k++) {
                        if (hnsw.neighbors[k] < 0) break;
                        hnsw.neighbors[k-1] = hnsw.neighbors[k];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                    break;
                }
            }

            // 후보 집합 구성: kNN 결과 + 기존 이웃들
            std::vector<std::pair<float, storage_idx_t>> candidates;

            // 1. kNN 검색 수행
            std::vector<float> distances(k);
            std::vector<idx_t> neighbors(k);

            // 현재 레벨에서만 검색하도록 임시로 max_level 조정
            int tmp_max_level = hnsw.max_level;
            hnsw.max_level = level;
            search(1, current_vec.data(), k, distances.data(), neighbors.data());
            hnsw.max_level = tmp_max_level;

            // kNN 결과를 후보에 추가
            for (size_t j = 0; j < k; j++) {
                idx_t neighbor_id = neighbors[j];
                if (neighbor_id == current || neighbor_id == node_id ||
                    neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                candidates.emplace_back(distances[j], neighbor_id);
            }

            // 2. 기존 이웃들도 후보에 추가
            for (size_t j = nb_begin; j < nb_end; j++) {
                storage_idx_t existing = hnsw.neighbors[j];
                if (existing == node_id || existing < 0 || is_deleted[existing]) continue;

                // 이미 후보에 있는지 확인
                bool already_added = false;
                for (const auto& candidate : candidates) {
                    if (candidate.second == existing) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added) {
                    float dist = dis->symmetric_dis(current, existing);
                    candidates.emplace_back(dist, existing);
                }
            }

            // 3. RobustPrune으로 최종 이웃 선택
            std::vector<std::pair<float, storage_idx_t>> pruned_candidates;
            std::vector<storage_idx_t> final_neighbors;

            // 거리 기준 정렬
            std::sort(candidates.begin(), candidates.end());

            // 가장 가까운 이웃은 무조건 추가 (검색 가능성 보장)
            if (!candidates.empty()) {
                pruned_candidates.push_back(candidates[0]);
                final_neighbors.push_back(candidates[0].second);

                // 나머지 후보들에 대해 RobustPrune 적용
                for (size_t j = 1; j < candidates.size() && pruned_candidates.size() < insert_maxM_; j++) {
                    const auto& current_candidate = candidates[j];
                    bool should_add = true;

                    // pruning 조건 체크
                    for (const auto& selected : pruned_candidates) {
                        float dist_between = dis->symmetric_dis(current_candidate.second, selected.second);
                        if (alpha * dist_between <= current_candidate.first) {
                            should_add = false;
                            break;
                        }
                    }

                    if (should_add) {
                        pruned_candidates.push_back(current_candidate);
                        final_neighbors.push_back(current_candidate.second);
                    }
                }
            }

            // 4. 최종 이웃 연결 설정
            size_t idx = nb_begin;
            for (auto neighbor_id : final_neighbors) {
                hnsw.neighbors[idx++] = neighbor_id;

                // 양방향 연결 설정
                omp_set_lock(&locks[neighbor_id]);
                size_t other_begin, other_end;
                hnsw.neighbor_range(neighbor_id, level, &other_begin, &other_end);

                bool connection_added = false;
                for (size_t j = other_begin; j < other_end; j++) {
                    if (hnsw.neighbors[j] < 0) {
                        hnsw.neighbors[j] = current;
                        connection_added = true;
                        break;
                    }
                }

                if (!connection_added) {
                    // 가장 먼 이웃 찾아서 대체 여부 결정
                    float max_dist = -1;
                    size_t max_idx = other_begin;
                    for (size_t j = other_begin; j < other_end; j++) {
                        if (hnsw.neighbors[j] >= 0) {
                            float dist = dis->symmetric_dis(neighbor_id, hnsw.neighbors[j]);
                            if (dist > max_dist) {
                                max_dist = dist;
                                max_idx = j;
                            }
                        }
                    }

                    float new_dist = dis->symmetric_dis(neighbor_id, current);
                    if (new_dist < max_dist) {
                        hnsw.neighbors[max_idx] = current;
                    }
                }
                omp_unset_lock(&locks[neighbor_id]);
            }

            // 남은 슬롯 초기화
            while (idx < nb_end) {
                hnsw.neighbors[idx++] = -1;
            }

            omp_unset_lock(&locks[current]);
        }
    }

/*
    void IndexHNSW::directDelete_knn_prune(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn_prune");
        }

        if (is_deleted[label]) {
            throw std::runtime_error("The requested to delete element is already deleted");
        }

        omp_set_lock(&locks[label]);

        try {
            // Top Level 처리
            if (label == hnsw.entry_point) {
                // 새로운 entry_point 찾기
                storage_idx_t new_entry_point = -1;
                int new_max_level = -1;

                // 현재 entry_point의 최상위 레벨에서 새로운 entry_point 찾기
                size_t begin, end;
                hnsw.neighbor_range(label, hnsw.max_level, &begin, &end);

                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;

                    if (new_entry_point == -1 || hnsw.levels[neighbor_id] > new_max_level) {
                        new_entry_point = neighbor_id;
                        new_max_level = hnsw.levels[neighbor_id];
                    }
                }

                // 최상위 레벨에 노드가 없으면 하위 레벨에서 찾기
                if (new_entry_point == -1) {
                    for (int level = hnsw.max_level - 1; level >= 0; level--) {
                        for (storage_idx_t i = 0; i < ntotal; i++) {
                            if (i != label && !is_deleted[i] && hnsw.levels[i] == level) {
                                new_entry_point = i;
                                new_max_level = level;
                                break;
                            }
                        }
                        if (new_entry_point != -1) break;
                    }
                }

                // entry_point 및 max_level 업데이트
                if (new_entry_point != -1) {
                    hnsw.entry_point = new_entry_point;
                    hnsw.max_level = new_max_level;
                } else {
                    // 그래프가 비어있게 되는 경우
                    hnsw.entry_point = -1;
                    hnsw.max_level = 0;
                }
            }
                // Top Level이 아닌 경우에도 max_level 조정이 필요할 수 있음
            else if (hnsw.levels[label] == hnsw.max_level) {
                // 현재 max_level에 다른 노드가 있는지 확인
                bool other_node_exists = false;
                for (storage_idx_t i = 0; i < ntotal; i++) {
                    if (i != label && !is_deleted[i] && hnsw.levels[i] == hnsw.max_level) {
                        other_node_exists = true;
                        break;
                    }
                }

                // 다른 노드가 없으면 max_level 감소
                if (!other_node_exists) {
                    int new_max_level = hnsw.max_level - 1;
                    while (new_max_level >= 0) {
                        bool level_has_nodes = false;
                        for (storage_idx_t i = 0; i < ntotal; i++) {
                            if (i != label && !is_deleted[i] && hnsw.levels[i] == new_max_level) {
                                level_has_nodes = true;
                                break;
                            }
                        }
                        if (level_has_nodes) break;
                        new_max_level--;
                    }
                    hnsw.max_level = std::max(0, new_max_level);
                }
            }

            // 모든 레벨에서 이웃 재구성 수행 (Top Level 포함)
            int max_level = hnsw.levels[label];
            for (int level = 0; level <= max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);
                std::vector<storage_idx_t> old_neighbors;

                // 현재 레벨의 이웃들 저장
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                    old_neighbors.push_back(neighbor_id);
                }

                // 이웃들 간의 연결 재구성
                reconnectNeighborsDirectDelete_knn_prune(label, old_neighbors, level);

                // 현재 노드의 이웃 목록 초기화
                for (size_t i = begin; i < end; i++) {
                    hnsw.neighbors[i] = -1;
                }
            }

            // level 정보 초기화
            hnsw.levels[label] = 0;

            // 노드 물리적 제거
            removePhysicalNode(label);
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }
    */

/*
    void IndexHNSW::directDelete_knn_prune(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn_prune");
        }

        omp_set_lock(&locks[label]);

        try {
            // 모든 레벨에서 작업 수행 (Top Layer 제외)
            int max_level = hnsw.levels[label];
            for (int level = 0; level < max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);
                std::vector<storage_idx_t> old_neighbors;

                // 현재 레벨의 이웃들 저장
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t neighbor_id = hnsw.neighbors[i];
                    if (neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                    old_neighbors.push_back(neighbor_id);
                }

                // 이웃들 간의 연결 재구성
                reconnectNeighborsDirectDelete_knn_prune(label, old_neighbors, level);
            }

            // 노드 물리적 제거
            removePhysicalNode(label);
            is_deleted[label] = true;

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::reconnectNeighborsDirectDelete_knn_prune(
            storage_idx_t node_id,
            std::vector<storage_idx_t>& old_neighbors,
            int level) {

        const size_t k = insert_maxM_;  // kNN search에서 찾을 이웃 수
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

        for (size_t i = 0; i < old_neighbors.size(); i++) {
            storage_idx_t current = old_neighbors[i];
            omp_set_lock(&locks[current]);

            std::vector<float> current_vec(d);
            storage->reconstruct(current, current_vec.data());

            size_t nb_begin, nb_end;
            hnsw.neighbor_range(current, level, &nb_begin, &nb_end);

            // 삭제할 노드와의 연결 제거
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] == node_id) {
                    for (size_t k = j + 1; k < nb_end; k++) {
                        if (hnsw.neighbors[k] < 0) break;
                        hnsw.neighbors[k-1] = hnsw.neighbors[k];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                    break;
                }
            }

            // 후보 집합 구성: kNN 결과 + 기존 이웃들
            std::vector<std::pair<float, storage_idx_t>> candidates;

            // 1. kNN 검색 수행
            std::vector<float> distances(k);
            std::vector<idx_t> neighbors(k);

            // 현재 레벨에서만 검색하도록 임시로 max_level 조정
            int tmp_max_level = hnsw.max_level;
            hnsw.max_level = level;
            search(1, current_vec.data(), k, distances.data(), neighbors.data());
            hnsw.max_level = tmp_max_level;

            // kNN 결과를 후보에 추가
            for (size_t j = 0; j < k; j++) {
                idx_t neighbor_id = neighbors[j];
                if (neighbor_id == current || neighbor_id == node_id ||
                    neighbor_id < 0 || is_deleted[neighbor_id]) continue;
                candidates.emplace_back(distances[j], neighbor_id);
            }

            // 2. 기존 이웃들도 후보에 추가
            for (size_t j = nb_begin; j < nb_end; j++) {
                storage_idx_t existing = hnsw.neighbors[j];
                if (existing == node_id || existing < 0 || is_deleted[existing]) continue;

                // 이미 후보에 있는지 확인
                bool already_added = false;
                for (const auto& candidate : candidates) {
                    if (candidate.second == existing) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added) {
                    float dist = dis->symmetric_dis(current, existing);
                    candidates.emplace_back(dist, existing);
                }
            }

            // 3. RobustPrune으로 최종 이웃 선택
            std::vector<std::pair<float, storage_idx_t>> pruned_candidates;
            std::vector<storage_idx_t> final_neighbors;

            // 거리 기준 정렬
            std::sort(candidates.begin(), candidates.end());

            // 가장 가까운 이웃은 무조건 추가 (검색 가능성 보장)
            if (!candidates.empty()) {
                pruned_candidates.push_back(candidates[0]);
                final_neighbors.push_back(candidates[0].second);

                // 나머지 후보들에 대해 RobustPrune 적용
                for (size_t j = 1; j < candidates.size() && pruned_candidates.size() < insert_maxM_; j++) {
                    const auto& current_candidate = candidates[j];
                    bool should_add = true;

                    // pruning 조건 체크
                    for (const auto& selected : pruned_candidates) {
                        float dist_between = dis->symmetric_dis(current_candidate.second, selected.second);
                        if (alpha * dist_between <= current_candidate.first) {
                            should_add = false;
                            break;
                        }
                    }

                    if (should_add) {
                        pruned_candidates.push_back(current_candidate);
                        final_neighbors.push_back(current_candidate.second);
                    }
                }
            }

            // 4. 최종 이웃 연결 설정
            size_t idx = nb_begin;
            for (auto neighbor_id : final_neighbors) {
                hnsw.neighbors[idx++] = neighbor_id;

                // 양방향 연결 설정
                omp_set_lock(&locks[neighbor_id]);
                size_t other_begin, other_end;
                hnsw.neighbor_range(neighbor_id, level, &other_begin, &other_end);

                bool connection_added = false;
                for (size_t j = other_begin; j < other_end; j++) {
                    if (hnsw.neighbors[j] < 0) {
                        hnsw.neighbors[j] = current;
                        connection_added = true;
                        break;
                    }
                }

                if (!connection_added) {
                    // 가장 먼 이웃 찾아서 대체 여부 결정
                    float max_dist = -1;
                    size_t max_idx = other_begin;
                    for (size_t j = other_begin; j < other_end; j++) {
                        if (hnsw.neighbors[j] >= 0) {
                            float dist = dis->symmetric_dis(neighbor_id, hnsw.neighbors[j]);
                            if (dist > max_dist) {
                                max_dist = dist;
                                max_idx = j;
                            }
                        }
                    }

                    float new_dist = dis->symmetric_dis(neighbor_id, current);
                    if (new_dist < max_dist) {
                        hnsw.neighbors[max_idx] = current;
                    }
                }
                omp_unset_lock(&locks[neighbor_id]);
            }

            // 남은 슬롯 초기화
            while (idx < nb_end) {
                hnsw.neighbors[idx++] = -1;
            }

            omp_unset_lock(&locks[current]);
        }
    }
*/
    void IndexHNSW::directDelete_knn(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete_knn");
        }

        omp_set_lock(&locks[label]);

        try {
            // 레벨 0의 이웃 정보 저장
            size_t begin, end;
            hnsw.neighbor_range(label, 0, &begin, &end);
            std::vector<storage_idx_t> old_neighbors;

            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor_id = hnsw.neighbors[i];
                if (neighbor_id < 0) break;
                old_neighbors.push_back(neighbor_id);
            }

            // 이웃들 간의 연결 재구성
            reconnectNeighborsDirectDelete_knn(label, old_neighbors);

            // 노드 물리적 제거
            removePhysicalNode(label);

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::reconnectNeighborsDirectDelete_knn(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors) {
        const size_t k = 10;              // kNN 검색에서 사용할 k값
        hnsw.efSearch = 10;               // efSearch 값 설정

        for (size_t i = 0; i < old_neighbors.size(); i++) {
            storage_idx_t current = old_neighbors[i];
            omp_set_lock(&locks[current]);

            // 현재 노드의 벡터 가져오기
            std::vector<float> current_vec(d);
            storage->reconstruct(current, current_vec.data());

            // 현재 노드에서 삭제할 노드로의 연결 제거
            size_t nb_begin, nb_end;
            hnsw.neighbor_range(current, 0, &nb_begin, &nb_end);

            // 현재 존재하는 이웃 수 확인
            std::vector<storage_idx_t> existing_neighbors;
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] == node_id) {
                    // 삭제할 노드와의 연결 제거
                    for (size_t k = j + 1; k < nb_end; k++) {
                        if (hnsw.neighbors[k] < 0) break;
                        hnsw.neighbors[k-1] = hnsw.neighbors[k];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                } else if (hnsw.neighbors[j] >= 0 && hnsw.neighbors[j] != node_id) {
                    existing_neighbors.push_back(hnsw.neighbors[j]);
                }
            }

            // kNN 검색 수행
            std::vector<float> distances(k);
            std::vector<idx_t> neighbors(k);
            search(1, current_vec.data(), k, distances.data(), neighbors.data());

            std::priority_queue<std::pair<float, storage_idx_t>> candidates;
            for (size_t j = 0; j < k; j++) {
                idx_t neighbor_id = neighbors[j];
                if (neighbor_id == current || neighbor_id == node_id ||
                    neighbor_id < 0 || is_deleted[neighbor_id])
                    continue;
                candidates.emplace(distances[j], neighbor_id);
            }

            // 이웃이 하나도 없는 경우, 첫 번째 후보는 무조건 연결
            if (existing_neighbors.empty() && !candidates.empty()) {
                auto first_candidate = candidates.top();
                storage_idx_t new_neighbor = first_candidate.second;
                candidates.pop();

                // 현재 노드에 첫 번째 후보 연결
                for (size_t j = nb_begin; j < nb_end; j++) {
                    if (hnsw.neighbors[j] < 0) {
                        hnsw.neighbors[j] = new_neighbor;

                        // 양방향 연결
                        omp_set_lock(&locks[new_neighbor]);
                        size_t other_begin, other_end;
                        hnsw.neighbor_range(new_neighbor, 0, &other_begin, &other_end);
                        for (size_t k = other_begin; k < other_end; k++) {
                            if (hnsw.neighbors[k] < 0) {
                                hnsw.neighbors[k] = current;
                                break;
                            }
                        }
                        omp_unset_lock(&locks[new_neighbor]);
                        break;
                    }
                }
            }

            // 나머지 후보들에 대해 휴리스틱 적용
            if (!candidates.empty()) {
                std::priority_queue<std::pair<float, storage_idx_t>> remaining_candidates;
                while (!candidates.empty()) {
                    auto candidate = candidates.top();
                    storage_idx_t new_neighbor = candidate.second;
                    float dist_to_query = candidate.first;
                    candidates.pop();

                    bool should_connect = true;
                    // 현재 연결된 이웃들과 비교
                    for (auto existing : existing_neighbors) {
                        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
                        float existing_dist = dis->symmetric_dis(current, existing);
                        if (alpha * existing_dist < dist_to_query) {
                            should_connect = false;
                            break;
                        }
                    }

                    if (should_connect) {
                        remaining_candidates.push(candidate);
                    }
                }

                // 연결 설정
                while (!remaining_candidates.empty()) {
                    auto candidate = remaining_candidates.top();
                    storage_idx_t new_neighbor = candidate.second;
                    remaining_candidates.pop();

                    // 현재 노드에서 새 이웃으로 연결
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] < 0) {
                            hnsw.neighbors[j] = new_neighbor;

                            // 양방향 연결
                            omp_set_lock(&locks[new_neighbor]);
                            size_t other_begin, other_end;
                            hnsw.neighbor_range(new_neighbor, 0, &other_begin, &other_end);
                            for (size_t k = other_begin; k < other_end; k++) {
                                if (hnsw.neighbors[k] < 0) {
                                    hnsw.neighbors[k] = current;
                                    break;
                                }
                            }
                            omp_unset_lock(&locks[new_neighbor]);
                            break;
                        }
                    }
                }
            }

            omp_unset_lock(&locks[current]);
        }
    }


    void IndexHNSW::directDelete(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in directDelete");
        }

        // 노드 락 획득
        omp_set_lock(&locks[label]);

        try {
            // 레벨 0의 이웃 정보 저장
            size_t begin, end;
            hnsw.neighbor_range(label, 0, &begin, &end);
            std::vector<storage_idx_t> old_neighbors;

            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor_id = hnsw.neighbors[i];
                if (neighbor_id < 0) break;
                old_neighbors.push_back(neighbor_id);
            }

            // 이웃들 간의 연결 재구성
            reconnectNeighborsDirectDelete(label, old_neighbors);

            // 노드 물리적 제거
            removePhysicalNode(label);

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::reconnectNeighborsDirectDelete(storage_idx_t node_id, std::vector<storage_idx_t>& old_neighbors) {
        for (size_t i = 0; i < old_neighbors.size(); i++) {
            storage_idx_t current = old_neighbors[i];
            omp_set_lock(&locks[current]);

            // 현재 노드에서 삭제할 노드로의 연결 제거
            size_t nb_begin, nb_end;
            hnsw.neighbor_range(current, 0, &nb_begin, &nb_end);

            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] == node_id) {
                    // 연결 제거 및 리스트 정리
                    for (size_t k = j + 1; k < nb_end; k++) {
                        if (hnsw.neighbors[k] < 0) break;
                        hnsw.neighbors[k-1] = hnsw.neighbors[k];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                    break;
                }
            }

            // 휴리스틱 기반으로 이웃 재연결
            reconnectNeighborsByHeuristic_delete(current, old_neighbors, hnsw.nb_neighbors(0));

            omp_unset_lock(&locks[current]);
        }
    }

    void IndexHNSW::removePhysicalNode(storage_idx_t node_id) {
        // Level 0에서 노드의 모든 연결 제거
        size_t begin, end;
        hnsw.neighbor_range(node_id, 0, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            hnsw.neighbors[i] = -1;
        }

        // 노드의 level 정보 초기화
        hnsw.levels[node_id] = 0;
    }


    void IndexHNSW::markDelete(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in markDelete");
        }

        omp_set_lock(&locks[label]);

        if (is_deleted[label]) {
            omp_unset_lock(&locks[label]);
            throw std::runtime_error("The requested to delete element is already deleted");
        }

        // 노드를 삭제 표시
        is_deleted[label] = true;

        // 레벨 0의 이웃 연결 제거
        size_t begin, end;
        hnsw.neighbor_range(label, 0, &begin, &end);

        // 이웃 노드들과의 연결 제거
        for (size_t j = begin; j < end; j++) {
            storage_idx_t neighbor_id = hnsw.neighbors[j];
            if (neighbor_id < 0) break;

            omp_set_lock(&locks[neighbor_id]);

            // 이웃 노드의 연결 목록에서 현재 노드 제거
            size_t nb_begin, nb_end;
            hnsw.neighbor_range(neighbor_id, 0, &nb_begin, &nb_end);

            for (size_t k = nb_begin; k < nb_end; k++) {
                if (hnsw.neighbors[k] == label) {
                    // 삭제된 노드를 연결 목록에서 제거
                    for (size_t l = k + 1; l < nb_end; l++) {
                        if (hnsw.neighbors[l] < 0) break;
                        hnsw.neighbors[l-1] = hnsw.neighbors[l];
                    }
                    hnsw.neighbors[nb_end - 1] = -1;
                    break;
                }
            }

            omp_unset_lock(&locks[neighbor_id]);
        }

        // 현재 노드의 이웃 목록 초기화
        for (size_t j = begin; j < end; j++) {
            hnsw.neighbors[j] = -1;
        }

        omp_unset_lock(&locks[label]);
    }

    void IndexHNSW::reconnectNeighbors(storage_idx_t node_id) {
        std::vector<float> node_vec(d);
        storage->reconstruct(node_id, node_vec.data());

        size_t begin, end;
        hnsw.neighbor_range(node_id, 0, &begin, &end);

        std::vector<storage_idx_t> old_neighbors;
        for (size_t i = begin; i < end; i++) {
            storage_idx_t neighbor_id = hnsw.neighbors[i];
            if (neighbor_id < 0) break;
            old_neighbors.push_back(neighbor_id);
        }

        // 각 이웃에 대해 새로운 연결 설정
        for (storage_idx_t neighbor_id : old_neighbors) {
            omp_set_lock(&locks[neighbor_id]);

            std::vector<float> neighbor_vec(d);
            storage->reconstruct(neighbor_id, neighbor_vec.data());

            // 후보 이웃 찾기
            std::priority_queue<std::pair<float, storage_idx_t>> candidates;

            for (storage_idx_t other_neighbor : old_neighbors) {
                if (other_neighbor == neighbor_id) continue;

                std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
                float distance = dis->symmetric_dis(neighbor_id, other_neighbor);
                candidates.emplace(distance, other_neighbor);
            }

            // 새로운 이웃 선택
            getNeighborsByHeuristic2_direct_delete(candidates, hnsw.nb_neighbors(0));

            // 이웃 리스트 업데이트
            size_t nb_begin, nb_end;
            hnsw.neighbor_range(neighbor_id, 0, &nb_begin, &nb_end);

            size_t idx = nb_begin;
            while (!candidates.empty() && idx < nb_end) {
                storage_idx_t new_neighbor = candidates.top().second;
                candidates.pop();
                hnsw.neighbors[idx++] = new_neighbor;
            }

            while (idx < nb_end) {
                hnsw.neighbors[idx++] = -1;
            }

            omp_unset_lock(&locks[neighbor_id]);
        }
    }

    void IndexHNSW::insertPoint_knn(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        omp_set_lock(&locks[label]);

        try {
            // 1. 레벨 할당
            int insert_level = hnsw.levels[label];
            if (insert_level <= 0) {
                insert_level = getRandomLevel(1.0 / log(1.0 * M_));
                hnsw.levels[label] = insert_level;
            }

            // 2. 진입점 찾기
            storage_idx_t currObj = hnsw.entry_point;
            std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
            dis->set_query(point_vec.data());
            float curr_dist = dis->symmetric_dis(currObj, label);

            // 상위 레벨부터 시작하여 좋은 진입점 찾기
            for (int level = hnsw.max_level; level > insert_level; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;

                    size_t begin, end;
                    hnsw.neighbor_range(currObj, level, &begin, &end);

                    for (size_t idx = begin; idx < end; idx++) {
                        storage_idx_t neighbor = hnsw.neighbors[idx];
                        if (neighbor < 0 || is_deleted[neighbor]) continue;

                        float dist = dis->symmetric_dis(neighbor, label);
                        if (dist < curr_dist) {
                            curr_dist = dist;
                            currObj = neighbor;
                            changed = true;
                        }
                    }
                }
            }

            // 3. 각 레벨에서 이웃 연결
            for (int level = std::min(insert_level, hnsw.max_level); level >= 0; level--) {
                std::priority_queue<std::pair<float, storage_idx_t>> candidates;
                const size_t ef_construction = std::max(size_t(M_), size_t(40));

                candidates.emplace(curr_dist, currObj);
                std::vector<bool> visited(ntotal, false);
                visited[currObj] = true;

                std::priority_queue<std::pair<float, storage_idx_t>> top_candidates;
                size_t candidate_count = 0;

                while (!candidates.empty() && candidate_count < ef_construction) {
                    const auto current = candidates.top();
                    float dist_to_query = current.first;
                    storage_idx_t current_node = current.second;
                    candidates.pop();

                    if (!top_candidates.empty() && dist_to_query > top_candidates.top().first) {
                        break;
                    }

                    size_t begin, end;
                    hnsw.neighbor_range(current_node, level, &begin, &end);
                    end = begin + (level == 0 ? insert_maxM0_ : insert_maxM_);

                    for (size_t idx = begin; idx < end; idx++) {
                        storage_idx_t neighbor = hnsw.neighbors[idx];
                        if (neighbor < 0 || is_deleted[neighbor] || visited[neighbor]) continue;

                        visited[neighbor] = true;
                        float dist = dis->symmetric_dis(neighbor, label);

                        if (top_candidates.size() < ef_construction || dist < top_candidates.top().first) {
                            candidates.emplace(dist, neighbor);
                            top_candidates.emplace(dist, neighbor);
                            candidate_count++;

                            if (top_candidates.size() > ef_construction) {
                                top_candidates.pop();
                            }
                        }
                    }
                }

                // 휴리스틱으로 이웃 선택
                size_t Mcurmax = level == 0 ? insert_maxM0_ : insert_maxM_;
                getNeighborsByHeuristic2_insert(top_candidates, Mcurmax);

                // 선택된 이웃과 양방향 연결 설정
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);
                end = begin + (level == 0 ? insert_maxM0_ : insert_maxM_);

                size_t idx = begin;
                std::vector<storage_idx_t> selected_neighbors;

                while (!top_candidates.empty() && idx < end) {
                    auto candidate = top_candidates.top();
                    storage_idx_t selected_neighbor = candidate.second;
                    top_candidates.pop();

                    if (selected_neighbor == label || is_deleted[selected_neighbor]) continue;

                    // 현재 노드에서 이웃으로 연결
                    hnsw.neighbors[idx] = selected_neighbor;
                    selected_neighbors.push_back(selected_neighbor);
                    idx++;

                    // 이웃에서 현재 노드로의 연결
                    omp_set_lock(&locks[selected_neighbor]);

                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(selected_neighbor, level, &nb_begin, &nb_end);
                    nb_end = nb_begin + (level == 0 ? insert_maxM0_ : insert_maxM_);

                    bool connection_added = false;
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] < 0) {
                            hnsw.neighbors[j] = label;
                            connection_added = true;
                            break;
                        }
                    }

                    // 가득 찬 경우 거리 기반 대체
                    if (!connection_added) {
                        float max_dist = -1;
                        size_t max_idx = nb_begin;
                        for (size_t j = nb_begin; j < nb_end; j++) {
                            float dist = dis->symmetric_dis(selected_neighbor, hnsw.neighbors[j]);
                            if (dist > max_dist) {
                                max_dist = dist;
                                max_idx = j;
                            }
                        }

                        float new_dist = dis->symmetric_dis(selected_neighbor, label);
                        if (new_dist < max_dist) {
                            hnsw.neighbors[max_idx] = label;
                        }
                    }

                    omp_unset_lock(&locks[selected_neighbor]);
                }

                // 남은 슬롯 초기화
                while (idx < end) {
                    hnsw.neighbors[idx++] = -1;
                }
            }

            is_deleted[label] = false;

            if (insert_level > hnsw.max_level) {
                hnsw.max_level = insert_level;
                hnsw.entry_point = label;
            }

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }


    /*
    void IndexHNSW::insertPoint_knn(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        omp_set_lock(&locks[label]);

        try {
            // 1. 레벨 할당
            int insert_level = hnsw.levels[label];
            if (insert_level <= 0) {
                insert_level = getRandomLevel(1.0 / log(1.0 * M_));
                hnsw.levels[label] = insert_level;
            }

            // 2. 진입점 찾기
            storage_idx_t currObj = hnsw.entry_point;
            std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
            dis->set_query(point_vec.data());
            float curr_dist = dis->symmetric_dis(currObj, label);

            // 상위 레벨부터 시작하여 좋은 진입점 찾기
            for (int level = hnsw.max_level; level > insert_level; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;

                    size_t begin, end;
                    hnsw.neighbor_range(currObj, level, &begin, &end);

                    for (size_t idx = begin; idx < end; idx++) {
                        storage_idx_t neighbor = hnsw.neighbors[idx];
                        if (neighbor < 0 || is_deleted[neighbor]) continue;

                        float dist = dis->symmetric_dis(neighbor, label);
                        if (dist < curr_dist) {
                            curr_dist = dist;
                            currObj = neighbor;
                            changed = true;
                        }
                    }
                }
            }

            // 3. 각 레벨에서 이웃 연결
            for (int level = std::min(insert_level, hnsw.max_level); level >= 0; level--) {
                std::priority_queue<std::pair<float, storage_idx_t>> candidates;
                const size_t ef_construction = std::max(size_t(M_), size_t(40));

                candidates.emplace(curr_dist, currObj);
                std::vector<bool> visited(ntotal, false);
                visited[currObj] = true;

                std::priority_queue<std::pair<float, storage_idx_t>> top_candidates;
                size_t candidate_count = 0;

                while (!candidates.empty() && candidate_count < ef_construction) {
                    const auto current = candidates.top();
                    float dist_to_query = current.first;
                    storage_idx_t current_node = current.second;
                    candidates.pop();

                    if (!top_candidates.empty() && dist_to_query > top_candidates.top().first) {
                        break;
                    }

                    size_t begin, end;
                    hnsw.neighbor_range(current_node, level, &begin, &end);

                    for (size_t idx = begin; idx < end; idx++) {
                        storage_idx_t neighbor = hnsw.neighbors[idx];
                        if (neighbor < 0 || is_deleted[neighbor] || visited[neighbor]) continue;

                        visited[neighbor] = true;
                        float dist = dis->symmetric_dis(neighbor, label);

                        if (top_candidates.size() < ef_construction || dist < top_candidates.top().first) {
                            candidates.emplace(dist, neighbor);
                            top_candidates.emplace(dist, neighbor);
                            candidate_count++;

                            if (top_candidates.size() > ef_construction) {
                                top_candidates.pop();
                            }
                        }
                    }
                }

                // 휴리스틱으로 이웃 선택
                size_t Mcurmax = level == 0 ? insert_maxM0_ : insert_maxM_;
                getNeighborsByHeuristic2_insert(top_candidates, Mcurmax);

                // 선택된 이웃과 양방향 연결 설정
                size_t begin, end;
                hnsw.neighbor_range(label, level, &begin, &end);

                size_t idx = begin;
                std::vector<storage_idx_t> selected_neighbors;

                while (!top_candidates.empty() && idx < end) {
                    auto candidate = top_candidates.top();
                    storage_idx_t selected_neighbor = candidate.second;
                    top_candidates.pop();

                    if (selected_neighbor == label || is_deleted[selected_neighbor]) continue;

                    // 현재 노드에서 이웃으로 연결
                    hnsw.neighbors[idx] = selected_neighbor;
                    selected_neighbors.push_back(selected_neighbor);
                    idx++;

                    // 이웃에서 현재 노드로의 연결
                    omp_set_lock(&locks[selected_neighbor]);
                    size_t nb_begin, nb_end;
                    hnsw.neighbor_range(selected_neighbor, level, &nb_begin, &nb_end);

                    bool connection_added = false;
                    for (size_t j = nb_begin; j < nb_end; j++) {
                        if (hnsw.neighbors[j] < 0) {
                            hnsw.neighbors[j] = label;
                            connection_added = true;
                            break;
                        }
                    }

                    // 가득 찬 경우 거리 기반 대체
                    if (!connection_added) {
                        float max_dist = -1;
                        size_t max_idx = nb_begin;
                        for (size_t j = nb_begin; j < nb_end; j++) {
                            float dist = dis->symmetric_dis(selected_neighbor, hnsw.neighbors[j]);
                            if (dist > max_dist) {
                                max_dist = dist;
                                max_idx = j;
                            }
                        }

                        float new_dist = dis->symmetric_dis(selected_neighbor, label);
                        if (new_dist < max_dist) {
                            hnsw.neighbors[max_idx] = label;
                        }
                    }

                    omp_unset_lock(&locks[selected_neighbor]);
                }

                // 남은 슬롯 초기화
                while (idx < end) {
                    hnsw.neighbors[idx++] = -1;
                }
            }

            is_deleted[label] = false;

            if (insert_level > hnsw.max_level) {
                hnsw.max_level = insert_level;
                hnsw.entry_point = label;
            }

        } catch (const std::exception& e) {
            omp_unset_lock(&locks[label]);
            throw;
        }

        omp_unset_lock(&locks[label]);
    }
*/
    /*
    void IndexHNSW::insertPoint_knn(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint_knn");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        const size_t k = 10;              // kNN 검색에서 사용할 k값
        hnsw.efSearch = 10;               // efSearch 값 설정

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        omp_set_lock(&locks[label]);

        // 레벨 0에서만 작업
        size_t begin, end;
        hnsw.neighbor_range(label, 0, &begin, &end);

        // kNN 검색 수행
        std::vector<float> distances(k);
        std::vector<idx_t> neighbors(k);
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
        dis->set_query(point_vec.data());

        // kNN 검색 결과를 후보로 변환
        std::priority_queue<std::pair<float, storage_idx_t>> candidates;
        search(1, point_vec.data(), k, distances.data(), neighbors.data());

        for (size_t i = 0; i < k; i++) {
            idx_t neighbor_id = neighbors[i];
            if (neighbor_id == label || neighbor_id < 0 || is_deleted[neighbor_id])
                continue;
            candidates.emplace(distances[i], neighbor_id);
        }

        // 휴리스틱 기반 이웃 선택
        getNeighborsByHeuristic2_insert(candidates, hnsw.nb_neighbors(0));

        // 새로운 이웃 연결 설정
        size_t idx = begin;
        std::vector<storage_idx_t> selected_neighbors;

        while (!candidates.empty() && idx < end) {
            storage_idx_t new_neighbor = candidates.top().second;
            candidates.pop();

            hnsw.neighbors[idx] = new_neighbor;
            selected_neighbors.push_back(new_neighbor);
            idx++;
        }

        // 남은 슬롯 초기화
        while (idx < end) {
            hnsw.neighbors[idx++] = -1;
        }

        // 양방향 연결 설정
        for (storage_idx_t new_neighbor : selected_neighbors) {
            omp_set_lock(&locks[new_neighbor]);

            size_t nb_begin, nb_end;
            hnsw.neighbor_range(new_neighbor, 0, &nb_begin, &nb_end);

            // neighbor의 연결 목록에 현재 노드 추가
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] < 0) {
                    hnsw.neighbors[j] = label;
                    break;
                }
            }
            omp_unset_lock(&locks[new_neighbor]);
        }

        is_deleted[label] = false;
        omp_unset_lock(&locks[label]);
    }
*/

    void IndexHNSW::insertPoint(storage_idx_t label) {
        if (label >= ntotal) {
            throw std::runtime_error("Invalid node_id in insertPoint");
        }

        if (!is_deleted[label]) {
            throw std::runtime_error("Cannot reinsert a non-deleted point");
        }

        std::vector<float> point_vec(d);
        storage->reconstruct(label, point_vec.data());

        omp_set_lock(&locks[label]);

        // 레벨 0에서만 작업
        size_t begin, end;
        hnsw.neighbor_range(label, 0, &begin, &end);

        // 새로운 이웃 찾기
        std::priority_queue<std::pair<float, storage_idx_t>> candidates;
        std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
        dis->set_query(point_vec.data());

        size_t ef_construction = hnsw.nb_neighbors(0) * 2;

        for (storage_idx_t i = 0; i < ntotal; i++) {
            if (i != label && !is_deleted[i]) {
                float distance = dis->symmetric_dis(i, label);
                candidates.emplace(distance, i);

                if (candidates.size() > ef_construction) {
                    candidates.pop();
                }
            }
        }

        // 이웃 선택
        getNeighborsByHeuristic2_insert(candidates, hnsw.nb_neighbors(0));

        // 새로운 이웃 연결 설정
        size_t idx = begin;
        std::vector<storage_idx_t> selected_neighbors;

        while (!candidates.empty() && idx < end) {
            storage_idx_t new_neighbor = candidates.top().second;
            candidates.pop();

            hnsw.neighbors[idx] = new_neighbor;
            selected_neighbors.push_back(new_neighbor);
            idx++;
        }

        // 남은 슬롯 초기화
        while (idx < end) {
            hnsw.neighbors[idx++] = -1;
        }

        // 양방향 연결 설정
        for (storage_idx_t new_neighbor : selected_neighbors) {
            omp_set_lock(&locks[new_neighbor]);

            size_t nb_begin, nb_end;
            hnsw.neighbor_range(new_neighbor, 0, &nb_begin, &nb_end);

            // neighbor의 연결 목록에 현재 노드 추가
            for (size_t j = nb_begin; j < nb_end; j++) {
                if (hnsw.neighbors[j] < 0) {
                    hnsw.neighbors[j] = label;
                    break;
                }
            }

            omp_unset_lock(&locks[new_neighbor]);
        }

        is_deleted[label] = false;
        omp_unset_lock(&locks[label]);
    }

    storage_idx_t IndexHNSW::selectRandomId() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<storage_idx_t> dist(0, ntotal - 1);

        storage_idx_t selected_id;
        do {
            selected_id = dist(gen);
        } while (is_deleted[selected_id]);

        return selected_id;
    }

    void IndexHNSW::resizeIndex(size_t new_max_elements) {
        if (new_max_elements < ntotal) {
            throw std::runtime_error("Cannot resize: new capacity is smaller than current size.");
        }

        hnsw.levels.resize(new_max_elements);
        hnsw.neighbors.resize(new_max_elements * hnsw.nb_neighbors(0) * (hnsw.max_level + 1));

        // 락 크기 조정 및 초기화
        locks.resize(new_max_elements);
        for (size_t i = ntotal; i < new_max_elements; i++) {
            omp_init_lock(&locks[i]);
        }

        is_deleted.resize(new_max_elements, false);
    }

//PTH END




    void IndexHNSW::search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe,
            int search_type,
            const SearchParameters* params_in) const {
        FAISS_THROW_IF_NOT(k > 0);
        FAISS_THROW_IF_NOT(nprobe > 0);

        const SearchParametersHNSW* params = nullptr;

        if (params_in) {
            params = dynamic_cast<const SearchParametersHNSW*>(params_in);
            FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        }

        storage_idx_t ntotal = hnsw.levels.size();

        using RH = HeapBlockResultHandler<HNSW::C>;
        RH bres(n, distances, labels, k);

#pragma omp parallel
        {
            std::unique_ptr<DistanceComputer> qdis(
                    storage_distance_computer(storage));
            HNSWStats search_stats;
            VisitedTable vt(ntotal);
            RH::SingleResultHandler res(bres);

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                res.begin(i);
                qdis->set_query(x + i * d);

                hnsw.search_level_0(
                        *qdis.get(),
                        res,
                        nprobe,
                        nearest + i * nprobe,
                        nearest_d + i * nprobe,
                        search_type,
                        search_stats,
                        vt,
                        params);
                res.end();
                vt.advance();
            }
#pragma omp critical
            { hnsw_stats.combine(search_stats); }
        }
        if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
            for (int64_t i = 0; i < k * n; i++) {
                distances[i] = -distances[i];
            }
        }
    }

    void IndexHNSW::init_level_0_from_knngraph(
            int k,
            const float* D,
            const idx_t* I) {
        int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
        for (idx_t i = 0; i < ntotal; i++) {
            DistanceComputer* qdis = storage_distance_computer(storage);
            std::vector<float> vec(d);
            storage->reconstruct(i, vec.data());
            qdis->set_query(vec.data());

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = 0; j < k; j++) {
                int v1 = I[i * k + j];
                if (v1 == i)
                    continue;
                if (v1 < 0)
                    break;
                initial_list.emplace(D[i * k + j], v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }

    void IndexHNSW::init_level_0_from_entry_points(
            int n,
            const storage_idx_t* points,
            const storage_idx_t* nearests) {
        std::vector<omp_lock_t> locks(ntotal);
        for (int i = 0; i < ntotal; i++)
            omp_init_lock(&locks[i]);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));
            std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
            for (int i = 0; i < n; i++) {
                storage_idx_t pt_id = points[i];
                storage_idx_t nearest = nearests[i];
                storage->reconstruct(pt_id, vec.data());
                dis->set_query(vec.data());

                hnsw.add_links_starting_from(
                        *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

                if (verbose && i % 10000 == 0) {
                    printf("  %d / %d\r", i, n);
                    fflush(stdout);
                }
            }
        }
        if (verbose) {
            printf("\n");
        }

        for (int i = 0; i < ntotal; i++)
            omp_destroy_lock(&locks[i]);
    }

    void IndexHNSW::reorder_links() {
        int M = hnsw.nb_neighbors(0);

#pragma omp parallel
        {
            std::vector<float> distances(M);
            std::vector<size_t> order(M);
            std::vector<storage_idx_t> tmp(M);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

#pragma omp for
            for (storage_idx_t i = 0; i < ntotal; i++) {
                size_t begin, end;
                hnsw.neighbor_range(i, 0, &begin, &end);

                for (size_t j = begin; j < end; j++) {
                    storage_idx_t nj = hnsw.neighbors[j];
                    if (nj < 0) {
                        end = j;
                        break;
                    }
                    distances[j - begin] = dis->symmetric_dis(i, nj);
                    tmp[j - begin] = nj;
                }

                fvec_argsort(end - begin, distances.data(), order.data());
                for (size_t j = begin; j < end; j++) {
                    hnsw.neighbors[j] = tmp[order[j - begin]];
                }
            }
        }
    }

    void IndexHNSW::link_singletons() {
        printf("search for singletons\n");

        std::vector<bool> seen(ntotal);

        for (size_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                storage_idx_t ni = hnsw.neighbors[j];
                if (ni >= 0)
                    seen[ni] = true;
            }
        }

        int n_sing = 0, n_sing_l1 = 0;
        std::vector<storage_idx_t> singletons;
        for (storage_idx_t i = 0; i < ntotal; i++) {
            if (!seen[i]) {
                singletons.push_back(i);
                n_sing++;
                if (hnsw.levels[i] > 1)
                    n_sing_l1++;
            }
        }

        printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
               n_sing,
               ntotal,
               n_sing_l1);

        std::vector<float> recons(singletons.size() * d);
        for (int i = 0; i < singletons.size(); i++) {
            FAISS_ASSERT(!"not implemented");
        }
    }

    void IndexHNSW::permute_entries(const idx_t* perm) {
        auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
        FAISS_THROW_IF_NOT_MSG(
                flat_storage, "don't know how to permute this index");
        flat_storage->permute_entries(perm);
        hnsw.permute_entries(perm);
    }

    DistanceComputer* IndexHNSW::get_distance_computer() const {
        return storage->get_distance_computer();
    }

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

    IndexHNSWFlat::IndexHNSWFlat() {
        is_trained = true;
    }

    IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
            : IndexHNSW(
            (metric == METRIC_L2) ? new IndexFlatL2(d)
                                  : new IndexFlat(d, metric),
            M) {
        own_fields = true;
        is_trained = true;
    }

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

    IndexHNSWPQ::IndexHNSWPQ() = default;

    IndexHNSWPQ::IndexHNSWPQ(
            int d,
            int pq_m,
            int M,
            int pq_nbits,
            MetricType metric)
            : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits, metric), M) {
        own_fields = true;
        is_trained = false;
    }

    void IndexHNSWPQ::train(idx_t n, const float* x) {
        IndexHNSW::train(n, x);
        (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
    }

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

    IndexHNSWSQ::IndexHNSWSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric)
            : IndexHNSW(new IndexScalarQuantizer(d, qtype, metric), M) {
        is_trained = this->storage->is_trained;
        own_fields = true;
    }

    IndexHNSWSQ::IndexHNSWSQ() = default;

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

    IndexHNSW2Level::IndexHNSW2Level(
            Index* quantizer,
            size_t nlist,
            int m_pq,
            int M)
            : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
        own_fields = true;
        is_trained = false;
    }

    IndexHNSW2Level::IndexHNSW2Level() = default;

    namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
        int search_from_candidates_2(
                const HNSW& hnsw,
                DistanceComputer& qdis,
                int k,
                idx_t* I,
                float* D,
                MinimaxHeap& candidates,
                VisitedTable& vt,
                HNSWStats& stats,
                int level,
                int nres_in = 0) {
            int nres = nres_in;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                FAISS_ASSERT(v1 >= 0);
                vt.visited[v1] = vt.visno + 1;
            }

            int nstep = 0;

            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    if (vt.visited[v1] == vt.visno + 1) {
                        // nothing to do
                    } else {
                        float d = qdis(v1);
                        candidates.push(v1, d);

                        // never seen before --> add to heap
                        if (vt.visited[v1] < vt.visno) {
                            if (nres < k) {
                                faiss::maxheap_push(++nres, D, I, d, v1);
                            } else if (d < D[0]) {
                                faiss::maxheap_replace_top(nres, D, I, d, v1);
                            }
                        }
                        vt.visited[v1] = vt.visno + 1;
                    }
                }

                nstep++;
                if (nstep > hnsw.efSearch) {
                    break;
                }
            }

            stats.n1++;
            if (candidates.size() == 0)
                stats.n2++;

            return nres;
        }

    } // namespace

    void IndexHNSW2Level::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const {
        FAISS_THROW_IF_NOT(k > 0);
        FAISS_THROW_IF_NOT_MSG(
                !params, "search params not supported for this index");

        if (dynamic_cast<const Index2Layer*>(storage)) {
            IndexHNSW::search(n, x, k, distances, labels);

        } else { // "mixed" search
            size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

            const IndexIVFPQ* index_ivfpq =
                    dynamic_cast<const IndexIVFPQ*>(storage);

            int nprobe = index_ivfpq->nprobe;

            std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
            std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

            index_ivfpq->quantizer->search(
                    n, x, nprobe, coarse_dis.get(), coarse_assign.get());

            index_ivfpq->search_preassigned(
                    n,
                    x,
                    k,
                    coarse_assign.get(),
                    coarse_dis.get(),
                    distances,
                    labels,
                    false);

#pragma omp parallel
            {
                VisitedTable vt(ntotal);
                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(storage));

                constexpr int candidates_size = 1;
                MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
                for (idx_t i = 0; i < n; i++) {
                    idx_t* idxi = labels + i * k;
                    float* simi = distances + i * k;
                    dis->set_query(x + i * d);

                    // mark all inverted list elements as visited

                    for (int j = 0; j < nprobe; j++) {
                        idx_t key = coarse_assign[j + i * nprobe];
                        if (key < 0)
                            break;
                        size_t list_length = index_ivfpq->get_list_size(key);
                        const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                        for (int jj = 0; jj < list_length; jj++) {
                            vt.set(ids[jj]);
                        }
                    }

                    candidates.clear();

                    for (int j = 0; j < k; j++) {
                        if (idxi[j] < 0)
                            break;
                        candidates.push(idxi[j], simi[j]);
                    }

                    // reorder from sorted to heap
                    maxheap_heapify(k, simi, idxi, simi, idxi, k);

                    HNSWStats search_stats;
                    search_from_candidates_2(
                            hnsw,
                            *dis,
                            k,
                            idxi,
                            simi,
                            candidates,
                            vt,
                            search_stats,
                            0,
                            k);
                    n1 += search_stats.n1;
                    n2 += search_stats.n2;
                    ndis += search_stats.ndis;
                    nhops += search_stats.nhops;

                    vt.advance();
                    vt.advance();

                    maxheap_reorder(k, simi, idxi);
                }
            }

            hnsw_stats.combine({n1, n2, ndis, nhops});
        }
    }

    void IndexHNSW2Level::flip_to_ivf() {
        Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

        FAISS_THROW_IF_NOT(storage2l);

        IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
                storage2l->q1.quantizer,
                d,
                storage2l->q1.nlist,
                storage2l->pq.M,
                8);
        index_ivfpq->pq = storage2l->pq;
        index_ivfpq->is_trained = storage2l->is_trained;
        index_ivfpq->precompute_table();
        index_ivfpq->own_fields = storage2l->q1.own_fields;
        storage2l->transfer_to_IVFPQ(*index_ivfpq);
        index_ivfpq->make_direct_map(true);

        storage = index_ivfpq;
        delete storage2l;
    }

/**************************************************************
 * IndexHNSWCagra implementation
 **************************************************************/

    IndexHNSWCagra::IndexHNSWCagra() {
        is_trained = true;
    }

    IndexHNSWCagra::IndexHNSWCagra(int d, int M, MetricType metric)
            : IndexHNSW(
            (metric == METRIC_L2)
            ? static_cast<IndexFlat*>(new IndexFlatL2(d))
            : static_cast<IndexFlat*>(new IndexFlatIP(d)),
            M) {
        FAISS_THROW_IF_NOT_MSG(
                ((metric == METRIC_L2) || (metric == METRIC_INNER_PRODUCT)),
                "unsupported metric type for IndexHNSWCagra");
        own_fields = true;
        is_trained = true;
        init_level0 = true;
        keep_max_size_level0 = true;
    }

    void IndexHNSWCagra::add(idx_t n, const float* x) {
        FAISS_THROW_IF_NOT_MSG(
                !base_level_only,
                "Cannot add vectors when base_level_only is set to True");

        IndexHNSW::add(n, x);
    }

    void IndexHNSWCagra::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const {
        if (!base_level_only) {
            IndexHNSW::search(n, x, k, distances, labels, params);
        } else {
            std::vector<storage_idx_t> nearest(n);
            std::vector<float> nearest_d(n);

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(this->storage));
                dis->set_query(x + i * d);
                nearest[i] = -1;
                nearest_d[i] = std::numeric_limits<float>::max();

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

                for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                    auto idx = distrib(gen);
                    auto distance = (*dis)(idx);
                    if (distance < nearest_d[i]) {
                        nearest[i] = idx;
                        nearest_d[i] = distance;
                    }
                }
                FAISS_THROW_IF_NOT_MSG(
                        nearest[i] >= 0, "Could not find a valid entrypoint.");
            }

            search_level_0(
                    n,
                    x,
                    k,
                    nearest.data(),
                    nearest_d.data(),
                    distances,
                    labels,
                    1, // n_probes
                    1, // search_type
                    params);
        }
    }


} // namespace faiss