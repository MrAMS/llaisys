#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <memory>
#include <utility>
#include <vector>
#include <queue>
#include <map>
#include <set>

#include <cassert>

#include "../core/llaisys_core.hpp"

namespace llaisys {
namespace PagedCache {
    using token_t = int64_t;
    const auto invalid_hash = uint64_t(-1);

    class Block{
    public:
        uint64_t id;
        uint64_t ref_count;
        uint64_t hash; // 当前及前面所有block中token的hash，若当前块未满，则为invalid_hash

        Block(uint64_t id);

        void update(uint64_t new_hash);

        void reset();
    };

    class Sequence;

    class BlockManager{
    public:
        BlockManager(uint64_t layer_n, uint64_t block_n, uint64_t block_sz, uint64_t token_sz, llaisysDeviceType_t device_type, int device);

        uint64_t allocate_block(uint64_t hash);
        void deallocate_block(uint64_t block_id);

        void add_ref_block(uint64_t block_id);
        void dec_ref_block(uint64_t block_id);

        std::pair<bool, uint64_t> find_block(uint64_t hash) const;
        uint64_t get_block_hash(uint64_t block_id) const{
            return blocks[block_id].hash;
        }
        void update_block_hash(uint64_t block_id, uint64_t new_hash){
            blocks[block_id].update(new_hash);
        }

        std::byte* data(uint64_t layer_i, bool is_v, uint64_t block_id, uint64_t token_i);

        uint64_t layer_num() const{
            return _layer_n;
        }

        uint64_t block_size() const{
            return _block_sz;
        }
        uint64_t token_size() const{
            return _token_sz;
        }
        uint64_t free_blocks() const{
            return free_block_ids.size();
        }
    private:
        uint64_t _layer_n; // how many layer
        uint64_t _block_n; // how many blocks in one layer
        uint64_t _block_sz; // block_sz tokens per block
        uint64_t _token_sz; // size in byte per token

        std::vector<Block> blocks;
        std::map<uint64_t, uint64_t> hash2id; // hash-block_id map
        std::queue<uint64_t> free_block_ids;
        std::set<uint64_t> used_block_ids;

        core::storage_t _storage;
    };

    enum class SequenceStatus{
        WAITING, RUNNING, FINISHED
    };

    class Sequence{
    public:
        SequenceStatus status;

        Sequence(uint64_t seq_id, const std::vector<token_t>& seq_tokens, BlockManager* manager, token_t eos_token, uint64_t max_tokens=0, float temperature=0.5);

        /* 为所有token分配KVCache Block */
        bool allocate();
        /* 释放所有的KVCache Block，保留tokens */
        void deallocate();

        /* 是否有空闲的block给新的token */
        bool can_append() const;
        /* 为新的token准备block */
        void may_append();

        /* 添加新的token并更新状态 */
        void add_token(token_t new_token);

        /* 获得token KVCache 实际地址 */
        std::byte* get_kvcache(uint64_t layer_i, bool is_v, uint64_t token_i);

        std::vector<std::byte*> get_slot_mapping(uint64_t layer_i, bool is_v);

        bool is_finished() const;
        uint64_t num_tokens() const;
        uint64_t num_blocks() const;
        uint64_t num_layers() const;
        uint64_t last_block_num_tokens() const;
        token_t* tokens_at_block(uint64_t i);
        uint64_t size_at_block(uint64_t i) const;
        uint64_t block_sz() const;
        token_t* tokens_ptr();
        token_t* tokens_ptr(uint64_t token_i);
        uint64_t get_id() const{
            return _id;
        }

        uint64_t num_cached() const{
            return _cached_tokens;
        }
        float temperature() const{
            return _temperature;
        }
    
    private:
        uint64_t _id;
        std::vector<token_t> _token_ids;
        BlockManager* _manager;

        uint64_t _prompt_tokens;
        uint64_t _cached_tokens;

        token_t _eos;
        uint64_t _max_tokens;
        float _temperature;

        std::vector<uint64_t> block_table;

        bool need_new_block() const;
    };
    
    class Scheduler{
    public:
        Scheduler(uint64_t max_running_seqs, uint64_t layer_n, uint64_t block_n, uint64_t block_sz, uint64_t token_sz, llaisysDeviceType_t device_type, int device);

        bool is_finished() const;

        void add(const Sequence& seq);
        void add(uint64_t seq_id, const std::vector<token_t>& seq_tokens, token_t eos_token, uint64_t max_tokens=0, float temperature=0.5);


        std::pair<std::vector<Sequence*>, bool> schedule(); // return <scheduled_seqs, is_prefill>

        void postprocess(Sequence* seq, token_t new_token);

    private:
        uint64_t max_running_seqs;

        std::shared_ptr<BlockManager> manager;
        std::deque<Sequence> waiting;
        std::deque<Sequence> running;
        std::deque<Sequence> finished;

        void preempt(Sequence& seq);
    };

}
}