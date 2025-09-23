#include "paged_cache.hpp"

#include "xxhash.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#define DEBUG_ON 1
#if DEBUG_ON
#include <cstdio>
#include <cstdlib>
#define DEBUG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define ASSERT_(X) assert(X)
#else
#define ASSERT_(X) 
#define DEBUG(X) 
#endif

namespace llaisys {
namespace PagedCache {
    
    Block::Block(uint64_t id):id(id),ref_count(0),hash(invalid_hash){
        
    };

    void Block::update(uint64_t new_hash){
        hash = new_hash;
    }

    void Block::reset(){
        hash = invalid_hash;
        ref_count = 1;
    }

    BlockManager::BlockManager(uint64_t layer_n, uint64_t block_n, uint64_t block_sz, uint64_t token_sz, llaisysDeviceType_t device_type, int device)
    :_layer_n(layer_n), _block_n(block_n),_block_sz(block_sz), _token_sz(token_sz){
        for(uint64_t i=0;i<block_n;++i){
            blocks.emplace_back(i);
            free_block_ids.push(i);
        }
        _storage = core::context().runtime().allocateHostStorage(layer_n*2*block_n*block_sz*token_sz);

    }

    static uint64_t compute_hash64(const token_t* input, size_t sz, uint64_t prefix=invalid_hash){
        XXH64_state_t* const state = XXH64_createState();
        XXH64_reset(state, 0); // seed=0
        XXH64_update(state, &prefix, sizeof(uint64_t));
        XXH64_update(state, input, sz*sizeof(token_t));
        uint64_t res = XXH64_digest(state);
        XXH64_freeState(state);
        return res;
    }

    uint64_t BlockManager::allocate_block(uint64_t hash){
        auto id = free_block_ids.front();
        ASSERT_(blocks[id].ref_count == 0);
        blocks[id].reset();
        blocks[id].update(hash);
        free_block_ids.pop();
        used_block_ids.insert(id);
        if(hash!=invalid_hash){
            hash2id[hash]=id;
        }
        DEBUG("allocate block (id=%ld)", id);
        return id;
    }
    void BlockManager::deallocate_block(uint64_t block_id){
        ASSERT_(blocks[id].ref_count == 0);
        used_block_ids.erase(block_id);
        free_block_ids.push(block_id);
        if(blocks[block_id].hash != invalid_hash){
            hash2id.erase(blocks[block_id].hash);
        }
        DEBUG("deallocate block (id=%ld)", block_id);
    }

    void BlockManager::add_ref_block(uint64_t block_id){
        ASSERT_(blocks[block_id].ref_count>0);
        ASSERT_(used_block_ids.find(block_id)!=used_block_ids.end());
        blocks[block_id].ref_count += 1;
    }

    void BlockManager::dec_ref_block(uint64_t block_id){
        ASSERT_(blocks[block_id].ref_count>0);
        ASSERT_(used_block_ids.find(block_id)!=used_block_ids.end());
        blocks[block_id].ref_count -= 1;
        if(blocks[block_id].ref_count == 0) deallocate_block(block_id);
    }

    std::byte* BlockManager::data(uint64_t layer_i, bool is_v, uint64_t block_id, uint64_t token_i){
        return _storage->memory() + layer_i*(2*_block_n*_block_sz*_token_sz) + is_v*(_block_n*_block_sz*_token_sz) + block_id*(_block_sz*_token_sz) + token_i*_token_sz;
    }

    std::pair<bool, uint64_t> BlockManager::find_block(uint64_t hash) const{
        const auto it = hash2id.find(hash);
        if(it!=hash2id.end()) return std::make_pair(true, it->second);
        else return std::make_pair(false, -1);
    }

    Sequence::Sequence(uint64_t seq_id, const std::vector<token_t>& seq_tokens, BlockManager* manager, token_t eos_token, uint64_t max_tokens)
    :_id(seq_id), _token_ids(seq_tokens), _manager(manager), _eos(eos_token), _max_tokens(max_tokens){
        ASSERT_(manager!=nullptr);
        _prompt_tokens = seq_tokens.size();
        _cached_tokens = 0;
        status = SequenceStatus::WAITING;
    }

    std::vector<std::byte*> Sequence::get_slot_mapping(uint64_t layer_i, bool is_v){
        std::vector<std::byte*> slot_mapping;
        for(uint64_t i=0;i<num_tokens();++i)
            slot_mapping.push_back(get_kvcache(layer_i, is_v, i));
        return slot_mapping;
    }


    bool Sequence::is_finished() const{
        return status == SequenceStatus::FINISHED;
    }

    uint64_t Sequence::block_sz() const{
        return _manager->block_size();
    }

    token_t* Sequence::tokens_ptr(){
        return _token_ids.data();
    }

    token_t* Sequence::tokens_ptr(uint64_t token_i){
        return tokens_ptr() + token_i;
    }

    uint64_t Sequence::num_tokens() const{
        return _token_ids.size();
    }

    uint64_t Sequence::num_blocks() const{
        return (num_tokens() + block_sz() - 1) / block_sz();
    }

    uint64_t Sequence::num_layers() const{
        return _manager->layer_num();
    }

    uint64_t Sequence::last_block_num_tokens() const{
        return num_tokens() - (num_blocks()-1)*block_sz();
    }

    token_t* Sequence::tokens_at_block(uint64_t i){
        return _token_ids.data()+i*block_sz();
    }

    uint64_t Sequence::size_at_block(uint64_t i) const{
        if(i==num_blocks()-1){
            return last_block_num_tokens();
        }else{
            return block_sz();
        }
    }

    bool Sequence::allocate(){
        ASSERT_(block_table.empty()); // 未分配应当为空
        if(_manager->free_blocks() < num_blocks())
            return false;
        for(uint64_t i=0;i<num_blocks();++i){
            uint64_t h=-1;
            auto sz = size_at_block(i);
            if(sz==block_sz()) h = compute_hash64(tokens_at_block(i), sz, h);
            uint64_t block_id;
            if(auto [hit, id] = _manager->find_block(h);
            !hit || sz!=block_sz()){
                // 如果没有找到hash对应的block，或者当前block未满，则需要申请新的block
                block_id = _manager->allocate_block(h);
            }else{
                // cache hit
                block_id = id;
                _cached_tokens += block_sz();
                // 更新block的引用计数
                _manager->add_ref_block(block_id);
                printf("Hit Cache\n");
            }
            block_table.push_back(block_id);
        }
        return true;
    }

    void Sequence::deallocate(){
        for(auto it=block_table.rbegin();it!=block_table.rend();it++){
            auto block_id = *it;
            _manager->dec_ref_block(block_id);
        }
        _cached_tokens = 0;
        block_table.clear();
        block_table.shrink_to_fit();
    }

    bool Sequence::can_append() const{
        return _manager->free_blocks() >= need_new_block();
    }

    void Sequence::may_append(){
        auto last_block_id = *(block_table.end()-1);
        if(need_new_block()){
            auto block_id = _manager->allocate_block(invalid_hash);
            block_table.push_back(block_id);
        }else if(num_tokens() % block_sz() == 0){
            // 当前块被填满，更新其hash
            uint64_t prefix = block_table.size()>1 ? 
                _manager->get_block_hash(block_table[block_table.size()-2]):
                invalid_hash;
            uint64_t h = compute_hash64(tokens_at_block(last_block_id), block_sz(), prefix);
            _manager->update_block_hash(last_block_id, h);
        }else{
            ASSERT_(manager->get_hash(last_block_id) == invalid_hash);
        }
    }

    void Sequence::add_token(token_t new_token){
        _token_ids.push_back(new_token);
        if(new_token == _eos) status = SequenceStatus::FINISHED;
        if(_max_tokens!=0 && _token_ids.size()>_max_tokens) status = SequenceStatus::FINISHED;
    }

    std::byte* Sequence::get_kvcache(uint64_t layer_i, bool is_v, uint64_t token_i){
        const auto block_id = token_i / block_sz();
        ASSERT_(block_id < block_table.size());
        return _manager->data(layer_i, is_v, block_table[block_id], token_i % block_sz());
    }

    bool Sequence::need_new_block() const{
        // nano-vllm这里是1而不是0比较奇怪，感觉是为了避免最后一个token独占一整块block的情况
        // 在至少有两个新token时才会分配新的block
        // 这里先修改为0，表示立即为新token分配新block
        return num_tokens() % block_sz() == 0;
    }

    Scheduler::Scheduler(uint64_t max_running_seqs, uint64_t layer_n, uint64_t block_n, uint64_t block_sz, uint64_t token_sz, llaisysDeviceType_t device_type, int device):
    max_running_seqs(max_running_seqs){
        manager = std::make_shared<BlockManager>(layer_n, block_n, block_sz, token_sz, device_type, device);
    }

    void Scheduler::add(const Sequence& seq){
        waiting.push_back(seq);
    }

    void Scheduler::add(uint64_t seq_id, const std::vector<token_t>& seq_tokens, token_t eos_token, uint64_t max_tokens){
        waiting.emplace_back(seq_id, seq_tokens, manager.get(), eos_token, max_tokens);
    }

    std::pair<std::vector<Sequence*>, bool> Scheduler::schedule(){
        uint64_t num_seqs=0;
        std::vector<Sequence*> scheduled_seqs;
        // prefill
        while(!waiting.empty() && running.size() < max_running_seqs && num_seqs < max_running_seqs){
            Sequence seq = waiting.front();
            bool allocated = true;
            while(!seq.allocate()){
                printf("seq%ld cannot prefill more\n", seq.get_id());
                if(!finished.empty()){
                    printf("try to deallocate finished seq%ld\n", finished.front().get_id());
                    finished.front().deallocate();
                    finished.pop_front();
                }else{
                    allocated = false;
                    break;
                }
            }
            if(!allocated) break;
            seq.status = SequenceStatus::RUNNING;
            waiting.pop_front();
            running.push_back(seq);
            num_seqs += 1;
            if(seq.num_cached()<seq.num_tokens()) // 如果cache全部命中，则无须prefill
                scheduled_seqs.push_back(&running.back());
        }
        if(!scheduled_seqs.empty())
            return std::make_pair(scheduled_seqs, true);

        // decode
        std::vector<Sequence> t;
        while(!running.empty() && num_seqs < max_running_seqs){
            Sequence seq = running.front();
            running.pop_front();
            if(seq.is_finished()){
                printf("seq%ld finished\n", seq.get_id());
                continue;
            }
            while(!seq.can_append()){
                printf("seq%ld cannot decode more\n", seq.get_id());
                if(!finished.empty()){
                    printf("try to deallocate finished seq%ld\n", finished.front().get_id());
                    finished.front().deallocate();
                    finished.pop_front();
                }else if(!running.empty()){
                    printf("try to deallocate running seq%ld\n", running.back().get_id());
                    preempt(running.back());  // 释放队尾的seq来满足新来的seq（调度策略:后进先出）
                    running.pop_back();
                }else{
                    printf("have to deallocate itself\n");
                    preempt(seq);
                    break;
                }
            }
            if(seq.can_append()){
                seq.may_append();
                t.push_back(seq);
                num_seqs += 1;
            }
        }
        for(auto it=t.rbegin();it!=t.rend();++it){
            running.push_front(*it); // 加到队首，保持顺序
            scheduled_seqs.push_back(&running.front());
        }
        std::reverse(scheduled_seqs.begin(), scheduled_seqs.end());
        return std::make_pair(scheduled_seqs, false);
    }

    bool Scheduler::is_finished() const{
        return running.empty() && waiting.empty();
    }

    void Scheduler::preempt(Sequence& seq){
        seq.status = SequenceStatus::WAITING;
        seq.deallocate();
        waiting.push_front(seq);
    }

    void Scheduler::postprocess(Sequence* seq, token_t new_token){
        seq->add_token(new_token);
        if(seq->is_finished()){
            finished.push_back(*seq);
            // while(finished.size()>max_finished_seqs){
            //     finished.front().deallocate();
            //     finished.pop_front();
            // }
        }
    }
    
}
}