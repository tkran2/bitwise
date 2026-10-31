#include "block_manager.h"

#include <cstring>
#include <stdexcept>

namespace bw {

uint64_t block_hash(uint64_t parent, const int32_t* tokens) {
  // FNV-1a over the parent hash and the tokens, finished with a 64-bit mixer.
  uint64_t h = 1469598103934665603ull ^ parent;
  for (int i = 0; i < kBlockSize; ++i) {
    h ^= static_cast<uint32_t>(tokens[i]);
    h *= 1099511628211ull;
  }
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  return h;
}

BlockManager::BlockManager(int num_blocks, bool prefix_caching)
    : prefix_caching_(prefix_caching),
      ref_(num_blocks, 0),
      hash_of_(num_blocks),
      tokens_of_(num_blocks),
      evict_pos_(num_blocks),
      in_evictable_(num_blocks, false) {
  free_.reserve(num_blocks);
  for (int b = num_blocks - 1; b >= 0; --b) free_.push_back(b);  // hand out low ids first
}

std::optional<int32_t> BlockManager::allocate() {
  int32_t b;
  if (!free_.empty()) {
    b = free_.back();
    free_.pop_back();
  } else if (!evictable_.empty()) {
    b = evictable_.front();
    evictable_.pop_front();
    in_evictable_[b] = false;
    forget(b);
    ++evictions_;
  } else {
    return std::nullopt;
  }
  ref_[b] = 1;
  return b;
}

void BlockManager::release(int32_t b) {
  if (ref_[b] <= 0) throw std::logic_error("release of unreferenced block");
  if (--ref_[b] > 0) return;
  if (hash_of_[b]) {
    evict_pos_[b] = evictable_.insert(evictable_.end(), b);
    in_evictable_[b] = true;
  } else {
    free_.push_back(b);
  }
}

std::optional<int32_t> BlockManager::lookup(uint64_t hash, const int32_t* tokens) {
  if (!prefix_caching_) return std::nullopt;
  auto it = by_hash_.find(hash);
  if (it == by_hash_.end()) return std::nullopt;
  const int32_t b = it->second;
  if (std::memcmp(tokens_of_[b].data(), tokens, sizeof(BlockTokens)) != 0) return std::nullopt;
  if (in_evictable_[b]) {
    evictable_.erase(evict_pos_[b]);
    in_evictable_[b] = false;
  }
  ++ref_[b];
  return b;
}

void BlockManager::register_full(int32_t b, uint64_t hash, const int32_t* tokens) {
  if (!prefix_caching_ || hash_of_[b]) return;
  if (by_hash_.count(hash)) return;  // an identical block is already cached; keep that one
  hash_of_[b] = hash;
  std::memcpy(tokens_of_[b].data(), tokens, sizeof(BlockTokens));
  by_hash_[hash] = b;
}

void BlockManager::forget(int32_t b) {
  if (hash_of_[b]) {
    by_hash_.erase(*hash_of_[b]);
    hash_of_[b].reset();
  }
}

}  // namespace bw
