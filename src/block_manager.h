#pragma once
// Paged KV block allocation with prefix caching.
//
// Blocks are reference counted. A full block can be registered under a hash of its
// tokens chained with the hash of the block before it, so two sequences that share a
// prompt prefix share the blocks holding it. When a block's count drops to zero it
// stays cached (evictable, least recently released first) until the space is needed.
// A lookup compares the stored tokens, not just the hash, so a collision can never
// hand a sequence the wrong cache.

#include <array>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace bw {

constexpr int kBlockSize = 16;
using BlockTokens = std::array<int32_t, kBlockSize>;

uint64_t block_hash(uint64_t parent, const int32_t* tokens);

class BlockManager {
 public:
  BlockManager(int num_blocks, bool prefix_caching);

  int num_blocks() const { return static_cast<int>(ref_.size()); }
  // Blocks that can be handed out right now (never-used or cached-but-unreferenced).
  int available() const { return static_cast<int>(free_.size() + evictable_.size()); }
  bool prefix_caching() const { return prefix_caching_; }

  std::optional<int32_t> allocate();
  void release(int32_t block);

  // Prefix cache.
  std::optional<int32_t> lookup(uint64_t hash, const int32_t* tokens);  // takes a reference on hit
  void register_full(int32_t block, uint64_t hash, const int32_t* tokens);

  int refcount(int32_t block) const { return ref_[block]; }
  int64_t evictions() const { return evictions_; }

 private:
  void forget(int32_t block);

  bool prefix_caching_;
  std::vector<int> ref_;
  std::vector<std::optional<uint64_t>> hash_of_;
  std::vector<BlockTokens> tokens_of_;
  std::vector<int32_t> free_;
  std::list<int32_t> evictable_;
  std::vector<std::list<int32_t>::iterator> evict_pos_;
  std::vector<bool> in_evictable_;
  std::unordered_map<uint64_t, int32_t> by_hash_;
  int64_t evictions_ = 0;
};

}  // namespace bw
