# -*- coding: utf-8 -*-
# @Time    : 2025/06/01 
# @Author  : Zaid Qassim Abdulyemmah
# @Email   : 23sf51033@hitsz.edu.cn, zaidqassim12@gmail.com

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class RAMRec(SequentialRecommender):
    input_type = 'seq'
 
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.user_mapping = {}
        self.user_counter = 0
        
        # Core dimensions
        self.device = config['device']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.loss_type = config['loss_type']
        
        # Memory configuration
        self.use_memory = config['enable_memory_management'] if 'enable_memory_management' in config else False
        self.memory_in_training = config['memory_in_training'] if 'memory_in_training' in config else False
        self.enable_adaptive_k = config['enable_adaptive_k'] if 'enable_adaptive_k' in config else False
        
        # Initialize embeddings
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )
        
        # Transformer encoder
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act='gelu',
            layer_norm_eps=1e-12
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Enhanced Memory components with proper config access
        if self.use_memory:
            # Memory parameters
            memory_size = config['memory_size'] if 'memory_size' in config else 100
            similarity_threshold = config['memory_similarity_threshold'] if 'memory_similarity_threshold' in config else 0.3
            temperature = config['memory_temperature'] if 'memory_temperature' in config else 0.5
            
            # Initialize memory with parameters
            self.memory = EpisodicMemory(
                capacity=memory_size, 
                device=self.device,
                similarity_threshold=similarity_threshold,
                temperature=temperature
            )
            
            # Memory gate with configurable influence
            max_influence = config['memory_gate_max_influence'] if 'memory_gate_max_influence' in config else 0.3
            self.memory_gate = AdaptiveMemoryGate(
                self.hidden_size,
                max_influence=max_influence
            )
            
            # Memory attention
            self.memory_attention = MemoryAugmentedAttention(self.hidden_size, num_heads=4)
            
            # Retrieval k
            self.retrieval_k = config['retrieval_k'] if 'retrieval_k' in config else 10
            
            # Initialize Adaptive-K if enabled
            if self.enable_adaptive_k:
                adaptive_min_k = config['adaptive_min_k'] if 'adaptive_min_k' in config else 3
                adaptive_max_k = config['adaptive_max_k'] if 'adaptive_max_k' in config else 15
                
                self.adaptive_k = AdaptiveKRetrieval(
                    min_k=adaptive_min_k,
                    max_k=adaptive_max_k,
                    base_k=self.retrieval_k
                )
            
            # Memory update tracking
            self.memory_update_freq = config['memory_update_freq'] if 'memory_update_freq' in config else 10
            self.batch_memory_buffer = []
        
        # Loss function
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            
        # Initialize weights
        self.apply(self._init_weights)
        
        # Training state
        self.training_step = 0
        
    def _init_weights(self, module):
        """Initialize weights with Xavier/He initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask"""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def _generate_user_id(self, item_seq, idx):
        """
        Generating a stable and effective user ID 
        """
        seq = item_seq[idx].cpu().numpy()
        non_zero = seq[seq > 0]
        
        if len(non_zero) == 0:
            return -1
        
        if len(non_zero) >= 4:
            signature = [non_zero[0], non_zero[1], non_zero[-2], non_zero[-1]]
        else:
            signature = list(non_zero)
        
        import hashlib
        sig_str = '-'.join(map(str, signature))
        hash_val = int(hashlib.md5(sig_str.encode()).hexdigest()[:8], 16)
        
        user_id = hash_val % 10000
        # user_id = hash_val % 50000  # for large datasets 
        
        return user_id

    def _enhance_with_memory(self, seq_output, user_id):
        """Enhanced memory augmentation with optional adaptive K"""
        # Determine K value
        if self.enable_adaptive_k and hasattr(self, 'adaptive_k'):
            k = self.adaptive_k.get_adaptive_k(user_id, self.memory)
        else:
            k = self.retrieval_k
        
        # Retrieve from memory
        mem_repr = self.memory.retrieve(user_id, seq_output, k=k)
        
        if mem_repr is not None:
            # Apply cross-attention
            attended = self.memory_attention(seq_output, mem_repr)
            # Apply adaptive gate
            fused = self.memory_gate(seq_output, attended)
            return fused
        else:
            return seq_output
            
    def forward(self, item_seq, item_seq_len):
        """Forward pass with memory enhancement"""
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        
        # Get embeddings
        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)
        
        # Combine embeddings
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # Self-attention
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, 
            extended_attention_mask, 
            output_all_encoded_layers=True
        )
        output = trm_output[-1]
        
        # Gather sequence representation
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        
        # Memory Enhancement
        if self.use_memory and (not self.training or self.memory_in_training):
            enhanced = []
            for i in range(seq_output.size(0)):
                user_id = self._generate_user_id(item_seq, i)
                enhanced_repr = self._enhance_with_memory(seq_output[i], user_id)
                enhanced.append(enhanced_repr)
            seq_output = torch.stack(enhanced)
        
        return seq_output
    
    def calculate_loss(self, interaction):
        """Training with optimized memory storage"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        self.training_step += 1
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Batch memory storage
        if self.training and self.use_memory:
            if self.training_step % self.memory_update_freq == 0:
                batch_size = seq_output.size(0)
                user_ids = []
                item_ids = []
                embeddings = []
                
                for i in range(batch_size):
                    item_id = pos_items[i].item()
                    if item_id > 0:
                        # Use consistent user_id generation
                        user_id = self._generate_user_id(item_seq, i)
                        user_ids.append(user_id)
                        item_ids.append(item_id)
                        embeddings.append(self.item_embedding(pos_items[i]))
                
                # Batch store
                if user_ids:
                    self.memory.batch_store(user_ids, item_ids, embeddings)
        
        # Calculate loss
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        # Log statistics
        if self.training_step % 100 == 0 and self.use_memory:
            total_users = len(self.memory.memory)
            total_memories = sum(len(m) for m in self.memory.memory.values())
            avg_per_user = total_memories / max(1, total_users)
            
            if total_users > 0:
                sizes = [len(m) for m in self.memory.memory.values()]
                print(f"\033[36m\033[1m[Step {self.training_step:,}]\033[0m \033[90mMemory:\033[0m \033[32m{total_users}\033[0m users | \033[33m{total_memories:,}\033[0m items | \033[34mavg {avg_per_user:.1f}/user\033[0m \033[90m│\033[0m \033[35mmin={min(sizes)}\033[0m \033[36mmax={max(sizes)}\033[0m \033[37mmed={np.median(sizes):.1f}\033[0m")
        
        return loss
    
    def predict(self, interaction):
        """Prediction with memory enhancement"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        was_training = self.training
        self.training = False
        seq_output = self.forward(item_seq, item_seq_len)
        self.training = was_training
        
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores
    
    def full_sort_predict(self, interaction):
        """Full ranking prediction with memory enhancement"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        was_training = self.training
        self.training = False
        seq_output = self.forward(item_seq, item_seq_len)
        self.training = was_training
        
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class EpisodicMemory:
    """Enhanced episodic memory with GPU acceleration"""
    
    def __init__(self, capacity=100, device='cuda', similarity_threshold=0.3, temperature=0.5):
        self.capacity = capacity
        self.device = device
        self.similarity_threshold = similarity_threshold  
        self.temperature = temperature 
        self.memory = defaultdict(list)
        self.access_counts = {}  # Track access frequency
        
    def store(self, user_id, item_id, embedding):
        """Store with quality-based management"""
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, device=self.device)
        
        embedding = embedding.detach()
        
        if user_id not in self.access_counts:
            self.access_counts[user_id] = {}
        
        existing_items = [iid for iid, _ in self.memory[user_id]]
        
        if item_id in existing_items:
            idx = existing_items.index(item_id)
            old_emb = self.memory[user_id][idx][1]
            
            freq = self.access_counts[user_id].get(item_id, 0)
            alpha = min(0.9, 0.7 + 0.05 * np.log(freq + 1))
            
            updated_emb = alpha * old_emb + (1 - alpha) * embedding
            self.memory[user_id][idx] = (item_id, updated_emb)
            self.access_counts[user_id][item_id] = freq + 1
            return
        
        if len(self.memory[user_id]) >= self.capacity:
            min_quality = float('inf')
            remove_idx = 0
            
            for idx, (iid, _) in enumerate(self.memory[user_id]):
                quality = self.access_counts[user_id].get(iid, 1)
                if quality < min_quality:
                    min_quality = quality
                    remove_idx = idx
            
            removed_id = self.memory[user_id][remove_idx][0]
            del self.memory[user_id][remove_idx]
            if removed_id in self.access_counts[user_id]:
                del self.access_counts[user_id][removed_id]
        
        self.memory[user_id].append((item_id, embedding))
        self.access_counts[user_id][item_id] = 1
    
    def batch_store(self, user_ids, item_ids, embeddings):
        """Efficient batch storage"""
        for uid, iid, emb in zip(user_ids, item_ids, embeddings):
            self.store(uid, iid, emb)
    
    def retrieve(self, user_id, query, k=10):
        """Enhanced retrieval with better similarity calculation"""
        if user_id not in self.memory or len(self.memory[user_id]) == 0:
            return None
        
        embeddings = torch.stack([emb for _, emb in self.memory[user_id]])
        
      
        query_norm = F.normalize(query.unsqueeze(0), dim=1)
        emb_norm = F.normalize(embeddings, dim=1)
        
        # Cosine similarity
        similarities = torch.matmul(query_norm, emb_norm.t()).squeeze(0)
        
        k = min(k, len(similarities))
        if k == 0:
            return None
        
        # Top-k مع threshold
        values, indices = torch.topk(similarities, k)
        
        # self.similarity_threshold
        mask = values > self.similarity_threshold
        if not mask.any():
            mask[0] = True
        
        values = values[mask]
        indices = indices[mask]
        
        # Temperature-scaled softmax -  self.temperature
        weights = F.softmax(values / self.temperature, dim=0)
        
        # Weighted aggregation
        selected = embeddings[indices]
        fused = (selected * weights.unsqueeze(1)).sum(dim=0)
        
        return fused


class AdaptiveMemoryGate(nn.Module):
    def __init__(self, hidden_size, max_influence=0.3): 
        super().__init__()
        self.max_influence = max_influence
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, seq_repr, mem_repr):
        if mem_repr is None:
            return seq_repr
        combined = torch.cat([seq_repr, mem_repr], dim=-1)
        gate = self.gate_net(combined)
        gate = torch.sigmoid((gate - 0.5) * self.temperature) * self.max_influence
        fused = (1 - gate) * seq_repr + gate * mem_repr
        return fused


class MemoryAugmentedAttention(nn.Module):
    """Cross-attention between sequence and memory"""
    
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, seq_repr, memory_embs):
        """Cross-attention fusion"""
        if memory_embs is None:
            return seq_repr
        
        batch_size = seq_repr.size(0) if seq_repr.dim() > 1 else 1
        
        # Handle different input dimensions
        if seq_repr.dim() == 1:
            seq_repr = seq_repr.unsqueeze(0)
        if memory_embs.dim() == 1:
            memory_embs = memory_embs.unsqueeze(0)
        
        # Multi-head projections
        Q = self.q_proj(seq_repr).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(memory_embs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory_embs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.out_proj(context).squeeze(1)
        
        # Residual connection
        output = self.layer_norm(output + seq_repr.squeeze(0))
        
        return output.squeeze(0) if batch_size == 1 else output


class AdaptiveKRetrieval:
    """Entropy-driven adaptive retrieval mechanism"""
    
    def __init__(self, min_k=3, max_k=20, base_k=10):
        self.min_k = min_k
        self.max_k = max_k
        self.base_k = base_k
        self.user_stats = {}  # Store user statistics
        self.cache_hits = 0
        self.refresh_freq = 100
        
    def calculate_user_entropy(self, user_id, memory):
        """Calculate entropy based on memory diversity"""
        if user_id not in memory.memory or len(memory.memory[user_id]) < 2:
            return 0.5  # Default entropy for new/sparse users
        
        # Get all embeddings for user
        embeddings = torch.stack([emb for _, emb in memory.memory[user_id]])
        
        # Calculate pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.t())
        
        # Convert to probability distribution
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(similarities.size(0), dtype=bool, device=similarities.device)
        sim_values = similarities[mask]
        
        # Normalize to probabilities
        probs = torch.softmax(sim_values, dim=0)
        
        # Calculate entropy: -Σ(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(float(len(sim_values))))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return normalized_entropy.item()
    
    def get_adaptive_k(self, user_id, memory):
        """Determine optimal K with caching"""
        
        if user_id in self.user_stats:
            self.cache_hits += 1
            
            if self.cache_hits % self.refresh_freq == 0:
                entropy = self.calculate_user_entropy(user_id, memory)
                self.user_stats[user_id]['entropy'] = entropy
                self.user_stats[user_id]['memory_size'] = len(memory.memory.get(user_id, []))
        else:
            entropy = self.calculate_user_entropy(user_id, memory)
            self.user_stats[user_id] = {
                'entropy': entropy,
                'memory_size': len(memory.memory.get(user_id, []))
            }
        
        entropy = self.user_stats[user_id]['entropy']
        memory_size = self.user_stats[user_id]['memory_size']
        
        if memory_size == 0:
            return self.min_k
        
        memory_factor = min(1.0, np.sqrt(memory_size / 100))
        entropy_factor = 0.5 + entropy
        
        adaptive_k = int(self.base_k * entropy_factor * memory_factor)
        adaptive_k = max(self.min_k, min(self.max_k, adaptive_k))
        adaptive_k = min(adaptive_k, memory_size)
        
        return adaptive_k
    
    def update_stats(self, user_id, interaction_diversity):
        """Update user statistics after interaction"""
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {'entropy': 0.5, 'memory_size': 0}
        
        # Exponential moving average for entropy
        alpha = 0.1
        old_entropy = self.user_stats[user_id]['entropy']
        self.user_stats[user_id]['entropy'] = (
            alpha * interaction_diversity + (1 - alpha) * old_entropy
        )