 # encoder.py - Complete Function Documentation

**Total Lines:** 1824
**Total Functions/Methods:** 67

---

## Class: SinusoidalPositionalEncoding

### `__init__(self, dim: int, max_len: int = 5000)`
- Initializes sinusoidal positional encoding
- Sets dimension and maximum sequence length
- Calls precompute method

### `_precompute_encodings(self)`
- Precomputes positional encodings for efficiency
- Creates sinusoidal patterns using sin/cos functions
- Stores encodings in pe array for reuse

### `__call__(self, seq_len: int) -> Tensor`
- Returns positional encoding for given sequence length
- Retrieves precomputed encodings up to seq_len
- Returns as Tensor with label 'pos_enc'

---

## Class: LearnedPositionalEncoding

### `__init__(self, dim: int, max_len: int = 512)`
- Initializes learned positional embeddings
- Creates random position embeddings matrix
- Supports adaptive position representation

### `__call__(self, seq_len: int) -> Tensor`
- Returns learned positional encoding for sequence
- Slices embeddings up to seq_len
- Returns as Tensor with label 'pos_enc'

### `parameters(self)`
- Returns list of learnable parameters
- Includes pos_embeddings tensor

---

## Class: ImageSpatialEncoder

### `__init__(self, output_dim: int = 256, num_scales: int = 3)`
- Initializes convolutional encoder for images
- Creates multi-scale feature extraction filters
- Sets up projection MLP for unified dimension
- Preserves spatial structure through processing

### `__call__(self, img_obs: 'ImageObservation') -> Tensor`
- Encodes image observation preserving spatial structure
- Extracts multi-scale spatial features from pyramid
- Creates patches from different scales
- Projects patches through scale-specific filters
- Adds global context to each patch
- Returns spatial feature map (num_patches, output_dim)

### `parameters(self)`
- Returns all learnable parameters
- Includes scale filters and projection MLP parameters

---

## Class: AudioTemporalEncoder

### `__init__(self, output_dim: int = 256, num_temporal_scales: int = 3)`
- Initializes AGI-grade temporal encoder for audio
- Creates multi-scale temporal filters (short/mid/long)
- Sets up mel-spectrogram, MFCC, and spectral filters
- Initializes temporal context encoder and projections
- Creates temporal attention mechanism

### `__call__(self, aud_obs: 'AudioObservation') -> Tensor`
- Encodes audio preserving temporal structure
- Extracts mel-spectrogram features at multiple scales
- Processes MFCC features for phonetic content
- Extracts spectral features (centroid, rolloff, ZCR)
- Concatenates all features per frame
- Applies multi-layer projection with residual
- Computes temporal attention over sequence
- Returns rich temporal feature sequence

### `parameters(self)`
- Returns all learnable parameters
- Includes filters, projections, and attention parameters

---

## Class: TrueVariationalLatentEncoder

### `__init__(self, slot_dim: int, latent_dim: int = 64)`
- Initializes true VAE-style latent encoder
- Creates mean and logvar encoder networks
- Creates decoder network for reconstruction
- Enables proper regularization

### `encode(self, slots: Tensor) -> Tuple[Tensor, Tensor]`
- Encodes slots to latent parameters
- Processes each slot through mean encoder
- Processes each slot through logvar encoder
- Returns mean and logvar tensors

### `reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor`
- Samples from N(μ, σ²) using reparameterization trick
- Clips logvar for numerical stability
- Computes std with safety guards
- Samples epsilon from standard normal
- Returns z = μ + σ * ε

### `decode(self, z: Tensor) -> Tensor`
- Decodes latent to slot reconstruction
- Processes each latent through decoder
- Returns reconstructed slots

### `kl_divergence(self, mean: Tensor, logvar: Tensor) -> float`
- Computes KL divergence KL(q(z|x) || p(z))
- Uses enhanced numerical stability
- Clips values to prevent overflow
- Returns scalar KL loss

### `reconstruction_loss(self, original: Tensor, reconstructed: Tensor) -> float`
- Computes reconstruction loss (MSE)
- Measures quality of reconstruction
- Returns scalar loss value

### `__call__(self, slots: Tensor) -> Dict[str, Any]`
- Full VAE forward pass
- Encodes slots to mean and logvar
- Samples latent z via reparameterization
- Decodes z to reconstructed slots
- Computes KL and reconstruction losses
- Returns dictionary with all outputs and losses

### `parameters(self)`
- Returns all learnable parameters
- Includes encoder and decoder parameters

---

## Class: FeedForwardNetwork

### `__init__(self, dim: int, hidden_dim: int = None)`
- Initializes position-wise feed-forward network
- Creates two linear layers
- Sets up layer normalization parameters

### `gelu(self, x: np.ndarray) -> np.ndarray`
- Applies Gaussian Error Linear Unit activation
- Smooth approximation of ReLU
- Returns activated values

### `layer_norm(self, x: Tensor, eps: float = 1e-6) -> Tensor`
- Applies layer normalization
- Computes mean and std
- Normalizes and applies learned scale/shift
- Returns normalized tensor

### `__call__(self, x: Tensor) -> Tensor`
- Forward pass through feed-forward network
- Applies first linear layer
- Applies GELU activation
- Applies second linear layer
- Adds residual connection
- Applies layer normalization
- Returns output tensor

### `parameters(self)`
- Returns all learnable parameters
- Includes linear layers and normalization parameters

---

## Class: TransformerEncoderBlock

### `__init__(self, dim: int, num_heads: int = 8, ff_dim: int = None, label: str = 'block')`
- Initializes AGI-grade transformer encoder block
- Creates AGIMultiHeadSelfAttention layer
- Creates feed-forward network
- Supports goal-directed modulation

### `__call__(self, x: Tensor, mask: Optional[np.ndarray] = None, goal_context: Optional[Tensor] = None) -> Tensor`
- Forward pass through transformer block
- Applies multi-head self-attention
- Applies feed-forward network
- Returns processed tensor

### `parameters(self)`
- Returns all learnable parameters
- Includes attention and FFN parameters

---

## Class: VariationalLatentEncoder

### `__init__(self, input_dim: int, latent_dim: int)`
- Initializes variational encoder
- Creates mean network
- Creates log-variance network

### `encode(self, x: Tensor) -> Tuple[Tensor, Tensor]`
- Encodes input to mean and log-variance
- Returns both parameters

### `reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor`
- Implements reparameterization trick
- Samples from learned distribution
- Returns sampled latent z

### `__call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]`
- Full encoding pipeline
- Encodes to mean and logvar
- Samples latent z
- Returns z, mean, and logvar

### `kl_divergence(self, mean: Tensor, logvar: Tensor) -> Tensor`
- Computes KL divergence from standard normal
- Returns KL loss tensor

### `parameters(self)`
- Returns all learnable parameters
- Includes mean and logvar network parameters

---

## Class: HierarchicalSemanticEncoder

### `__init__(self, vocab: AGIVocabulary, dim: int = 256, num_layers: int = 4, num_heads: int = 8, latent_dim: int = 64)`
- Initializes AGI-grade hierarchical encoder
- Creates token projection layer
- Initializes positional encoding
- Creates modality-specific encoders (image, audio)
- Builds transformer layers with AGI attention
- Initializes slot attention mechanism
- Creates true variational latent encoder
- Sets up slot factorizer and relation encoder
- Creates global context and goal networks

### `encode_tokens(self, token_ids: List[int]) -> Tensor`
- Encodes token IDs to embeddings
- Projects embeddings to model dimension
- Returns stacked token embeddings

### `__call__(self, text: str, goal_context: Optional[Tensor] = None) -> Dict[str, Tensor]`
- Hierarchically encodes text to true latent space
- Tokenizes input text
- Adds positional encoding
- Processes through transformer layers with goal modulation
- Applies slot attention for object decomposition
- Encodes to true variational latent space
- Factorizes slots into disentangled factors
- Computes relations between slots
- Returns comprehensive encoding dictionary

### `encode_multimodal(self, observation: Union[...], goal_context: Optional[Tensor] = None) -> Dict[str, Tensor]`
- Encodes multi-modal observations to shared latent space
- Handles text, image, audio, and multimodal inputs
- Preserves spatial structure for images
- Preserves temporal structure for audio
- Fuses multiple modalities in latent space
- Returns unified latent representation

### `parameters(self)`
- Returns all learnable parameters
- Includes all encoder components

---

## Class: ContrastiveSemanticEncoder

### `__init__(self, base_encoder: HierarchicalSemanticEncoder, projection_dim: int = 128)`
- Initializes contrastive learning encoder
- Wraps base hierarchical encoder
- Creates projection head for contrastive space
- Sets temperature parameter

### `project(self, z: Tensor) -> Tensor`
- Projects latent to contrastive space
- Applies L2 normalization
- Returns normalized projection

### `contrastive_loss(self, z_i: Tensor, z_j: Tensor, negatives: List[Tensor]) -> Tensor`
- Computes InfoNCE contrastive loss
- Projects anchor, positive, and negatives
- Computes similarities with temperature scaling
- Returns contrastive loss value

### `parameters(self)`
- Returns all learnable parameters
- Includes encoder and projection head parameters

---

## Class: SemanticSimilarityComputer

### `cosine_similarity(z1: Tensor, z2: Tensor) -> float` (static)
- Computes cosine similarity between latents
- Normalizes by vector norms
- Returns similarity score

### `euclidean_distance(z1: Tensor, z2: Tensor) -> float` (static)
- Computes Euclidean distance
- Returns distance value

### `mahalanobis_distance(z1: Tensor, z2: Tensor, cov_inv: np.ndarray) -> float` (static)
- Computes Mahalanobis distance
- Uses inverse covariance matrix
- Returns distance value

### `kl_divergence_gaussian(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> float` (static)
- Computes KL divergence between two Gaussians
- Returns KL divergence value

---

## Class: SlotAttention

### `__init__(self, num_slots: int, dim: int, max_iters: int = 10, convergence_threshold: float = 1e-4)`
- Initializes AGI-grade slot attention
- Creates learned slot initialization parameters
- Sets up query, key, value projections
- Creates GRU and MLP for slot updates
- Configures adaptive iteration with convergence detection

### `__call__(self, tokens: Tensor) -> Tensor`
- Extracts object slots from token features
- Initializes slots from learned distribution
- Iteratively refines slots with attention
- Detects convergence and stops early
- Tracks attention weights for visualization
- Returns final slot representations

### `get_attention_weights(self) -> Optional[np.ndarray]`
- Returns last computed attention weights
- Used for visualization and interpretability

### `get_convergence_info(self) -> Dict[str, Any]`
- Returns convergence information
- Includes iterations used and convergence status

### `parameters(self)`
- Returns all learnable parameters
- Includes slot parameters and networks

---

## Class: SlotFactorizer

### `__init__(self, dim: int)`
- Initializes slot factorizer
- Creates heads for type, state, properties, embedding
- Each factor gets sufficient capacity (32 dims)

### `__call__(self, slots: Tensor) -> Dict[str, Tensor]`
- Splits slots into disentangled factors
- Returns dictionary with all factor types

### `parameters(self)`
- Returns all learnable parameters
- Includes all factor head parameters

---

## Class: RelationEncoder

### `__init__(self, slot_dim: int, rel_dim: int = 64)`
- Initializes relation encoder
- Creates MLP for pairwise relations
- Sets relation dimensionality

### `__call__(self, slots: Tensor) -> Tensor`
- Builds pairwise relation tensor
- Uses vectorized computation for efficiency
- Processes all slot pairs through MLP
- Returns relation tensor (N, N, rel_dim)

### `parameters(self)`
- Returns all learnable parameters
- Includes relation MLP parameters

---

## Class: AGISemanticEncoder

### `__init__(self, embedding_dim: int = 64, latent_dim: int = 256, num_layers: int = 4, num_heads: int = 8)`
- Initializes AGI-grade unified semantic encoder
- Creates vocabulary
- Builds hierarchical encoder with AGI attention
- Wraps with contrastive encoder
- Initializes similarity computer
- Creates sensory observer for multi-modal input
- Sets up encoding cache

### `encode(self, text: str, use_cache: bool = True, goal_context: Optional[Tensor] = None) -> Dict[str, Tensor]`
- Encodes text to semantic latent representation
- Uses cache for efficiency if enabled
- Applies goal-directed modulation if provided
- Returns multi-scale representation dictionary

### `encode_observation(self, observation: Union[...], goal_context: Optional[Tensor] = None) -> Dict[str, Tensor]`
- Encodes any sensory observation
- Handles text, image, audio, or multi-modal
- Auto-detects modality if needed
- Returns unified latent representation

### `encode_batch(self, texts: List[str], goal_context: Optional[Tensor] = None) -> List[Dict[str, Tensor]]`
- Batch encodes multiple texts
- Supports optional parallelization
- Returns list of encoding dictionaries

### `encode_batch_to_latents(self, texts: List[str]) -> Tensor`
- Batch encodes texts directly to latent matrix
- Optimized for downstream tasks
- Returns stacked latent tensor

### `get_latent(self, text: str, goal_context: Optional[Tensor] = None) -> Tensor`
- Gets compact latent representation
- Returns 384-dim latent vector (6×64)

### `get_world_state(self, input_data: Union[...], goal_context: Optional[Tensor] = None) -> Dict[str, Any]`
- Gets world-state representation from latent space
- Includes latent, slots, relations, uncertainty
- Includes VAE losses
- Returns comprehensive world-state dictionary

### `compute_similarity(self, text1: str, text2: str, metric: str = 'cosine') -> float`
- Computes semantic similarity in latent space
- Supports cosine, euclidean, or KL metrics
- Returns similarity score

### `get_uncertainty(self, text: str) -> float`
- Gets uncertainty estimate for encoding
- Computes mean per-slot variance
- Returns uncertainty value

### `get_attention_maps(self, text: str) -> Dict[str, Any]`
- Extracts attention maps for visualization
- Returns slot attention weights
- Includes convergence information

### `interpolate_latents(self, text1: str, text2: str, steps: int = 10) -> List[Tensor]`
- Interpolates between two latent representations
- Enables semantic morphing
- Returns list of interpolated latents

### `decode_latent(self, latent_z: Tensor) -> Dict[str, Tensor]`
- Decodes latent back to slot space
- Reconstructs slots through VAE decoder
- Factorizes and computes relations
- Returns reconstruction dictionary

### `analyze_latent_space(self, texts: List[str]) -> Dict[str, Any]`
- Comprehensive latent space analysis
- Analyzes coverage, clusters, organization
- Computes statistics and distances
- Returns analysis dictionary

### `clear_cache(self)`
- Clears encoding cache
- Frees memory

### `parameters(self)`
- Returns all learnable parameters
- Includes contrastive encoder parameters

### `integrate_with_core(self, core: 'UltimateAGICognitiveCore')`
- Integrates encoder with AGI cognitive core
- Enables neuro-symbolic grounding
- Augments perception function

### `integrate_with_world_model(self, world_model: 'WorldModel')`
- Integrates encoder with world model
- Adds temporal prediction capabilities
- Adds counterfactual imagination
- Adds multi-scenario comparison
- Creates predict_world_state method
- Creates imagine_scenario method
- Creates compare_scenarios method

---

## Summary Statistics

**Classes:** 14
**Total Methods:** 67
**Key Features:**
- Multi-modal encoding (text, image, audio)
- True variational latent space with VAE
- Object-centric slot attention
- Disentangled factor representation
- Relation encoding between objects
- Goal-directed modulation
- Adaptive convergence detection
- Attention visualization
- Latent space interpolation
- Comprehensive analysis tools
- Integration with AGI core and world model
