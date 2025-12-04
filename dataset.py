import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List
from datasets import load_dataset
from tqdm import tqdm

# huggingface-cli login

class WikiTextDataset(Dataset):
    """
    Dataset for WikiText that returns fixed-length token sequences.

    TokCom v5 数据处理规则:
    1. max_length = 1024 tokens (固定长度)
    2. 不添加 BOS token，只在末尾添加 EOS token
    3. 如果 text 不足 max_length，用 EOS 补齐到 1024
    4. 如果 text 超过 max_length，截断到 1024，超出部分继续划分成新的 sample
    5. 每个 sample 会被非重叠地划分成 window_length=4 的窗口
    """

    def __init__(
        self,
        dataset_name: str,
        chunk_size: int,  # window_length = 4
        tokenizer_path: str,
        split: str = "train",
        max_length: int = 1024,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., "wikitext-2-raw-v1")
            chunk_size: Size of each chunk / window_length (4 tokens per latent vector)
            tokenizer_path: Path to tokenizer directory
            split: Dataset split ("train", "validation", "test")
            max_length: Maximum sequence length (1024 tokens)
        """
        self.chunk_size = chunk_size
        self.max_length = max_length

        # Ensure max_length is divisible by chunk_size
        assert max_length % chunk_size == 0, f"max_length ({max_length}) must be divisible by chunk_size ({chunk_size})"
        self.num_chunks_per_sample = max_length // chunk_size

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.bos_token_id = self.tokenizer.bos_token_idxs
        self.eos_token_id = self.tokenizer.eos_token_id

        # Load dataset
        print(f"Loading dataset {dataset_name} (split: {split})...")
        dataset = load_dataset("wikitext", dataset_name, split=split)

        # Process texts into fixed-length samples
        print(f"Processing texts (max_length={max_length}, chunk_size={chunk_size})...")
        self.samples = self._process_texts(dataset)

        print(f"Dataset ready: {len(self.samples)} samples")

    def _process_texts(self, dataset) -> List[dict]:
        """
        Process texts into fixed-length samples (1024 tokens each).

        处理规则（每条 txt 独立处理）:
        1. 每条文本单独 tokenize，末尾添加 EOS
        2. 短文本（< max_length）：用 EOS 补齐到 1024
        3. 长文本（> max_length）：截断成多个 sample，每个 1024 tokens
        """
        samples = []
        total_tokens = 0

        for item in tqdm(dataset, desc="Tokenizing texts"):
            text = item['text'].strip()

            if not text:
                continue

            # Tokenize (不添加任何特殊 token)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if len(token_ids) == 0:
                continue

            # 添加 EOS token 在文本末尾
            token_ids =[self.bos_token_id] + token_ids + [self.eos_token_id]
            total_tokens += len(token_ids)

            # 按 max_length 划分这条文本
            idx = 0
            while idx < len(token_ids):
                # 取 max_length 个 tokens
                sample_tokens = token_ids[idx:idx + self.max_length]

                # 如果不足 max_length，用 EOS 补齐
                if len(sample_tokens) < self.max_length:
                    padding_length = self.max_length - len(sample_tokens)
                    sample_tokens = sample_tokens + [self.eos_token_id] * padding_length

                # 划分成 chunks
                chunks = []
                for chunk_start in range(0, self.max_length, self.chunk_size):
                    chunk = sample_tokens[chunk_start:chunk_start + self.chunk_size]
                    chunks.append(chunk)

                chunks_tensor = torch.tensor(chunks, dtype=torch.long)  # (num_chunks, chunk_size)

                samples.append({
                    'input_ids': chunks_tensor,
                    'num_chunks': self.num_chunks_per_sample
                })

                idx += self.max_length

        print(f"Total tokens collected: {total_tokens}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary containing:
                - input_ids: tensor of shape (num_chunks, chunk_size) = (256, 4)
                - num_chunks: int (always 256 for max_length=1024, chunk_size=4)
        """
        return self.samples[idx]


def make_collate_fn(eos_token_id: int):
    """
    Create a collate function with specified EOS token for padding.

    Note: In v5, all samples have the same shape (num_chunks, chunk_size),
    so no padding is needed across samples.
    """
    def collate_fn(batch):
        """
        Custom collate function.

        Since all samples have fixed length in v5, this is straightforward.

        Returns:
            - input_ids: (batch_size, num_chunks, chunk_size)
            - num_chunks: (batch_size,) - all the same value
            - mask: (batch_size, num_chunks) - all True
        """
        batch_size = len(batch)
        num_chunks = batch[0]['num_chunks']
        chunk_size = batch[0]['input_ids'].shape[1]

        # Stack all samples
        input_ids = torch.stack([sample['input_ids'] for sample in batch], dim=0)
        num_chunks_tensor = torch.tensor([sample['num_chunks'] for sample in batch], dtype=torch.long)
        mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)

        return {
            'input_ids': input_ids,
            'num_chunks': num_chunks_tensor,
            'mask': mask
        }

    return collate_fn


# Example usage
if __name__ == "__main__":
    print("Testing WikiText Dataset (TokCom v5)...")
    print("=" * 60)
    print("v5 特点:")
    print("  - max_length = 1024 tokens (固定长度)")
    print("  - chunk_size = 4 (window_length)")
    print("  - num_chunks = 256 per sample")
    print("  - 不添加 BOS，只有 EOS")
    print("  - 超长文本截断并保留剩余部分")
    print("=" * 60)

    dataset = WikiTextDataset(
        dataset_name="wikitext-2-raw-v1",
        chunk_size=4,
        tokenizer_path="meta-llama/CodeLlama-7b-hf",
        split="train",
        max_length=1024
    )

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Chunk size: {dataset.chunk_size}")
    print(f"  Max length: {dataset.max_length}")
    print(f"  Chunks per sample: {dataset.num_chunks_per_sample}")

    # Test first few samples
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: shape={sample['input_ids'].shape}, num_chunks={sample['num_chunks']}")

    # Create dataloader
    collate_fn = make_collate_fn(dataset.eos_token_id)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Test batch
    print(f"\nTesting batch iteration:")
    for batch in dataloader:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  num_chunks: {batch['num_chunks'].tolist()}")
        print(f"  mask shape: {batch['mask'].shape}")
        print(f"  All masks True: {batch['mask'].all().item()}")
        break

    print("\nDataset module test completed!")
