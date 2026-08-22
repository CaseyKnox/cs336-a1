import argparse
from tests.adapters import run_train_bpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BPE tokenizer trainer.")

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the BPE tokenizer training data corpus."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Total number of items in the tokenizer's vocabulary (including special tokens)."
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="A space-separated list of string special tokens (e.g., --special_tokens <|pad|> <|unk|>)"
    )

    args = parser.parse_args()

    print(f"Starting BPE training on '{args.input_path}'...")
    print(f"Target vocab size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")

    # Run the trainer
    vocab_dict, merges = run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    print(f"Training complete! Generated {len(vocab_dict)} vocab items and {len(merges)} merges.")
    
    # Example usage:
    # uv run your_script.py fixtures/corpus.en --vocab_size 500 --special_tokens "<|endoftext|>"