import argparse
import mlx.core as mx

from aether.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text from a saved checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/bigram")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_checkpoint(args.checkpoint)
    prompt_ids = mx.array([tokenizer.encode(args.prompt)])
    generated = model.generate(prompt_ids, args.max_new_tokens)
    print(tokenizer.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()