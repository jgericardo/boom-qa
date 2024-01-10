from transformers import AutoModel, AutoTokenizer


def main():
    # Base model name to use for the QA search from sentence-transformers
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

    # If model files exist locally, load them from disk.
    # Otherwise, download them from HuggingFace's model hub.
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[INFO] Model files have been downloaded.")

    # Save the model files locally if they haven't been saved yet.
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    print(f"[INFO] Model files have been saved locally in {model_name}")


if __name__ == "__main__":
    main()
