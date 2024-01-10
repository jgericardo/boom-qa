from transformers import AutoModel, AutoTokenizer

def main():
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[INFO] Model files have been downloaded.")
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    print(f"[INFO] Model files have been saved locally in {model_name}")

if __name__ == "__main__":
    main()

