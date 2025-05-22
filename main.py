import argparse
from llm_infer import infer_dataset
from model_selector import llm_model_selector

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based Log Template Inference")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSONL file.")
    parser.add_argument("--method", type=str, default="naive", choices=[
        "naive", "contrastive", "contrastive_v2", "demonstration", "fixed_demonstration"
    ], help="Inference method.")
    parser.add_argument("--demo_path", type=str, default=None, help="Path to demonstration data (for demonstration methods).")
    parser.add_argument("--max_shots", type=int, default=5, help="Max number of few-shot demonstrations.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    llm = llm_model_selector()

    # 调用推理函数
    infer_dataset(
        llm=llm,
        data_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        max_shots=args.max_shots,
        demo_path=args.demo_path
    )

if __name__ == "__main__":
    main()
