import json
import argparse
import os
import re
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


# ================= 1. 配置和辅助函数 =================

def load_config(config_path: str = "config.txt") -> Dict[str, str]:
    """从txt文件加载Azure OpenAI配置"""
    config = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 未找到。请确保该文件存在于当前目录。")
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    required_keys = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件 {config_path} 中缺少必需的键: {key}")
    return config


def initialize_llm(config: Dict[str, str]) -> AzureOpenAI:
    """初始化Azure OpenAI客户端"""
    client = AzureOpenAI(
        azure_endpoint=config['AZURE_OPENAI_ENDPOINT'],
        api_key=config['AZURE_OPENAI_KEY'],
        api_version="2024-02-15-preview",  # 使用一个稳定的API版本
    )
    return client


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 .jsonl 文件，每行一个JSON对象"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_docs_json(file_path: str) -> Dict[int, Dict[str, Any]]:
    """加载 docs.json 文件，创建一个以 topic_id 为键的字典，方便快速查找"""
    docs_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 整个文件是一个JSON对象列表
        for item in data:
            topic_id = item['topic_id']
            docs_map[topic_id] = item
    return docs_map


def build_one_shot_prompt(example_question: Dict, example_docs: List, target_question: Dict, target_docs: List) -> str:
    """
    构建一个包含一个示例（One-shot）的Prompt。
    example_* 是作为示例的样本，target_* 是待预测的样本。
    """
    # 1. 构建示例部分
    example_context = "示例：\n"
    example_context += f"## 目标事件\n{example_question['target_event']}\n\n"
    example_context += f"## 相关文档\n"
    for doc in example_docs[:3]:  # 限制文档数量，避免prompt过长，取前3个即可
        example_context += f"- {doc['snippet']}\n"
    example_context += f"\n## 候选原因\nA. {example_question['option_A']}\nB. {example_question['option_B']}\nC. {example_question['option_C']}\nD. {example_question['option_D']}\n"
    example_context += f"\n## 正确答案\n{example_question['golden_answer']}\n\n"
    example_context += "---\n\n"

    # 2. 构建实际问题部分
    query_context = "现在，请回答以下问题：\n"
    query_context += f"## 目标事件\n{target_question['target_event']}\n\n"
    query_context += f"## 相关文档\n"
    for doc in target_docs[:3]:  # 同样限制文档数量
        query_context += f"- {doc['snippet']}\n"
    query_context += f"\n## 候选原因\nA. {target_question['option_A']}\nB. {target_question['option_B']}\nC. {target_question['option_C']}\nD. {target_question['option_D']}\n"
    query_context += "\n请只输出正确答案的字母（例如 A, B, C, D 或组合如 AB, CD）。你的答案："

    return example_context + query_context


def get_llm_prediction(client: AzureOpenAI, prompt: str, deployment_name: str) -> str:
    """调用LLM并返回预测的答案字符串"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system",
                 "content": "你是一个擅长进行溯因推理（Abductive Reasoning）的助手。请根据目标事件和相关文档，从候选原因中选出最合理的一个或多个。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度使输出更确定
            max_tokens=50
        )
        prediction = response.choices[0].message.content.strip()
        # 提取答案字母（例如，从 "A" 或 "答案是 B" 或 "A, B" 中提取）
        match = re.search(r'([A-D](?:,\s*[A-D])*)', prediction.upper())
        if match:
            return match.group(1).replace(" ", "")
        else:
            # 如果正则匹配失败，尝试直接取前几个字符
            return prediction[:1].upper()
    except Exception as e:
        print(f"LLM调用出错: {e}")
        return ""  # 返回空字符串表示预测失败


# ================= 2. 核心评估逻辑 =================

def evaluate_predictions(predictions: List[str], golden_answers: List[str]) -> float:
    """根据官方指标计算平均分（1.0, 0.5, 0.0）"""
    total_score = 0.0
    for pred, gold in zip(predictions, golden_answers):
        pred_set = set(pred.split(',')) if pred else set()
        gold_set = set(gold.split(','))

        if not pred_set:
            score = 0.0
        elif pred_set == gold_set:
            score = 1.0
        elif pred_set.issubset(gold_set) and pred_set:  # 非空真子集
            score = 0.5
        else:
            score = 0.0
        total_score += score
    return total_score / len(predictions) if predictions else 0.0


# ================= 3. 主函数 =================

def main(args):
    print("正在加载配置...")
    config = load_config(args.config)
    client = initialize_llm(config)
    print("LLM客户端初始化成功。")

    print("正在加载数据集...")
    questions_path = os.path.join(args.data_dir, args.split, "questions.jsonl")
    docs_path = os.path.join(args.data_dir, args.split, "docs.json")

    if not os.path.exists(questions_path) or not os.path.exists(docs_path):
        raise FileNotFoundError(f"在目录 {args.data_dir}/{args.split} 下未找到 questions.jsonl 或 docs.json")

    questions = load_jsonl(questions_path)
    docs_map = load_docs_json(docs_path)
    print(f"加载了 {len(questions)} 个问题实例。")

    # 为了One-shot，我们选择第一个样本作为示例（也可以固定选择一个逻辑清晰的样本）
    # 注意：如果使用训练集，示例样本的答案应该是已知的。这里我们简单取第一个。
    example_question = questions[0]
    example_docs = docs_map.get(example_question['topic_id'], {}).get('docs', [])

    # 对剩余的所有样本进行预测
    predictions = []
    golden_answers = []
    # 使用tqdm显示进度条
    for i, q in enumerate(tqdm(questions[1:], desc="进行预测")):
        target_docs = docs_map.get(q['topic_id'], {}).get('docs', [])
        # 构建One-shot Prompt
        prompt = build_one_shot_prompt(example_question, example_docs, q, target_docs)
        # 获取LLM预测
        pred = get_llm_prediction(client, prompt, config['AZURE_OPENAI_DEPLOYMENT'])
        predictions.append(pred)
        golden_answers.append(q['golden_answer'])

    # 评估结果
    avg_score = evaluate_predictions(predictions, golden_answers)
    print(f"\n=== 评估结果 (One-shot) ===")
    print(f"数据集: {args.split}")
    print(f"样本数量: {len(predictions)}")
    print(f"平均分 (官方指标): {avg_score:.4f}")

    # 可选：将结果保存到文件
    if args.output:
        output_file = args.output
        with open(output_file, 'w') as f:
            json.dump({"predictions": predictions, "golden_answers": golden_answers, "avg_score": avg_score}, f,
                      indent=2)
        print(f"详细结果已保存至 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemEval2026 Task12 AER - One-shot 基线实验")
    parser.add_argument("--data_dir", type=str, default="./semeval2026-task12-dataset", help="数据集根目录")
    parser.add_argument("--split", type=str, default="sample_data", choices=["sample_data", "train_data", "dev_data"],
                        help="要运行的数据集分割 (sample_data, train_data, dev_data)")
    parser.add_argument("--config", type=str, default="config.txt", help="包含Azure OpenAI配置的txt文件路径")
    parser.add_argument("--output", type=str, default=None, help="保存预测结果的JSON文件路径（可选）")
    args = parser.parse_args()
    main(args)