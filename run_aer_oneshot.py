import json
import argparse
import os
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import AsyncAzureOpenAI
import asyncio


# ================= 1. 配置和辅助函数 =================

def load_config(config_path: str = "config.txt") -> Dict[str, str]:
    """从txt文件加载Azure OpenAI配置"""
    config = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 未找到。请确保该文件存在于当前目录。")
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    required_keys = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件 {config_path} 中缺少必需的键: {key}")
    return config


def initialize_llm(config: Dict[str, str]):
    """
    初始化Azure OpenAI客户端
    返回: (client, deployment_name)
    """
    endpoint = config['AZURE_OPENAI_ENDPOINT']

    # 从 URL 中解析 deployment name 和 api-version
    match = re.match(
        r"(https?://[^/]+)/openai/deployments/([^/]+)/chat/completions\?api-version=([^&]+)",
        endpoint,
    )
    if not match:
        raise ValueError(
            f"AZURE_OPENAI_ENDPOINT 格式不正确。\n"
            f"期望格式: https://YOUR_RESOURCE.azure-api.net/openai/deployments/DEPLOYMENT_NAME/chat/completions?api-version=VERSION\n"
            f"实际: {endpoint}"
        )

    azure_endpoint = match.group(1)
    deployment_name = match.group(2)
    api_version = match.group(3)

    print(f"解析结果:")
    print(f"  Endpoint: {azure_endpoint}")
    print(f"  Deployment: {deployment_name}")
    print(f"  API Version: {api_version}")

    client = AsyncAzureOpenAI(
        api_key=config['AZURE_OPENAI_KEY'],
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    return client, deployment_name


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
    """加载 docs.json 文件，创建一个以 topic_id 为键的字典"""
    docs_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            topic_id = item['topic_id']
            docs_map[topic_id] = item
    return docs_map


def build_zero_shot_prompt(target_question: Dict, target_docs: List) -> str:
    """Zero-shot Prompt"""
    prompt = f"## 目标事件\n{target_question['target_event']}\n\n"
    prompt += f"## 相关文档\n"
    for doc in target_docs[:3]:
        prompt += f"- {doc['snippet']}\n"
    prompt += f"\n## 候选原因\n"
    prompt += f"A. {target_question['option_A']}\n"
    prompt += f"B. {target_question['option_B']}\n"
    prompt += f"C. {target_question['option_C']}\n"
    prompt += f"D. {target_question['option_D']}\n"
    prompt += "\n请只输出正确答案的字母（例如 A, B, C, D 或组合如 AB, CD）。你的答案："
    return prompt


def build_one_shot_prompt(example_question: Dict, example_docs: List, target_question: Dict, target_docs: List) -> str:
    """One-shot Prompt（包含1个示例）"""
    # 示例部分
    example_context = "示例：\n"
    example_context += f"## 目标事件\n{example_question['target_event']}\n\n"
    example_context += f"## 相关文档\n"
    for doc in example_docs[:3]:
        example_context += f"- {doc['snippet']}\n"
    example_context += f"\n## 候选原因\n"
    example_context += f"A. {example_question['option_A']}\n"
    example_context += f"B. {example_question['option_B']}\n"
    example_context += f"C. {example_question['option_C']}\n"
    example_context += f"D. {example_question['option_D']}\n"
    example_context += f"\n## 正确答案\n{example_question['golden_answer']}\n\n"
    example_context += "---\n\n"

    # 实际问题部分
    query_context = "现在，请回答以下问题：\n"
    query_context += f"## 目标事件\n{target_question['target_event']}\n\n"
    query_context += f"## 相关文档\n"
    for doc in target_docs[:3]:
        query_context += f"- {doc['snippet']}\n"
    query_context += f"\n## 候选原因\n"
    query_context += f"A. {target_question['option_A']}\n"
    query_context += f"B. {target_question['option_B']}\n"
    query_context += f"C. {target_question['option_C']}\n"
    query_context += f"D. {target_question['option_D']}\n"
    query_context += "\n请只输出正确答案的字母（例如 A, B, C, D 或组合如 AB, CD）。你的答案："

    return example_context + query_context


def build_few_shot_prompt(example_questions: List[Dict], example_docs_list: List[List], target_question: Dict,
                          target_docs: List, num_shots: int = 3) -> str:
    """Few-shot Prompt（包含多个示例）"""
    prompt = f"以下是 {num_shots} 个示例：\n\n"

    # 添加多个示例
    for i, (ex_q, ex_docs) in enumerate(zip(example_questions[:num_shots], example_docs_list[:num_shots])):
        prompt += f"### 示例 {i + 1}\n"
        prompt += f"**目标事件**: {ex_q['target_event']}\n\n"
        prompt += f"**相关文档**:\n"
        for doc in ex_docs[:2]:  # 示例中只用2个文档，节省token
            prompt += f"- {doc['snippet']}\n"
        prompt += f"\n**候选原因**:\n"
        prompt += f"A. {ex_q['option_A']}\n"
        prompt += f"B. {ex_q['option_B']}\n"
        prompt += f"C. {ex_q['option_C']}\n"
        prompt += f"D. {ex_q['option_D']}\n"
        prompt += f"**正确答案**: {ex_q['golden_answer']}\n\n"

    # 实际问题部分
    prompt += "---\n\n"
    prompt += "### 现在请回答以下问题\n\n"
    prompt += f"**目标事件**: {target_question['target_event']}\n\n"
    prompt += f"**相关文档**:\n"
    for doc in target_docs[:3]:
        prompt += f"- {doc['snippet']}\n"
    prompt += f"\n**候选原因**:\n"
    prompt += f"A. {target_question['option_A']}\n"
    prompt += f"B. {target_question['option_B']}\n"
    prompt += f"C. {target_question['option_C']}\n"
    prompt += f"D. {target_question['option_D']}\n"
    prompt += "\n请只输出正确答案的字母（例如 A, B, C, D 或组合如 AB, CD）。你的答案："

    return prompt


def build_cot_prompt(target_question: Dict, target_docs: List, is_few_shot: bool = False,
                     example_questions: List[Dict] = None, example_docs_list: List[List] = None) -> str:
    """
    Chain-of-Thought (CoT) Prompt
    引导模型逐步推理
    """
    system_prompt = """你是一个擅长溯因推理的助手。请按照以下步骤进行推理：

步骤1: 理解目标事件 - 这个事件描述了什么？
步骤2: 分析相关文档 - 文档中提供了哪些关键信息？
步骤3: 评估每个候选原因 - 哪个原因最合理？为什么？
步骤4: 得出结论 - 选出最合理的原因。

请按步骤输出你的推理过程，最后一行输出答案格式为："答案: X" 或 "答案: AB"。
"""

    if is_few_shot and example_questions:
        # Few-shot CoT: 先给示例的推理过程
        prompt = "以下是带有推理过程的示例：\n\n"
        for i, (ex_q, ex_docs) in enumerate(zip(example_questions[:2], example_docs_list[:2])):
            prompt += f"### 示例 {i + 1}\n"
            prompt += f"**目标事件**: {ex_q['target_event']}\n\n"
            prompt += f"**相关文档**:\n"
            for doc in ex_docs[:2]:
                prompt += f"- {doc['snippet']}\n"
            prompt += f"\n**候选原因**:\n"
            prompt += f"A. {ex_q['option_A']}\n"
            prompt += f"B. {ex_q['option_B']}\n"
            prompt += f"C. {ex_q['option_C']}\n"
            prompt += f"D. {ex_q['option_D']}\n"
            prompt += f"\n**推理过程**:\n"
            prompt += f"1. 目标事件是：{ex_q['target_event']}\n"
            prompt += f"2. 文档指出：{ex_docs[0]['snippet'] if ex_docs else '无'}\n"
            prompt += f"3. 分析：{ex_q['option_A'] if ex_q['golden_answer'] == 'A' else ex_q['option_B']} 是最合理的原因\n"
            prompt += f"**正确答案**: {ex_q['golden_answer']}\n\n"

        prompt += "---\n\n"
        prompt += "### 现在请按照同样的推理步骤回答以下问题\n\n"
    else:
        # Zero-shot CoT
        prompt = ""

    prompt += f"**目标事件**: {target_question['target_event']}\n\n"
    prompt += f"**相关文档**:\n"
    for doc in target_docs[:3]:
        prompt += f"- {doc['snippet']}\n"
    prompt += f"\n**候选原因**:\n"
    prompt += f"A. {target_question['option_A']}\n"
    prompt += f"B. {target_question['option_B']}\n"
    prompt += f"C. {target_question['option_C']}\n"
    prompt += f"D. {target_question['option_D']}\n"
    prompt += "\n请按步骤推理，最后输出答案。"

    return system_prompt, prompt


async def get_llm_prediction(client: AsyncAzureOpenAI, prompt: str, deployment_name: str, is_cot: bool = False) -> \
tuple[str, Optional[str]]:
    """
    调用LLM并返回预测的答案字符串
    如果是CoT，同时返回推理过程
    """
    try:
        messages = []
        if is_cot:
            # CoT模式：system prompt包含推理指令
            system_prompt, user_prompt = prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [
                {"role": "system",
                 "content": "你是一个擅长进行溯因推理（Abductive Reasoning）的助手。请根据目标事件和相关文档，从候选原因中选出最合理的一个或多个。"},
                {"role": "user", "content": prompt}
            ]

        response = await client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.1,
            max_tokens=500 if is_cot else 50  # CoT需要更多token
        )

        full_response = response.choices[0].message.content.strip()

        if is_cot:
            # 提取推理过程和答案
            # 尝试从最后一行或"答案:"后面提取答案
            answer_match = re.search(r'答案[：:]\s*([A-D](?:[,\s]*[A-D])*)', full_response, re.IGNORECASE)
            if answer_match:
                prediction = answer_match.group(1).replace(" ", "").replace(",", ",")
            else:
                # 如果没找到"答案:"，尝试找单独的字母
                letter_match = re.findall(r'\b([A-D])\b', full_response.upper())
                if letter_match:
                    # 去重并保持顺序
                    seen = set()
                    prediction = ",".join([c for c in letter_match if not (c in seen or seen.add(c))])
                else:
                    prediction = ""
            return prediction, full_response
        else:
            # 非CoT模式：直接提取答案
            match = re.search(r'([A-D](?:,\s*[A-D])*)', full_response.upper())
            if match:
                prediction = match.group(1).replace(" ", "")
            else:
                if full_response and full_response[0].upper() in ['A', 'B', 'C', 'D']:
                    prediction = full_response[0].upper()
                else:
                    prediction = ""
            return prediction, None

    except Exception as e:
        print(f"LLM调用出错: {e}")
        return "", None


# ================= 2. 核心评估逻辑 =================

def evaluate_predictions(predictions: List[str], golden_answers: List[str]) -> float:
    """根据官方指标计算平均分（1.0, 0.5, 0.0）"""
    total_score = 0.0
    details = []
    for pred, gold in zip(predictions, golden_answers):
        pred_set = set(pred.split(',')) if pred else set()
        gold_set = set(gold.split(','))

        if not pred_set:
            score = 0.0
            status = "empty"
        elif pred_set == gold_set:
            score = 1.0
            status = "full_match"
        elif pred_set.issubset(gold_set) and pred_set:
            score = 0.5
            status = "partial_match"
        else:
            score = 0.0
            status = "incorrect"
        total_score += score
        details.append({"pred": pred, "gold": gold, "score": score, "status": status})
    return total_score / len(predictions) if predictions else 0.0, details


# ================= 3. 主函数 =================

async def async_main(args):
    print("正在加载配置...")
    config = load_config(args.config)
    client, deployment_name = initialize_llm(config)
    print("LLM客户端初始化成功。")

    print("正在加载数据集...")
    questions_path = os.path.join(args.data_dir, args.split, "questions.jsonl")
    docs_path = os.path.join(args.data_dir, args.split, "docs.json")

    if not os.path.exists(questions_path) or not os.path.exists(docs_path):
        raise FileNotFoundError(f"在目录 {args.data_dir}/{args.split} 下未找到 questions.jsonl 或 docs.json")

    questions = load_jsonl(questions_path)
    docs_map = load_docs_json(docs_path)
    print(f"加载了 {len(questions)} 个问题实例。")

    # 准备示例（用于 One-shot, Few-shot, CoT-Few-shot）
    examples = []
    example_docs_list = []
    if args.method in ["oneshot", "fewshot", "cot_fewshot"] and len(questions) > args.num_shots:
        # 选择前 num_shots 个样本作为示例
        for i in range(min(args.num_shots, len(questions) - 1)):
            examples.append(questions[i])
            example_docs_list.append(docs_map.get(questions[i]['topic_id'], {}).get('docs', []))
        print(f"准备了 {len(examples)} 个示例样本")

        if args.method == "oneshot":
            # One-shot 只用第一个示例
            example_question = examples[0]
            example_docs = example_docs_list[0]
            start_idx = 1
            target_questions = questions[1:]
        else:
            # Few-shot 或 CoT few-shot 用多个示例，从示例数量后开始
            start_idx = len(examples)
            target_questions = questions[start_idx:]
    else:
        # Zero-shot 或 Zero-shot CoT
        start_idx = 0
        target_questions = questions
        example_question = None
        example_docs = None
        examples = []
        example_docs_list = []

    # 存储预测结果
    predictions = []
    golden_answers = []
    reasoning_traces = [] if args.method.startswith("cot") else None

    print(f"使用 {args.method.upper()} 方法")
    print(f"待预测样本数: {len(target_questions)}")

    # 使用tqdm显示进度条
    for i, q in enumerate(tqdm(target_questions, desc="进行预测")):
        target_docs = docs_map.get(q['topic_id'], {}).get('docs', [])

        # 根据方法构建 Prompt
        if args.method == "zeroshot":
            prompt = build_zero_shot_prompt(q, target_docs)
            pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, is_cot=False)

        elif args.method == "oneshot":
            prompt = build_one_shot_prompt(example_question, example_docs, q, target_docs)
            pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, is_cot=False)

        elif args.method == "fewshot":
            prompt = build_few_shot_prompt(examples, example_docs_list, q, target_docs, num_shots=args.num_shots)
            pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, is_cot=False)

        elif args.method == "cot_zeroshot":
            prompt = build_cot_prompt(q, target_docs, is_few_shot=False)
            pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, is_cot=True)
            if reasoning:
                reasoning_traces.append(reasoning)

        elif args.method == "cot_fewshot":
            prompt = build_cot_prompt(q, target_docs, is_few_shot=True, example_questions=examples,
                                      example_docs_list=example_docs_list)
            pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, is_cot=True)
            if reasoning:
                reasoning_traces.append(reasoning)
        else:
            raise ValueError(f"未知的方法: {args.method}")

        predictions.append(pred)
        golden_answers.append(q['golden_answer'])

        # 每20个样本打印一次进度和当前分数
        if (i + 1) % 20 == 0:
            current_score, _ = evaluate_predictions(predictions, golden_answers)
            print(f"\n  已预测 {i + 1} 个样本，当前平均分: {current_score:.4f}")

    # 评估结果
    avg_score, score_details = evaluate_predictions(predictions, golden_answers)

    print(f"\n{'=' * 60}")
    print(f"=== 评估结果 ({args.method.upper()}) ===")
    print(f"数据集: {args.split}")
    print(f"样本数量: {len(predictions)}")
    if args.method in ["fewshot", "cot_fewshot"]:
        print(f"示例数量: {args.num_shots}")
    print(f"平均分 (官方指标): {avg_score:.4f}")
    print(f"{'=' * 60}")

    # 统计各类别的数量
    full_matches = sum(1 for d in score_details if d['status'] == 'full_match')
    partial_matches = sum(1 for d in score_details if d['status'] == 'partial_match')
    incorrect = sum(1 for d in score_details if d['status'] == 'incorrect')
    empty = sum(1 for d in score_details if d['status'] == 'empty')

    print(f"\n详细统计:")
    print(f"  完全匹配 (1.0分): {full_matches} ({full_matches / len(predictions) * 100:.1f}%)")
    print(f"  部分匹配 (0.5分): {partial_matches} ({partial_matches / len(predictions) * 100:.1f}%)")
    print(f"  错误/空预测 (0分): {incorrect + empty} ({(incorrect + empty) / len(predictions) * 100:.1f}%)")

    # 打印样例预测
    print(f"\n样例预测（前10个）:")
    for i in range(min(10, len(predictions))):
        status_icon = "✓" if score_details[i]['score'] == 1.0 else "◐" if score_details[i]['score'] == 0.5 else "✗"
        print(f"  {status_icon} 样本 {i + 1}: 预测={predictions[i]:<5} 正确答案={golden_answers[i]}")

    # 如果CoT模式，打印一个推理样例
    if args.method.startswith("cot") and reasoning_traces and len(reasoning_traces) > 0:
        print(f"\n=== CoT 推理样例 ===")
        print(reasoning_traces[0][:800] + "..." if len(reasoning_traces[0]) > 800 else reasoning_traces[0])

    # 保存结果到文件
    if args.output:
        output_file = args.output
        results = {
            "method": args.method,
            "split": args.split,
            "num_samples": len(predictions),
            "num_shots": args.num_shots if args.method in ["fewshot", "cot_fewshot"] else (
                1 if args.method == "oneshot" else 0),
            "avg_score": avg_score,
            "full_matches": full_matches,
            "partial_matches": partial_matches,
            "incorrect": incorrect + empty,
            "predictions": predictions,
            "golden_answers": golden_answers,
            "score_details": score_details
        }
        if reasoning_traces:
            results["reasoning_traces"] = reasoning_traces[:10]  # 只保存前10个推理过程

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存至 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SemEval2026 Task12 AER - 多种推理方法实验")
    parser.add_argument("--data_dir", type=str, default="./semeval2026-task12-dataset", help="数据集根目录")
    parser.add_argument("--split", type=str, default="sample_data", choices=["sample_data", "train_data", "dev_data"],
                        help="要运行的数据集分割")
    parser.add_argument("--method", type=str, default="zeroshot",
                        choices=["zeroshot", "oneshot", "fewshot", "cot_zeroshot", "cot_fewshot"],
                        help="实验方法: zeroshot, oneshot, fewshot, cot_zeroshot, cot_fewshot")
    parser.add_argument("--num_shots", type=int, default=3, help="Few-shot 示例数量 (默认: 3)")
    parser.add_argument("--config", type=str, default="config.txt", help="包含Azure OpenAI配置的txt文件路径")
    parser.add_argument("--output", type=str, default=None, help="保存预测结果的JSON文件路径（可选）")
    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()