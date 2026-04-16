"""
SemEval2026 Task12: Abductive Event Reasoning (AER)
UPDATED VERSION: Full document + Context Filtering (proposal required)
"""

import json
import argparse
import os
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
from openai import AsyncAzureOpenAI


# ================= Configuration =================

def load_config(config_path: str = "config.txt") -> Dict[str, str]:
    config = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

    required_keys = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    return config


def initialize_llm(config: Dict[str, str]) -> Tuple[AsyncAzureOpenAI, str]:
    endpoint = config['AZURE_OPENAI_ENDPOINT']
    match = re.match(
        r"(https?://[^/]+)/openai/deployments/([^/]+)/chat/completions\?api-version=([^&]+)",
        endpoint,
    )
    if not match:
        raise ValueError(
            f"AZURE_OPENAI_ENDPOINT format incorrect.\n"
            f"Expected format: https://YOUR_RESOURCE.azure-api.net/openai/deployments/DEPLOYMENT_NAME/chat/completions?api-version=VERSION\n"
            f"Got: {endpoint}"
        )

    azure_endpoint = match.group(1)
    deployment_name = match.group(2)
    api_version = match.group(3)

    print(f"Parsed configuration:")
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
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_docs_json(file_path: str) -> Dict[int, Dict[str, Any]]:
    docs_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            topic_id = item['topic_id']
            docs_map[topic_id] = item
    return docs_map


# ==================== 👇 新增：你们 proposal 要求的上下文过滤函数 ====================
def filter_relevant_documents(event: str, docs: List[Dict], top_k: int = 1) -> str:
    event_words = set(event.lower().split())
    scored_docs = []

    for doc in docs:
        content = doc.get("content", "") or doc.get("snippet", "")
        title = doc.get("title", "")
        text = (title + " " + content).lower()
        doc_words = set(text.split())
        overlap = len(event_words & doc_words)
        scored_docs.append((-overlap, title, content))

    scored_docs.sort()
    selected = []
    for i, (_, title, content) in enumerate(scored_docs[:top_k]):
        selected.append(f"\n[Filtered Document {i+1}] {title}\n{content}\n")

    return "\n".join(selected)


def extract_document_text(doc_item: Dict[str, Any], question: Dict = None, context_mode: str = "full") -> str:
    docs_list = doc_item.get('docs', [])
    if not docs_list:
        return "No documents available."

    # ==================== 👇 新增：上下文过滤 ====================
    if context_mode == "filtered" and question is not None:
        return filter_relevant_documents(question["target_event"], docs_list, top_k=1)

    # 原始 full context
    formatted_docs = []
    for i, doc in enumerate(docs_list):
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '') or doc.get('snippet', '')
        formatted_doc = f"\n{'='*60}\n[Document {i+1}] {title}\n{'='*60}\n{content}\n"
        formatted_docs.append(formatted_doc)
    return "\n".join(formatted_docs)


def normalize_answer(answer: str) -> Set[str]:
    if not answer:
        return set()
    letters = re.findall(r'[A-D]', answer.upper())
    return set(letters)


def format_options(question: Dict[str, Any]) -> str:
    return f"""A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}"""


# ================= Prompt Building =================

def build_zero_shot_prompt(question: Dict, docs_text: str) -> str:
    return f"""# Task: Abductive Event Reasoning

You are given:
1. A target event that occurred
2. Retrieved documents providing context
3. Four candidate causes (A, B, C, D)

Your job: Select the MOST PLAUSIBLE cause(s) that directly led to the target event.

## Target Event
{question['target_event']}

## Retrieved Documents
{docs_text if docs_text else "No documents available."}

## Candidate Causes
{format_options(question)}

## Instructions
- Base your reasoning on the documents AND your general knowledge
- Multiple causes can be correct if they all plausibly contributed to the event
- Output ONLY the letter(s) of your answer
- Format examples: "A", "C", "AB", "BCD", "AD"

Answer:"""


def build_one_shot_prompt(example_q: Dict, example_docs: str, target_q: Dict, target_docs: str) -> str:
    return f"""# Task: Abductive Event Reasoning (with Example)

## Example

### Target Event
{example_q['target_event']}

### Retrieved Documents
{example_docs if example_docs else "No documents available."}

### Candidate Causes
{format_options(example_q)}

### Correct Answer
{example_q['golden_answer']}

---
## Now answer the following question

### Target Event
{target_q['target_event']}

### Retrieved Documents
{target_docs if target_docs else "No documents available."}

### Candidate Causes
{format_options(target_q)}

## Instructions
Output ONLY the letter(s) of your answer (e.g., "A", "C", "AB", "BCD").

Answer:"""


def build_few_shot_prompt(examples: List[Dict], example_docs_list: List[str],
                          target_q: Dict, target_docs: str, num_shots: int = 3) -> str:
    prompt = "# Task: Abductive Event Reasoning\n\n"
    prompt += "Here are several examples showing how to identify the most plausible cause(s):\n\n"

    for i, (ex_q, ex_docs) in enumerate(zip(examples[:num_shots], example_docs_list[:num_shots])):
        prompt += f"## Example {i+1}\n\n"
        prompt += f"**Target Event:** {ex_q['target_event']}\n\n"
        prompt += f"**Retrieved Documents:**\n{ex_docs if ex_docs else 'None'}\n\n"
        prompt += f"**Candidate Causes:**\n{format_options(ex_q)}\n\n"
        prompt += f"**Correct Answer:** {ex_q['golden_answer']}\n\n"
        prompt += "---\n\n"

    prompt += f"## Now answer this question\n\n"
    prompt += f"**Target Event:** {target_q['target_event']}\n\n"
    prompt += f"**Retrieved Documents:**\n{target_docs if target_docs else 'No documents available.'}\n\n"
    prompt += f"**Candidate Causes:**\n{format_options(target_q)}\n\n"
    prompt += "Answer (only the letter(s), e.g., 'A', 'BC', 'ABD'):"
    return prompt


def build_cot_prompt(question: Dict, docs_text: str,
                     examples: Optional[List[Dict]] = None,
                     example_docs_list: Optional[List[str]] = None) -> Tuple[str, str]:
    system_prompt = """You are an expert in abductive reasoning and causal inference.

When analyzing a cause-effect scenario, follow this reasoning process:
1. Understand the target event
2. Extract key information from documents
3. Evaluate each candidate cause
4. Compare and select the most plausible cause(s)
5. Output final answer as "Answer: X"

Be thorough. Multiple causes can be correct if supported by evidence."""

    user_prompt = ""
    if examples and example_docs_list:
        user_prompt += "# Examples with reasoning\n\n"
        for i, (ex_q, ex_docs) in enumerate(zip(examples[:2], example_docs_list[:2])):
            user_prompt += f"## Example {i+1}\n\n"
            user_prompt += f"Event: {ex_q['target_event']}\n\n"
            user_prompt += f"Docs:\n{ex_docs}\n\n"
            user_prompt += f"Candidates:\n{format_options(ex_q)}\n\n"
            user_prompt += f"Answer: {ex_q['golden_answer']}\n\n---\n\n"

    user_prompt += f"## Target Event\n{question['target_event']}\n\n"
    user_prompt += f"## Documents\n{docs_text}\n\n"
    user_prompt += f"## Candidates\n{format_options(question)}\n\n"
    user_prompt += "## Reasoning:\n"
    return system_prompt, user_prompt


# ================= LLM Inference =================

async def get_llm_prediction(client: AsyncAzureOpenAI, prompt, deployment_name: str,
                             is_cot: bool = False, temperature: float = 0.1,
                             max_tokens: int = 4000) -> Tuple[str, Optional[str]]:
    try:
        messages = []
        if is_cot:
            system_prompt, user_prompt = prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an expert at abductive reasoning."},
                {"role": "user", "content": prompt}
            ]

        response = await client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        full_response = response.choices[0].message.content.strip()

        if is_cot:
            answer_match = re.search(r'Answer[:\s]*([A-D,\s]+)', full_response, re.I)
            if answer_match:
                ans = answer_match.group(1)
            else:
                ans = full_response
            answer_set = normalize_answer(ans)
            answer_str = ",".join(sorted(answer_set))
            return answer_str, full_response
        else:
            answer_set = normalize_answer(full_response)
            answer_str = ",".join(sorted(answer_set))
            return answer_str, None

    except Exception as e:
        print(f"LLM API error: {e}")
        return "", None


# ================= Evaluation =================

def evaluate_predictions(predictions: List[str], golden_answers: List[str]) -> Tuple[float, List[Dict]]:
    total_score = 0.0
    details = []
    for pred, gold in zip(predictions, golden_answers):
        pred_set = normalize_answer(pred)
        gold_set = normalize_answer(gold)
        if not pred_set:
            score = 0.0
            status = "empty"
        elif pred_set == gold_set:
            score = 1.0
            status = "full_match"
        elif pred_set.issubset(gold_set):
            score = 0.5
            status = "partial_match"
        else:
            score = 0.0
            status = "incorrect"
        total_score += score
        details.append({"pred": pred, "gold": gold, "score": score, "status": status})
    return total_score / len(predictions) if predictions else 0.0, details


# ================= Main Async =================

async def async_main(args):
    print("Loading configuration...")
    config = load_config(args.config)
    client, deployment_name = initialize_llm(config)
    print("LLM client initialized successfully.")

    print("Loading dataset...")
    questions_path = os.path.join(args.data_dir, args.split, "questions.jsonl")
    docs_path = os.path.join(args.data_dir, args.split, "docs.json")
    questions = load_jsonl(questions_path)
    docs_map = load_docs_json(docs_path)

    print(f"Loaded {len(questions)} questions.")
    print(f"Context mode: {args.context}")  # 显示上下文模式

    all_docs_texts = []
    for q in questions:
        doc_item = docs_map.get(q['topic_id'], {})
        # ==================== 👇 传入 question & context mode ====================
        docs_text = extract_document_text(doc_item, q, context_mode=args.context)
        all_docs_texts.append(docs_text)

    examples = []
    example_docs_texts = []
    if args.method in ["oneshot", "fewshot", "cot_fewshot"] and len(questions) > args.num_shots:
        for i in range(min(args.num_shots, len(questions)-1)):
            examples.append(questions[i])
            example_docs_texts.append(all_docs_texts[i])
        if args.method == "oneshot":
            target_questions = questions[1:]
            target_docs_texts = all_docs_texts[1:]
        else:
            start = len(examples)
            target_questions = questions[start:]
            target_docs_texts = all_docs_texts[start:]
    else:
        target_questions = questions
        target_docs_texts = all_docs_texts

    predictions = []
    golden_answers = []
    reasoning_traces = [] if args.method.startswith("cot") else None
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_one(q, docs_text, idx):
        async with semaphore:
            if args.method == "zeroshot":
                prompt = build_zero_shot_prompt(q, docs_text)
                pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, False, args.temperature, args.max_tokens)
            elif args.method == "oneshot":
                prompt = build_one_shot_prompt(examples[0], example_docs_texts[0], q, docs_text)
                pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, False, args.temperature, args.max_tokens)
            elif args.method == "fewshot":
                prompt = build_few_shot_prompt(examples, example_docs_texts, q, docs_text, args.num_shots)
                pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, False, args.temperature, args.max_tokens)
            elif args.method == "cot_zeroshot":
                prompt = build_cot_prompt(q, docs_text, None)
                pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, True, args.temperature, args.max_tokens)
            elif args.method == "cot_fewshot":
                prompt = build_cot_prompt(q, docs_text, examples, example_docs_texts)
                pred, reasoning = await get_llm_prediction(client, prompt, deployment_name, True, args.temperature, args.max_tokens)
            else:
                raise ValueError(f"Unknown method: {args.method}")
            return idx, pred, reasoning

    tasks = [process_one(q, dt, i) for i, (q, dt) in enumerate(zip(target_questions, target_docs_texts))]
    results = [None]*len(tasks)
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Predicting"):
        i, p, r = await f
        results[i] = (p, r)

    for pred, reasoning in results:
        predictions.append(pred)
        golden_answers.append(target_questions[len(predictions)-1]['golden_answer'])
        if reasoning_traces is not None and reasoning:
            reasoning_traces.append(reasoning)

    avg_score, details = evaluate_predictions(predictions, golden_answers)
    full = sum(1 for d in details if d['status']=='full_match')
    part = sum(1 for d in details if d['status']=='partial_match')
    wrong = len(details) - full - part

    print(f"\n=== RESULTS ===")
    print(f"Method: {args.method}")
    print(f"Context: {args.context}")  # 输出上下文模式
    print(f"Score: {avg_score:.4f}")
    print(f"Full: {full} | Partial: {part} | Wrong: {wrong}")

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "method": args.method,
                "context": args.context,
                "score": avg_score,
                "full_matches": full,
                "partial_matches": part,
                "predictions": predictions,
                "golden": golden_answers
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="AER with Context Filtering")
    parser.add_argument("--data_dir", default="./semeval2026-task12-dataset")
    parser.add_argument("--split", default="dev_data", choices=["sample_data","train_data","dev_data"])
    parser.add_argument("--method", default="zeroshot", choices=["zeroshot","oneshot","fewshot","cot_zeroshot","cot_fewshot"])
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--max_concurrent", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--config", default="config.txt")
    parser.add_argument("--output", default=None)
    # ==================== 👇 新增：上下文参数 ====================
    parser.add_argument("--context", default="full", choices=["full","filtered"], help="full = all docs | filtered = only most relevant (proposal)")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()