"""
SemEval2026 Task12: Abductive Event Reasoning (AER)
NO TRUNCATION VERSION - uses full document content everywhere
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
    """Load Azure OpenAI configuration from txt file"""
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
    """Initialize Azure OpenAI client"""
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
    """Load .jsonl file (one JSON object per line)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_docs_json(file_path: str) -> Dict[int, Dict[str, Any]]:
    """Load docs.json file, return mapping from topic_id to full document data"""
    docs_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            topic_id = item['topic_id']
            docs_map[topic_id] = item
    return docs_map


def extract_document_text(doc_item: Dict[str, Any]) -> str:
    """
    Extract ALL document text WITHOUT ANY truncation.
    Uses full content for every document.
    """
    docs_list = doc_item.get('docs', [])
    if not docs_list:
        return "No documents available."

    formatted_docs = []
    for i, doc in enumerate(docs_list):
        title = doc.get('title', 'Untitled')
        # Always use full content, never truncate
        content = doc.get('content', '')
        if not content:
            content = doc.get('snippet', '')

        formatted_doc = f"\n{'='*60}\n[Document {i+1}] {title}\n{'='*60}\n{content}\n"
        formatted_docs.append(formatted_doc)

    total_chars = sum(len(d) for d in formatted_docs)
    doc_count = len(docs_list)

    print(f"  Extracted {doc_count} documents, total {total_chars} characters")

    return "\n".join(formatted_docs)


def normalize_answer(answer: str) -> Set[str]:
    """Normalize answer string to a set of uppercase letters (A-D)"""
    if not answer:
        return set()
    letters = re.findall(r'[A-D]', answer.upper())
    return set(letters)


def format_options(question: Dict[str, Any]) -> str:
    """Format the four options A, B, C, D"""
    return f"""A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}"""


# ================= Prompt Building (NO TRUNCATION ANYWHERE) =================

def build_zero_shot_prompt(question: Dict, docs_text: str) -> str:
    """Zero-shot prompt - uses full documents"""
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
    """One-shot prompt - uses FULL documents for both example and target (NO truncation)"""
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
    """Few-shot prompt - uses FULL documents for all examples (NO truncation)"""
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
    """
    Chain-of-Thought prompt - uses FULL documents for everything (NO truncation)
    """

    system_prompt = """You are an expert in abductive reasoning and causal inference.

When analyzing a cause-effect scenario, follow this reasoning process:

1. **Understand the target event** - What exactly happened? What are its key characteristics?

2. **Extract key information from documents** - What facts, background, or contextual clues are provided?

3. **Evaluate each candidate cause**:
   - Does the candidate have direct evidence in the documents?
   - Is there a plausible causal mechanism linking this cause to the event?
   - Would this cause likely produce the observed outcome?

4. **Compare and select** - Which cause(s) are most plausible? Consider both directness and strength of evidence.

5. **Output your final answer** in format: "Answer: X" where X is letter(s) like A, BC, ABD

Be thorough in your reasoning. Multiple causes can be correct if they all plausibly contributed."""

    user_prompt = ""

    if examples and example_docs_list:
        user_prompt += "# Examples with reasoning traces\n\n"
        for i, (ex_q, ex_docs) in enumerate(zip(examples[:2], example_docs_list[:2])):
            # NO TRUNCATION - use full document content
            user_prompt += f"## Example {i+1}\n\n"
            user_prompt += f"**Target Event:** {ex_q['target_event']}\n\n"
            user_prompt += f"**Documents:**\n{ex_docs if ex_docs else 'None'}\n\n"
            user_prompt += f"**Candidates:**\n{format_options(ex_q)}\n\n"
            user_prompt += f"**Correct Answer:** {ex_q['golden_answer']}\n\n"
            user_prompt += "---\n\n"

        user_prompt += "# Now analyze the following question\n\n"

    user_prompt += f"## Target Event\n{question['target_event']}\n\n"
    user_prompt += f"## Retrieved Documents\n{docs_text if docs_text else 'No documents available.'}\n\n"
    user_prompt += f"## Candidate Causes\n{format_options(question)}\n\n"
    user_prompt += "## Your Reasoning\n"

    return system_prompt, user_prompt


# ================= LLM Inference =================

async def get_llm_prediction(client: AsyncAzureOpenAI, prompt, deployment_name: str,
                             is_cot: bool = False, temperature: float = 0.1,
                             max_tokens: int = 4000) -> Tuple[str, Optional[str]]:
    """
    Call LLM and return predicted answer string.
    Increased default max_tokens to 4000 for long documents.
    """
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
                {"role": "system",
                 "content": "You are an expert at abductive reasoning. Select the most plausible cause(s) for each given event."},
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
            answer_match = re.search(r'Answer[:\s]*([A-D](?:[,\s]*[A-D])*)', full_response, re.IGNORECASE)
            if answer_match:
                answer_set = normalize_answer(answer_match.group(1))
                answer_str = ",".join(sorted(answer_set))
            else:
                letters = re.findall(r'\b([A-D])\b', full_response.upper())
                if letters:
                    answer_str = ",".join(sorted(set(letters)))
                else:
                    answer_str = ""
            return answer_str, full_response
        else:
            answer_set = normalize_answer(full_response)
            if not answer_set:
                if full_response and full_response[0].upper() in 'ABCD':
                    answer_set = {full_response[0].upper()}
            answer_str = ",".join(sorted(answer_set))
            return answer_str, None

    except Exception as e:
        print(f"LLM API error: {e}")
        return "", None


# ================= Evaluation =================

def evaluate_predictions(predictions: List[str], golden_answers: List[str]) -> Tuple[float, List[Dict]]:
    """Evaluate predictions according to official metrics"""
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
        details.append({
            "pred": pred,
            "gold": gold,
            "score": score,
            "status": status,
            "pred_set": list(pred_set),
            "gold_set": list(gold_set)
        })

    return total_score / len(predictions) if predictions else 0.0, details


# ================= Main Async Function =================

async def async_main(args):
    print("Loading configuration...")
    config = load_config(args.config)
    client, deployment_name = initialize_llm(config)
    print("LLM client initialized successfully.")

    print("Loading dataset...")
    questions_path = os.path.join(args.data_dir, args.split, "questions.jsonl")
    docs_path = os.path.join(args.data_dir, args.split, "docs.json")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Docs file not found: {docs_path}")

    questions = load_jsonl(questions_path)
    docs_map = load_docs_json(docs_path)
    print(f"Loaded {len(questions)} questions.")
    print(f"Loaded docs for {len(docs_map)} topics.")

    # Prepare document texts for all questions (NO TRUNCATION)
    print("\nPreparing document contexts (using FULL content, NO truncation)...")
    all_docs_texts = []
    for q in questions:
        doc_item = docs_map.get(q['topic_id'], {})
        docs_text = extract_document_text(doc_item)
        all_docs_texts.append(docs_text)

    # Print token usage estimation
    total_chars = sum(len(t) for t in all_docs_texts)
    print(f"\nTotal document characters across all questions: {total_chars:,}")
    print(f"Estimated tokens (approx 4 chars/token): ~{total_chars // 4:,}")

    # Prepare examples for few-shot methods
    examples = []
    example_docs_texts = []

    if args.method in ["oneshot", "fewshot", "cot_fewshot"] and len(questions) > args.num_shots:
        for i in range(min(args.num_shots, len(questions) - 1)):
            examples.append(questions[i])
            example_docs_texts.append(all_docs_texts[i])

        print(f"\nPrepared {len(examples)} example(s).")

        if args.method == "oneshot":
            start_idx = 1
            target_questions = questions[1:]
            target_docs_texts = all_docs_texts[1:]
        else:
            start_idx = len(examples)
            target_questions = questions[start_idx:]
            target_docs_texts = all_docs_texts[start_idx:]
    else:
        start_idx = 0
        target_questions = questions
        target_docs_texts = all_docs_texts
        examples = []
        example_docs_texts = []

    print(f"\nUsing {args.method.upper()} method (NO TRUNCATION)")
    print(f"Samples to predict: {len(target_questions)}")

    predictions = []
    golden_answers = []
    reasoning_traces = [] if args.method.startswith("cot") else None

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_one(q, docs_text, idx):
        async with semaphore:
            if args.method == "zeroshot":
                prompt = build_zero_shot_prompt(q, docs_text)
                pred, reasoning = await get_llm_prediction(
                    client, prompt, deployment_name, is_cot=False,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )
            elif args.method == "oneshot":
                prompt = build_one_shot_prompt(
                    examples[0], example_docs_texts[0], q, docs_text
                )
                pred, reasoning = await get_llm_prediction(
                    client, prompt, deployment_name, is_cot=False,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )
            elif args.method == "fewshot":
                prompt = build_few_shot_prompt(
                    examples, example_docs_texts, q, docs_text, num_shots=args.num_shots
                )
                pred, reasoning = await get_llm_prediction(
                    client, prompt, deployment_name, is_cot=False,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )
            elif args.method == "cot_zeroshot":
                prompt = build_cot_prompt(q, docs_text, examples=None)
                pred, reasoning = await get_llm_prediction(
                    client, prompt, deployment_name, is_cot=True,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )
            elif args.method == "cot_fewshot":
                prompt = build_cot_prompt(q, docs_text, examples, example_docs_texts)
                pred, reasoning = await get_llm_prediction(
                    client, prompt, deployment_name, is_cot=True,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )
            else:
                raise ValueError(f"Unknown method: {args.method}")

            return idx, pred, reasoning

    tasks = []
    for idx, (q, docs_text) in enumerate(zip(target_questions, target_docs_texts)):
        tasks.append(process_one(q, docs_text, idx))

    results = [None] * len(target_questions)
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Predicting"):
        idx, pred, reasoning = await future
        results[idx] = (pred, reasoning)

    for i, (pred, reasoning) in enumerate(results):
        predictions.append(pred)
        golden_answers.append(target_questions[i]['golden_answer'])
        if reasoning_traces is not None and reasoning:
            reasoning_traces.append(reasoning)

        if (i + 1) % 20 == 0:
            current_score, _ = evaluate_predictions(predictions, golden_answers)
            print(f"\n  Progress: {i+1}/{len(target_questions)}, current score: {current_score:.4f}")

    avg_score, score_details = evaluate_predictions(predictions, golden_answers)

    print(f"\n{'='*60}")
    print(f"=== Evaluation Results ({args.method.upper()}) ===")
    print(f"Split: {args.split}")
    print(f"Samples: {len(predictions)}")
    if args.method in ["fewshot", "cot_fewshot"]:
        print(f"Few-shot examples: {args.num_shots}")
    print(f"Truncation: NO (full documents)")
    print(f"Average Score (official metric): {avg_score:.4f}")
    print(f"{'='*60}")

    full_matches = sum(1 for d in score_details if d['status'] == 'full_match')
    partial_matches = sum(1 for d in score_details if d['status'] == 'partial_match')
    incorrect = sum(1 for d in score_details if d['status'] == 'incorrect')
    empty = sum(1 for d in score_details if d['status'] == 'empty')

    print(f"\nDetailed Statistics:")
    print(f"  Full match (1.0): {full_matches} ({full_matches/len(predictions)*100:.1f}%)")
    print(f"  Partial match (0.5): {partial_matches} ({partial_matches/len(predictions)*100:.1f}%)")
    print(f"  Incorrect/Empty (0): {incorrect+empty} ({(incorrect+empty)/len(predictions)*100:.1f}%)")

    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(predictions))):
        icon = "✓" if score_details[i]['score'] == 1.0 else "◐" if score_details[i]['score'] == 0.5 else "✗"
        print(f"  {icon} Sample {i+1}: Pred={predictions[i]:<6} Gold={golden_answers[i]}")

    if args.method.startswith("cot") and reasoning_traces and len(reasoning_traces) > 0:
        print(f"\n=== Sample CoT Reasoning ===")
        trace = reasoning_traces[0]
        print(trace[:1500] + "..." if len(trace) > 1500 else trace)

    if args.output:
        output_file = args.output
        results_dict = {
            "method": args.method,
            "split": args.split,
            "num_samples": len(predictions),
            "num_shots": args.num_shots if args.method in ["fewshot", "cot_fewshot"] else (1 if args.method == "oneshot" else 0),
            "truncation": False,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "avg_score": avg_score,
            "full_matches": full_matches,
            "partial_matches": partial_matches,
            "incorrect": incorrect + empty,
            "predictions": predictions,
            "golden_answers": golden_answers,
            "score_details": score_details
        }
        if reasoning_traces:
            results_dict["reasoning_traces"] = reasoning_traces[:10]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SemEval2026 Task12 - AER (NO TRUNCATION - Full Documents)")
    parser.add_argument("--data_dir", type=str, default="./semeval2026-task12-dataset",
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="sample_data",
                        choices=["sample_data", "train_data", "dev_data"],
                        help="Dataset split to evaluate")
    parser.add_argument("--method", type=str, default="zeroshot",
                        choices=["zeroshot", "oneshot", "fewshot", "cot_zeroshot", "cot_fewshot"],
                        help="Reasoning method")
    parser.add_argument("--num_shots", type=int, default=3,
                        help="Number of examples for few-shot (default: 3)")
    parser.add_argument("--max_concurrent", type=int, default=2,
                        help="Maximum concurrent API requests (default: 2 - reduce if hitting rate limits)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--max_tokens", type=int, default=4000,
                        help="Max tokens for LLM response (default: 4000)")
    parser.add_argument("--config", type=str, default="config.txt",
                        help="Azure OpenAI config file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("RUNNING WITH NO TRUNCATION - Full document content will be used")
    print("This may cause high token usage and potential API timeouts")
    print("="*60 + "\n")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()