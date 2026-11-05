import json
import re
import sys
from typing import List, Dict
from collections import defaultdict
import statistics
import argparse
import regex
import string
from collections import Counter
from bert_score import score
from nltk.stem import PorterStemmer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import uuid
import numpy as np
import torch
import random
import os

# Add project root to path
script_dir = Path(__file__).parent
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from omnimembench.engine.evaluate.scoring import qa_score, compute_nugget_metrics, retrieval_judgment

# Define data directories
PROJECT_ROOT = project_root
DATA_DIR = PROJECT_ROOT / "data"
DIALOG_DIR = DATA_DIR / "dialog"
RESULT_DIR = PROJECT_ROOT / "result_debug"

# Runtime-overridable by --data_root / --tier
_DATA_ROOT: Path = None
_TIER: str = None
_RESULT_SCAN_DIR: Path = None  # When set, get_available_datasets() scans this dir


def get_available_datasets():
    """
    Discover datasets to evaluate.
    Priority: 1) scan result dir (_RESULT_SCAN_DIR) for subdirs with *_results.json
              2) scan source data dir (_DATA_ROOT) for subdirs with {tier}.json
              3) legacy: scan DIALOG_DIR for .json files
    """
    # 1) Scan result directory — evaluate everything that has results
    if _RESULT_SCAN_DIR and _RESULT_SCAN_DIR.is_dir():
        datasets = []
        for entry in sorted(_RESULT_SCAN_DIR.iterdir()):
            if entry.is_dir() and list(entry.glob("*_results.json")):
                datasets.append(entry.name)
        if datasets:
            return datasets

    # 2) Scan source data directory
    if _DATA_ROOT and _TIER:
        datasets = []
        if _DATA_ROOT.is_dir():
            for entry in sorted(_DATA_ROOT.iterdir()):
                if entry.is_dir() and (entry / f"{_TIER}.json").exists():
                    datasets.append(entry.name)
        return datasets

    # 3) Legacy layout
    datasets = []
    if DIALOG_DIR.exists():
        for filename in DIALOG_DIR.iterdir():
            if filename.is_file() and filename.name.endswith('.json'):
                if '_results_' in filename.name or '_evaluate_result_' in filename.name or filename.name == 'rename.py':
                    continue
                datasets.append(filename.name.replace('.json', ''))
    return sorted(datasets)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

ps = PorterStemmer()


def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Fix normalization logic
def normalize_answer_robust(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        # Replace punctuation with spaces to prevent word concatenation
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answer_universal(s):
    """
    Universal answer normalization function
    - Protect decimal points (3.14)
    - Protect underscores (IMG_001, D1:IMG_001)
    - Replace other punctuation with spaces
    - Remove articles (a, an, the, and)
    """
    # 1. Basic preprocessing: convert to lowercase
    s = str(s).lower()

    # 2. Use fixed placeholders (containing no punctuation to avoid corruption during subsequent punctuation processing)
    # CRITICAL FIX: Previously uuid.uuid4() generated different placeholders each time, making identical answers unmatchable
    dot_placeholder = 'DOTPLACEHOLDER'
    us_placeholder = 'UNDERSCOREPLACEHOLDER'

    # 3. Protect decimal points: match dots surrounded by digits
    s = regex.sub(r'(?<=\d)\.(?=\d)', dot_placeholder, s)

    # 4. Protect underscores: underscores are core features of IDs (e.g., IMG_001)
    s = s.replace('_', us_placeholder)

    # 5. Remove articles (a, an, the, and) - consistent with existing logic
    s = regex.sub(r'\b(a|an|the|and)\b', ' ', s)

    # 6. Handle punctuation: replace all other punctuation with spaces
    exclude = set(string.punctuation)
    s = ''.join(ch if ch not in exclude else ' ' for ch in s)

    # 7. Restore placeholders
    s = s.replace(dot_placeholder, '.')
    s = s.replace(us_placeholder, '_')

    # 8. Fix extra spaces
    s = ' '.join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str):
    """
    Calculate exact match score.

    Function: Compare whether normalized prediction and ground truth answers match exactly.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer

    Returns:
        bool: True if normalized strings match exactly, False otherwise
    """
    return normalize_answer_universal(prediction) == normalize_answer_universal(ground_truth)


def f1_score(prediction: str, ground_truth: str):
    """
    Calculate F1 score.

    Function: Calculate F1 score based on word overlap after stemming.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer

    Returns:
        float: F1 score in range [0,1], higher values indicate greater overlap
    """
    norm_p = normalize_answer_universal(prediction)
    norm_g = normalize_answer_universal(ground_truth)
    
    prediction_tokens = [ps.stem(w) for w in norm_p.split()]
    ground_truth_tokens = [ps.stem(w) for w in norm_g.split()]
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    
    return (2 * precision * recall) / (precision + recall)


def bleu_score(prediction: str, ground_truth: str, weights: List[float] = (0.25, 0.25, 0.25, 0.25)):
    """
    Calculate BLEU score.

    Function: Evaluate similarity between predicted and ground truth answers using BLEU metric.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer
    """
    norm_p = normalize_answer_universal(prediction)
    norm_g = normalize_answer_universal(ground_truth)
    
    pred_tokens = norm_p.split()
    ref_tokens = norm_g.split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    chencherry = SmoothingFunction()
    return sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=chencherry.method1)


def bert_score(prediction: str, ground_truth: str):
    """
    Calculate BERT score.

    Function: Use BERT model to compute semantic similarity between predicted and ground truth answers.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer

    Returns:
        float: BERT F1 score in range [0,1], higher values indicate greater semantic similarity
    """
    prediction = normalize_answer_universal(prediction)
    ground_truth = normalize_answer_universal(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    return max(0, F1[0].item())


def calculate_retrieval_metrics(retrieved_ids: List[str], clue_ids: List[str], k: int = 10) -> Dict[str, float]:
    """
    Calculate retrieval metrics: Precision@K, Recall@K, HitRate@K

    Args:
        retrieved_ids: List of retrieved IDs (strings)
        clue_ids: List of ground truth clue IDs (strings)
        k: Top-K cutoff for evaluation

    Returns:
        dict: Dictionary containing precision_k, recall_k, hitrate_k
    """
    # Normalize to strings and cut to K
    retrieved_ids = [str(x) for x in retrieved_ids][:k]
    # Deduplicate preserving order
    seen = set()
    retrieved_ids = [x for x in retrieved_ids if not (x in seen or seen.add(x))]

    # Normalize ground truth to strings
    if isinstance(clue_ids, list):
        gt_set = set([str(x) for x in clue_ids])
    else:
        gt_set = set()

    R = retrieved_ids
    G = gt_set

    # If both are empty, return zero metrics
    if len(R) == 0 and len(G) == 0:
        return {
            'precision_k': 0.0,
            'recall_k': 0.0,
            'hitrate_k': 0.0
        }

    # Calculate hit count
    hit_rels = sum(1 for x in R if x in G)

    # Precision@K: use actual returned count as denominator
    denom = max(1, len(R))
    prec_k = hit_rels / denom

    # Recall@K: use ground truth count as denominator
    rec_k = (hit_rels / len(G)) if len(G) > 0 else 0.0

    # HitRate@K: binary indicator (1 if any hit, 0 otherwise)
    hitrate_k = 1.0 if hit_rels > 0 else 0.0

    return {
        'precision_k': prec_k,
        'recall_k': rec_k,
        'hitrate_k': hitrate_k
    }


def load_judge_prompt(prompt_path: str) -> str:
    """
    Load the LLM judge prompt template from file.
    
    Args:
        prompt_path: Path to the prompt template file (relative to evaluate directory)
        
    Returns:
        str: The prompt template content
    """
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    full_path = script_dir / prompt_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Prompt template not found at: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_judge_response(response_text: str) -> Dict[str, any]:
    """Parse a JSON judgment using the 0/0.5/1 rubric."""
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        result = json.loads(match.group() if match else response_text)
        return {'score': qa_score(result), 'reasoning': result.get('reasoning', '')}
    except (ValueError, TypeError):
        return None


def llm_judge_score(
    question: str, ground_truth: str, model_output: str,
    client: OpenAI, model_name: str, prompt_template: str,
    max_retries: int = 3, timeout: int = 60, delay_base: float = 1.0,
) -> Dict[str, any]:
    """Return a valid QA judgment within the configured attempt limit."""
    prompt = (prompt_template.replace('{{question}}', question)
              .replace('{{ground_truth}}', ground_truth)
              .replace('{{model_output}}', model_output))
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0, timeout=timeout,
            )
            result = parse_judge_response(response.choices[0].message.content)
            if result is None:
                raise ValueError('Judge output must contain a QA score of 0, 0.5, or 1')
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay_base * (2 ** attempt))
    raise RuntimeError(f'LLM judge failed after {max_retries} attempts: {last_error}')


def load_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def aggregate_overall_metrics(individual_results: List[Dict], eval_retrieval_metrics: bool = False) -> Dict:
    """
    Aggregate metrics across multiple datasets to compute overall statistics.
    
    Args:
        individual_results: List of result dictionaries from individual dataset evaluations
        eval_retrieval_metrics: Whether retrieval metrics were evaluated
        
    Returns:
        dict: Aggregated metrics with overall statistics
    """
    # Initialize aggregators
    all_f1 = []
    all_bleu = []
    all_bleu_1 = []
    all_bleu_2 = []
    all_em = []
    all_llm_judge = []
    all_precision_k = []
    all_recall_k = []
    all_hitrate_k = []
    
    # Aggregate category-wise metrics
    category_f1 = defaultdict(list)
    category_bleu = defaultdict(list)
    category_bleu_1 = defaultdict(list)
    category_bleu_2 = defaultdict(list)
    category_em = defaultdict(list)
    category_llm_judge = defaultdict(list)
    category_precision_k = defaultdict(list)
    category_recall_k = defaultdict(list)
    category_hitrate_k = defaultdict(list)
    
    # Collect all metrics from individual results
    for result in individual_results:
        # Overall metrics
        if 'overall_f1' in result:
            all_f1.extend(result['overall_f1'])
        if 'overall_bleu' in result:
            all_bleu.extend(result['overall_bleu'])
        if 'overall_bleu_1' in result:
            all_bleu_1.extend(result['overall_bleu_1'])
        if 'overall_bleu_2' in result:
            all_bleu_2.extend(result['overall_bleu_2'])
        if 'overall_em' in result:
            all_em.extend(result['overall_em'])
        if 'overall_llm_judge' in result:
            all_llm_judge.extend(result['overall_llm_judge'])
        if eval_retrieval_metrics:
            if 'overall_precision_k' in result:
                all_precision_k.extend(result['overall_precision_k'])
            if 'overall_recall_k' in result:
                all_recall_k.extend(result['overall_recall_k'])
            if 'overall_hitrate_k' in result:
                all_hitrate_k.extend(result['overall_hitrate_k'])
        
        # Category-wise metrics
        for category, scores in result.get('category_f1', {}).items():
            category_f1[category].extend(scores)
        for category, scores in result.get('category_bleu', {}).items():
            category_bleu[category].extend(scores)
        for category, scores in result.get('category_bleu_1', {}).items():
            category_bleu_1[category].extend(scores)
        for category, scores in result.get('category_bleu_2', {}).items():
            category_bleu_2[category].extend(scores)
        for category, scores in result.get('category_em', {}).items():
            category_em[category].extend(scores)
        if 'category_llm_judge' in result:
            for category, scores in result['category_llm_judge'].items():
                category_llm_judge[category].extend(scores)
        if eval_retrieval_metrics:
            for category, scores in result.get('category_precision_k', {}).items():
                category_precision_k[category].extend(scores)
            for category, scores in result.get('category_recall_k', {}).items():
                category_recall_k[category].extend(scores)
            for category, scores in result.get('category_hitrate_k', {}).items():
                category_hitrate_k[category].extend(scores)
    
    # Compute overall averages
    avg_overall_metrics = {}
    if all_f1:
        avg_overall_metrics['F1'] = statistics.mean(all_f1)
    if all_bleu:
        avg_overall_metrics['BLEU'] = statistics.mean(all_bleu)
    if all_bleu_1:
        avg_overall_metrics['BLEU_1'] = statistics.mean(all_bleu_1)
    if all_bleu_2:
        avg_overall_metrics['BLEU_2'] = statistics.mean(all_bleu_2)
    if all_em:
        avg_overall_metrics['EM'] = statistics.mean(all_em)
    if all_llm_judge:
        avg_overall_metrics['LLM_JUDGE'] = statistics.mean(all_llm_judge)
    if eval_retrieval_metrics:
        if all_precision_k:
            avg_overall_metrics['Precision@K'] = statistics.mean(all_precision_k)
        if all_recall_k:
            avg_overall_metrics['Recall@K'] = statistics.mean(all_recall_k)
        if all_hitrate_k:
            avg_overall_metrics['HitRate@K'] = statistics.mean(all_hitrate_k)
    
    # Compute category-wise averages
    avg_category_metrics = defaultdict(dict)
    for category, scores in category_f1.items():
        if scores:
            avg_category_metrics[category]['F1'] = statistics.mean(scores)
    for category, scores in category_bleu.items():
        if scores:
            avg_category_metrics[category]['BLEU'] = statistics.mean(scores)
    for category, scores in category_bleu_1.items():
        if scores:
            avg_category_metrics[category]['BLEU_1'] = statistics.mean(scores)
    for category, scores in category_bleu_2.items():
        if scores:
            avg_category_metrics[category]['BLEU_2'] = statistics.mean(scores)
    for category, scores in category_em.items():
        if scores:
            avg_category_metrics[category]['EM'] = statistics.mean(scores)
    for category, scores in category_llm_judge.items():
        if scores:
            avg_category_metrics[category]['LLM_JUDGE'] = statistics.mean(scores)
    if eval_retrieval_metrics:
        for category, scores in category_precision_k.items():
            if scores:
                avg_category_metrics[category]['Precision@K'] = statistics.mean(scores)
        for category, scores in category_recall_k.items():
            if scores:
                avg_category_metrics[category]['Recall@K'] = statistics.mean(scores)
        for category, scores in category_hitrate_k.items():
            if scores:
                avg_category_metrics[category]['HitRate@K'] = statistics.mean(scores)
    
    for field, metric in [('clue_coverage', 'Clue_Coverage'), ('entry_relevance', 'Entry_Relevance')]:
        scores = [score for result in individual_results for score in result.get(f'overall_{field}', [])]
        if scores:
            avg_overall_metrics[metric] = statistics.mean(scores)
        categories = defaultdict(list)
        for result in individual_results:
            for category, values in result.get(f'category_{field}', {}).items():
                categories[category].extend(values)
        for category, values in categories.items():
            if values:
                avg_category_metrics[category][metric] = statistics.mean(values)
    counts = {}
    for field in ['llm_judge_counts', 'retrieval_judge_counts']:
        available = [result[field] for result in individual_results if field in result]
        if available:
            counts[field] = {key: sum(item[key] for item in available) for key in ['valid', 'failed']}
    return {
        'avg_overall_metrics_final': avg_overall_metrics,
        'avg_category_metrics_final': dict(avg_category_metrics),
        **counts,
    }

def _nugget_judge_one(
    question: str, model_output: str, nuggets: List[Dict],
    client: OpenAI, model_name: str, prompt_template: str,
    max_retries: int = 5, timeout: int = 60,
) -> Dict[str, any]:
    """Evaluate a SUM answer using complete reference-nugget coverage."""
    nuggets_list = "\n".join(f"  {n['nugget_id']}: {n.get('content', '')}" for n in nuggets)
    prompt = (prompt_template.replace('{{question}}', question)
              .replace('{{model_output}}', model_output)
              .replace('{{nuggets_list}}', nuggets_list))
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0, timeout=timeout,
            )
            text = response.choices[0].message.content or ''
            text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
            text = re.sub(r'\n?```$', '', text)
            metrics = compute_nugget_metrics(json.loads(text), nuggets)
            return {
                'score': metrics['nugget_recall'],
                'reasoning': f"nugget coverage: {metrics['nugget_covered']}/{metrics['nugget_total']}",
                'is_nugget': True, **metrics,
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'SUM judge failed after {max_retries} attempts: {last_error}')


def main(file_path: str, enable_llm_judge: bool = False, judge_config: Dict = None, eval_retrieval_metrics: bool = False, eval_topk: int = 5, num_workers: int = 1, enable_lexical_metrics: bool = False, enable_retrieval_judge: bool = False):
    """Main function to calculate average F1 scores per category."""
    # Load data from file
    data = load_data(file_path)

    # Initialize category dictionary
    # Dataset uses string categories
    category_f1 = defaultdict(list)
    overall_f1 = []
    category_bleu = defaultdict(list)
    overall_bleu = []
    category_bleu_1 = defaultdict(list)
    overall_bleu_1 = []
    category_bleu_2 = defaultdict(list)
    overall_bleu_2 = []
    category_em = defaultdict(list)
    overall_em = []

    # LLM Judge aggregations (only if enabled)
    category_llm_judge = defaultdict(list)
    overall_llm_judge = []
    llm_judge_details = []  # Store detailed results for each sample

    # Retrieval Judge aggregations (only if enabled)
    category_clue_coverage = defaultdict(list)
    overall_clue_coverage = []
    category_entry_relevance = defaultdict(list)
    overall_entry_relevance = []
    retrieval_judge_details = []

    # Retrieval metrics aggregations (only if enabled)
    category_precision_k = defaultdict(list)
    overall_precision_k = []
    category_recall_k = defaultdict(list)
    overall_recall_k = []
    category_hitrate_k = defaultdict(list)
    overall_hitrate_k = []

    avg_category_metrics_final = defaultdict(list)
    avg_overall_metrics_final = defaultdict(list)

    # Initialize LLM / Retrieval judge clients if enabled
    judge_client = None
    judge_prompt_template = None
    judge_model_name = None
    retrieval_judge_prompt_template = None
    if enable_llm_judge or enable_retrieval_judge:
        if judge_config is None:
            raise ValueError("judge_config must be provided when enable_llm_judge or enable_retrieval_judge is True")
        judge_client = OpenAI(
            api_key=judge_config.get('api_key'),
            base_url=judge_config.get('base_url')
        )
        judge_model_name = judge_config.get('name') or 'glm-5.2'
    if enable_retrieval_judge:
        retrieval_judge_prompt_template = load_judge_prompt('llm_judge_retrieval.txt')
        print(f"\nRetrieval Judge enabled using model: {judge_model_name}")
    nugget_judge_prompt_template = None
    if enable_llm_judge:
        # Load prompt template
        prompt_path = judge_config.get('prompt_path', 'llm_judge.txt')
        judge_prompt_template = load_judge_prompt(prompt_path)
        # Load nugget judge template for SUM category
        try:
            nugget_judge_prompt_template = load_judge_prompt('llm_judge_nugget.txt')
        except FileNotFoundError:
            nugget_judge_prompt_template = None
        print(f"\nLLM Judge enabled using model: {judge_model_name}")
        print("Loading prompt template from:", prompt_path)
        if nugget_judge_prompt_template:
            print("Nugget judge template loaded for SUM category")

    # Calculate scores for each sample (CPU metrics)
    valid_samples = []
    for sample in tqdm(data, total=len(data)):
        category = sample['category']
        # Skip if category is empty or None
        if not category:
            continue

        # Skip error-status QAs from run_bench (API failures)
        if sample.get('status') == 'error':
            continue

        system_answer = str(sample['system_answer'])
        # For IS (Image Search) category, handle multiple answers separated by commas
        # if category == 'IS':
        #     system_answer = system_answer.split(',')[0].strip()

        original_answer = str(sample['original_answer'])
        question = str(sample.get('question', ''))

        if enable_lexical_metrics:
            f1_metric = f1_score(system_answer, original_answer)
            bleu_metric = bleu_score(system_answer, original_answer)
            bleu_metric_1 = bleu_score(system_answer, original_answer, weights=(1, 0, 0, 0))
            bleu_metric_2 = bleu_score(system_answer, original_answer, weights=(0.5, 0.5, 0, 0))
            em_metric = exact_match_score(system_answer, original_answer)

            # Append metrics to the corresponding category
            category_f1[category].append(f1_metric)
            overall_f1.append(f1_metric)
            category_bleu[category].append(bleu_metric)
            overall_bleu.append(bleu_metric)
            category_bleu_1[category].append(bleu_metric_1)
            overall_bleu_1.append(bleu_metric_1)
            category_bleu_2[category].append(bleu_metric_2)
            overall_bleu_2.append(bleu_metric_2)
            category_em[category].append(em_metric)
            overall_em.append(em_metric)

        # Evaluate retrieval metrics (if enabled and data available)
        if eval_retrieval_metrics and sample.get("has_retrieval", True):
            retrieved_ids = sample.get("retrieved_ids", [])
            clue_ids = sample.get("clue", [])

            if retrieved_ids is not None and clue_ids is not None:
                retrieval_metrics = calculate_retrieval_metrics(retrieved_ids, clue_ids, k=eval_topk)

                category_precision_k[category].append(retrieval_metrics['precision_k'])
                overall_precision_k.append(retrieval_metrics['precision_k'])
                category_recall_k[category].append(retrieval_metrics['recall_k'])
                overall_recall_k.append(retrieval_metrics['recall_k'])
                category_hitrate_k[category].append(retrieval_metrics['hitrate_k'])
                overall_hitrate_k.append(retrieval_metrics['hitrate_k'])

        valid_samples.append(sample)

    # Evaluate with LLM Judge (parallel, with resume)
    if enable_llm_judge:
        # ── Resume: load existing judge results ──────────────────────────────
        existing_judge = {}  # question -> {score, reasoning}
        try:
            eval_output_path = Path(file_path).parent / (Path(file_path).stem.replace('_results', '_evaluate_result') + '.json')
            if eval_output_path.exists():
                with open(eval_output_path, encoding='utf-8') as stream:
                    prev = json.load(stream)
                if prev.get('judge_model') != judge_model_name:
                    prev = {}  # Re-score caches from another or an unspecified judge.
                for detail in prev.get('llm_judge_details', []):
                    q = detail.get('question', '')
                    if q and detail.get('score') is not None:
                        existing_judge[q] = detail
                if existing_judge:
                    print(f"Judge resume: loaded {len(existing_judge)} existing judge scores")
        except Exception:
            existing_judge = {}

        samples_to_judge = []
        for sample in valid_samples:
            q = str(sample.get('question', ''))
            if q in existing_judge:
                # Reuse existing judge result
                jr = existing_judge[q]
                try:
                    if str(sample['category']).upper() == 'SUM':
                        metrics = compute_nugget_metrics(jr.get('nugget_coverage'), sample.get('nuggets') or [])
                        jr = {**jr, **metrics, 'score': metrics['nugget_recall']}
                    else:
                        jr = {**jr, 'score': qa_score(jr)}
                except (ValueError, KeyError, TypeError):
                    samples_to_judge.append(sample)
                    continue
                category = sample['category']
                category_llm_judge[category].append(jr['score'])
                overall_llm_judge.append(jr['score'])
                llm_judge_details.append({
                    **jr,
                    'sample_id': sample.get('sample_id', ''),
                    'category': category,
                    'question': q,
                    'ground_truth': str(sample['original_answer']),
                    'model_output': str(sample['system_answer']),
                    'score': jr['score'],
                    'reasoning': jr.get('reasoning', '')
                })
            else:
                samples_to_judge.append(sample)

        def _judge_one(sample):
            question = str(sample.get('question', ''))
            try:
                if str(sample.get('category', '')).upper() == 'SUM':
                    result = _nugget_judge_one(
                        question=question, model_output=str(sample['system_answer']),
                        nuggets=sample.get('nuggets') or [],
                        client=judge_client, model_name=judge_model_name,
                        prompt_template=nugget_judge_prompt_template,
                        max_retries=judge_config.get('max_retries', 5),
                        timeout=judge_config.get('timeout', 60),
                    )
                else:
                    result = llm_judge_score(
                        question=question, ground_truth=str(sample['original_answer']),
                        model_output=str(sample['system_answer']),
                        client=judge_client, model_name=judge_model_name,
                        prompt_template=judge_prompt_template,
                        max_retries=judge_config.get('max_retries', 5),
                        timeout=judge_config.get('timeout', 60),
                    )
                return sample, result
            except Exception as e:
                return sample, {'error': str(e)}

        if samples_to_judge:
            workers = max(1, num_workers)
            print(f"\nRunning LLM Judge on {len(samples_to_judge)} samples ({len(llm_judge_details)} cached) with {workers} workers...")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_judge_one, s): s for s in samples_to_judge}
                for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
                    sample, judge_result = future.result()
                    category = sample['category']
                    judge_score = judge_result.get('score')
                    if judge_score is not None:
                        category_llm_judge[category].append(judge_score)
                        overall_llm_judge.append(judge_score)
                    llm_judge_details.append({
                        **judge_result,
                        'sample_id': sample.get('sample_id', ''),
                        'category': category,
                        'question': str(sample.get('question', '')),
                        'ground_truth': str(sample['original_answer']),
                        'model_output': str(sample['system_answer']),
                        'score': judge_score,
                    })
        else:
            print(f"\nAll {len(llm_judge_details)} judge scores loaded from cache, skipping LLM calls")

    # ── Retrieval Judge (LLM semantic retrieval evaluation) ──────────────────
    if enable_retrieval_judge:
        # ── Resume: load existing retrieval judge results ─────────────────
        existing_ret_judge = {}  # question -> {clue_coverage, entry_relevance, reasoning}
        try:
            eval_output_path = Path(file_path).parent / (Path(file_path).stem.replace('_results', '_evaluate_result') + '.json')
            if eval_output_path.exists():
                with open(eval_output_path, encoding='utf-8') as stream:
                    prev = json.load(stream)
                if prev.get('judge_model') != judge_model_name:
                    prev = {}
                for detail in prev.get('retrieval_judge_details', []):
                    q = detail.get('question', '')
                    if q and detail.get('clue_coverage') is not None:
                        existing_ret_judge[q] = {
                            'clue_coverage': detail['clue_coverage'],
                            'entry_relevance': detail['entry_relevance'],
                            'reasoning': detail.get('reasoning', ''),
                        }
                if existing_ret_judge:
                    print(f"Retrieval Judge resume: loaded {len(existing_ret_judge)} existing results")
        except Exception:
            existing_ret_judge = {}

        # Collect samples eligible for retrieval judge: has retrieved_texts and clue_texts
        ret_judge_samples = []
        for sample in valid_samples:
            q = str(sample.get('question', ''))
            retrieved_texts = sample.get('retrieved_texts', [])
            clue_texts = sample.get('clue_texts', [])
            if not retrieved_texts or not clue_texts:
                continue
            if q in existing_ret_judge:
                try:
                    rj = retrieval_judgment(existing_ret_judge[q], len(clue_texts), len(retrieved_texts))
                except (ValueError, TypeError):
                    ret_judge_samples.append(sample)
                    continue
                category = sample['category']
                cc = rj['clue_coverage']
                er = rj['entry_relevance']
                cc_rate = sum(cc) / len(cc) if cc else 0.0
                er_rate = sum(er) / len(er) if er else 0.0
                category_clue_coverage[category].append(cc_rate)
                overall_clue_coverage.append(cc_rate)
                category_entry_relevance[category].append(er_rate)
                overall_entry_relevance.append(er_rate)
                retrieval_judge_details.append({
                    'sample_id': sample.get('sample_id', ''),
                    'category': category,
                    'question': q,
                    'clue_coverage': cc,
                    'entry_relevance': er,
                    'clue_coverage_rate': cc_rate,
                    'entry_relevance_rate': er_rate,
                    'reasoning': rj['reasoning'],
                })
            else:
                ret_judge_samples.append(sample)

        def _parse_retrieval_judge_response(response_text, clue_count, entry_count):
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            result = json.loads(match.group() if match else response_text)
            return retrieval_judgment(result, clue_count, entry_count)

        def _retrieval_judge_one(sample):
            question = str(sample.get('question', ''))
            retrieved_texts = sample.get('retrieved_texts', [])
            clue_texts = sample.get('clue_texts', [])
            clue_texts_str = '\n'.join(f'[{i+1}] {t}' for i, t in enumerate(clue_texts))
            retrieved_texts_str = '\n'.join(f'[{i+1}] {t}' for i, t in enumerate(retrieved_texts))
            prompt = (retrieval_judge_prompt_template.replace('{{question}}', question)
                      .replace('{{clue_texts}}', clue_texts_str)
                      .replace('{{retrieved_texts}}', retrieved_texts_str))
            max_retries = judge_config.get('max_retries', 5)
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = judge_client.chat.completions.create(
                        model=judge_model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0, timeout=judge_config.get('timeout', 60),
                    )
                    result = _parse_retrieval_judge_response(
                        response.choices[0].message.content, len(clue_texts), len(retrieved_texts))
                    return sample, result
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(min(2 ** attempt, 8))
            return sample, {'error': f'Retrieval judge failed after {max_retries} attempts: {last_error}'}

        if ret_judge_samples:
            workers = max(1, num_workers)
            print(f"\nRunning Retrieval Judge on {len(ret_judge_samples)} samples ({len(retrieval_judge_details)} cached) with {workers} workers...")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_retrieval_judge_one, s): s for s in ret_judge_samples}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Retrieval Judge"):
                    sample, rj_result = future.result()
                    category = sample['category']
                    if 'error' in rj_result:
                        retrieval_judge_details.append({
                            'sample_id': sample.get('sample_id', ''), 'category': category,
                            'question': str(sample.get('question', '')), **rj_result,
                        })
                        continue
                    cc = rj_result['clue_coverage']
                    er = rj_result['entry_relevance']
                    cc_rate = sum(cc) / len(cc) if cc else 0.0
                    er_rate = sum(er) / len(er) if er else 0.0
                    category_clue_coverage[category].append(cc_rate)
                    overall_clue_coverage.append(cc_rate)
                    category_entry_relevance[category].append(er_rate)
                    overall_entry_relevance.append(er_rate)
                    retrieval_judge_details.append({
                        'sample_id': sample.get('sample_id', ''),
                        'category': category,
                        'question': str(sample.get('question', '')),
                        'clue_coverage': cc,
                        'entry_relevance': er,
                        'clue_coverage_rate': cc_rate,
                        'entry_relevance_rate': er_rate,
                        'reasoning': rj_result['reasoning'],
                    })
        else:
            print(f"\nAll {len(retrieval_judge_details)} retrieval judge scores loaded from cache, skipping LLM calls")

    # Calculate and print average F1 scores for each category
    print("\n" + "="*60)
    print("Category-wise Results:")
    print("="*60)
    if enable_lexical_metrics:
        for category, f1_scores in sorted(category_f1.items()):
            avg_f1 = statistics.mean(f1_scores)
            avg_category_metrics_final[category].append({"F1": avg_f1})
            print(f"Category {category}: Average F1 Score = {avg_f1:.4f} (n={len(f1_scores)})")

        for category, bleu_scores in sorted(category_bleu.items()):
            avg_bleu = statistics.mean(bleu_scores)
            avg_category_metrics_final[category].append({"BLEU": avg_bleu})
            print(f"Category {category}: Average BLEU Score = {avg_bleu:.4f} (n={len(bleu_scores)})")

        for category, bleu_1_scores in sorted(category_bleu_1.items()):
            avg_bleu_1 = statistics.mean(bleu_1_scores)
            avg_category_metrics_final[category].append({"BLEU_1": avg_bleu_1})
            print(f"Category {category}: Average BLEU_1 Score = {avg_bleu_1:.4f} (n={len(bleu_1_scores)})")

        for category, bleu_2_scores in sorted(category_bleu_2.items()):
            avg_bleu_2 = statistics.mean(bleu_2_scores)
            avg_category_metrics_final[category].append({"BLEU_2": avg_bleu_2})
            print(f"Category {category}: Average BLEU_2 Score = {avg_bleu_2:.4f} (n={len(bleu_2_scores)})")

        for category, em_scores in sorted(category_em.items()):
            avg_em = statistics.mean(em_scores)
            avg_category_metrics_final[category].append({"EM": avg_em})
            print(f"Category {category}: Average EM Score = {avg_em:.4f} (n={len(em_scores)})")

    # Print LLM Judge results if enabled
    if enable_llm_judge:
        for category, judge_scores in sorted(category_llm_judge.items()):
            avg_judge = statistics.mean(judge_scores)
            avg_category_metrics_final[category].append({"LLM_JUDGE": avg_judge})
            print(f"Category {category}: Average LLM Judge Score = {avg_judge:.4f} (n={len(judge_scores)})")

    # Print Retrieval Judge results if enabled
    if enable_retrieval_judge:
        for category, cc_scores in sorted(category_clue_coverage.items()):
            avg_cc = statistics.mean(cc_scores)
            avg_category_metrics_final[category].append({"Clue_Coverage": avg_cc})
            print(f"Category {category}: Average Clue Coverage Rate = {avg_cc:.4f} (n={len(cc_scores)})")
        for category, er_scores in sorted(category_entry_relevance.items()):
            avg_er = statistics.mean(er_scores)
            avg_category_metrics_final[category].append({"Entry_Relevance": avg_er})
            print(f"Category {category}: Average Entry Relevance Rate = {avg_er:.4f} (n={len(er_scores)})")

    # Print retrieval metrics results if enabled
    if eval_retrieval_metrics:
        for category, prec_scores in sorted(category_precision_k.items()):
            avg_prec = statistics.mean(prec_scores)
            avg_category_metrics_final[category].append({"Precision@K": avg_prec})
            print(f"Category {category}: Average Precision@{eval_topk} = {avg_prec:.4f} (n={len(prec_scores)})")
        
        for category, rec_scores in sorted(category_recall_k.items()):
            avg_rec = statistics.mean(rec_scores)
            avg_category_metrics_final[category].append({"Recall@K": avg_rec})
            print(f"Category {category}: Average Recall@{eval_topk} = {avg_rec:.4f} (n={len(rec_scores)})")
        
        for category, hr_scores in sorted(category_hitrate_k.items()):
            avg_hr = statistics.mean(hr_scores)
            avg_category_metrics_final[category].append({"HitRate@K": avg_hr})
            print(f"Category {category}: Average HitRate@{eval_topk} = {avg_hr:.4f} (n={len(hr_scores)})")

    # Calculate and print average scores for overall
    print("\n" + "="*60)
    print("Overall Results:")
    print("="*60)
    if enable_lexical_metrics:
        overall_avg_f1 = statistics.mean(overall_f1) if overall_f1 else 0.0
        avg_overall_metrics_final["F1"] = overall_avg_f1
        print(f"Overall on all categories: Average F1 Score = {overall_avg_f1:.4f} (n={len(overall_f1)})")

        overall_avg_bleu = statistics.mean(overall_bleu) if overall_bleu else 0.0
        avg_overall_metrics_final["BLEU"] = overall_avg_bleu
        print(f"Overall on all categories: Average BLEU Score = {overall_avg_bleu:.4f} (n={len(overall_bleu)})")

        overall_avg_bleu_1 = statistics.mean(overall_bleu_1) if overall_bleu_1 else 0.0
        avg_overall_metrics_final["BLEU_1"] = overall_avg_bleu_1
        print(f"Overall on all categories: Average BLEU_1 Score = {overall_avg_bleu_1:.4f} (n={len(overall_bleu_1)})")

        overall_avg_bleu_2 = statistics.mean(overall_bleu_2) if overall_bleu_2 else 0.0
        avg_overall_metrics_final["BLEU_2"] = overall_avg_bleu_2
        print(f"Overall on all categories: Average BLEU_2 Score = {overall_avg_bleu_2:.4f} (n={len(overall_bleu_2)})")

        overall_avg_em = statistics.mean(overall_em) if overall_em else 0.0
        avg_overall_metrics_final["EM"] = overall_avg_em
        print(f"Overall on all categories: Average EM Score = {overall_avg_em:.4f} (n={len(overall_em)})")

    # Print overall LLM Judge result if enabled
    if enable_llm_judge:
        overall_avg_judge = statistics.mean(overall_llm_judge) if overall_llm_judge else None
        avg_overall_metrics_final["LLM_JUDGE"] = overall_avg_judge
        score_text = f'{overall_avg_judge:.4f}' if overall_avg_judge is not None else 'NA'
        failed = sum('error' in item for item in llm_judge_details)
        print(f"Overall LLM Judge Score = {score_text} (valid={len(overall_llm_judge)}, failed={failed})")

    # Print overall Retrieval Judge result if enabled
    if enable_retrieval_judge:
        if overall_clue_coverage:
            overall_avg_cc = statistics.mean(overall_clue_coverage)
            avg_overall_metrics_final["Clue_Coverage"] = overall_avg_cc
            print(f"Overall on all categories: Average Clue Coverage Rate = {overall_avg_cc:.4f} (n={len(overall_clue_coverage)})")
        if overall_entry_relevance:
            overall_avg_er = statistics.mean(overall_entry_relevance)
            avg_overall_metrics_final["Entry_Relevance"] = overall_avg_er
            print(f"Overall on all categories: Average Entry Relevance Rate = {overall_avg_er:.4f} (n={len(overall_entry_relevance)})")

    # Print overall retrieval metrics result if enabled
    if eval_retrieval_metrics:
        if len(overall_precision_k) > 0:
            overall_avg_prec = statistics.mean(overall_precision_k)
            avg_overall_metrics_final["Precision@K"] = overall_avg_prec
            print(f"Overall on all categories: Average Precision@{eval_topk} = {overall_avg_prec:.4f} (n={len(overall_precision_k)})")
        
        if len(overall_recall_k) > 0:
            overall_avg_rec = statistics.mean(overall_recall_k)
            avg_overall_metrics_final["Recall@K"] = overall_avg_rec
            print(f"Overall on all categories: Average Recall@{eval_topk} = {overall_avg_rec:.4f} (n={len(overall_recall_k)})")
        
        if len(overall_hitrate_k) > 0:
            overall_avg_hr = statistics.mean(overall_hitrate_k)
            avg_overall_metrics_final["HitRate@K"] = overall_avg_hr
            print(f"Overall on all categories: Average HitRate@{eval_topk} = {overall_avg_hr:.4f} (n={len(overall_hitrate_k)})")

        avg_overall_metrics_final["Eval@K"] = eval_topk

    # Build return dictionary
    result_dict = {
        "avg_category_metrics_final": avg_category_metrics_final,
        "avg_overall_metrics_final": avg_overall_metrics_final,
    }
    if enable_llm_judge or enable_retrieval_judge:
        result_dict["judge_model"] = judge_model_name
        result_dict["judge_temperature"] = 0
    if enable_lexical_metrics:
        result_dict.update({
            "category_f1": {k: v for k, v in category_f1.items()},
            "overall_f1": overall_f1,
            "category_bleu": {k: v for k, v in category_bleu.items()},
            "overall_bleu": overall_bleu,
            "category_bleu_1": {k: v for k, v in category_bleu_1.items()},
            "overall_bleu_1": overall_bleu_1,
            "category_bleu_2": {k: v for k, v in category_bleu_2.items()},
            "overall_bleu_2": overall_bleu_2,
            "category_em": {k: v for k, v in category_em.items()},
            "overall_em": overall_em,
        })
    
    # Add LLM Judge results if enabled
    if enable_llm_judge:
        result_dict["category_llm_judge"] = {k: v for k, v in category_llm_judge.items()}
        result_dict["overall_llm_judge"] = overall_llm_judge
        result_dict["llm_judge_details"] = llm_judge_details
        result_dict["llm_judge_counts"] = {
            "valid": len(overall_llm_judge),
            "failed": sum('error' in item for item in llm_judge_details),
        }

    # Add Retrieval Judge results if enabled
    if enable_retrieval_judge:
        result_dict["category_clue_coverage"] = {k: v for k, v in category_clue_coverage.items()}
        result_dict["overall_clue_coverage"] = overall_clue_coverage
        result_dict["category_entry_relevance"] = {k: v for k, v in category_entry_relevance.items()}
        result_dict["overall_entry_relevance"] = overall_entry_relevance
        result_dict["retrieval_judge_details"] = retrieval_judge_details
        result_dict["retrieval_judge_counts"] = {
            "valid": len(overall_clue_coverage),
            "failed": sum('error' in item for item in retrieval_judge_details),
        }

    # Add retrieval metrics results if enabled
    if eval_retrieval_metrics:
        result_dict["category_precision_k"] = {k: v for k, v in category_precision_k.items()}
        result_dict["overall_precision_k"] = overall_precision_k
        result_dict["category_recall_k"] = {k: v for k, v in category_recall_k.items()}
        result_dict["overall_recall_k"] = overall_recall_k
        result_dict["category_hitrate_k"] = {k: v for k, v in category_hitrate_k.items()}
        result_dict["overall_hitrate_k"] = overall_hitrate_k
    
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', default='')
    parser.add_argument('--memory_name', default='NaiveRAG')
    parser.add_argument('--emb_mode', default='raw', choices=['caption', 'raw'],
                        help='Embedding mode, must match the run_bench emb_mode used')
    parser.add_argument('--data_name', default=None, help='Dataset name (e.g., Dog_Behavior_Research_Academic_Life or sample_id like 00001). If not specified and --all_datasets is used, will evaluate all datasets.')
    parser.add_argument('--all_datasets', action="store_true", help='Evaluate all available datasets and compute overall statistics')
    parser.add_argument('--data_root', default=None,
                        help='Root dir of new-style per-sample data layout (e.g. /OmniMemBench/data). '
                             'When set, --tier is also required.')
    parser.add_argument('--tier', default=None,
                        help='Benchmark tier (e.g. benchmark_128k). Required when --data_root is used.')
    # Retrieval metrics arguments
    parser.add_argument('--eval_retrieval_metrics', action="store_true", help='Evaluate retrieval metrics (Recall@K, HitRate@K, Precision@K)')
    parser.add_argument('--enable_lexical_metrics', action="store_true", help='Enable lexical metrics (F1, BLEU, BLEU_1, BLEU_2, EM). Disabled by default.')
    parser.add_argument('--eval_topk', type=int, default=10, help='Top-K cutoff for retrieval metrics')
    # LLM Judge arguments
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--enable_llm_judge', action="store_true", help='Enable LLM-as-a-judge evaluation')
    parser.add_argument('--enable_retrieval_judge', action="store_true", help='Enable LLM retrieval judge (Clue Coverage & Entry Relevance via llm_judge_retrieval.txt)')
    parser.add_argument('--judge_model', default=None, help='Judge model name (overrides default config)')
    parser.add_argument('--judge_api_key', default=None, help='Judge API key (overrides default config)')
    parser.add_argument('--judge_api_base', default=None, help='Judge API base URL (overrides default config)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for LLM judge calls')
    parser.add_argument('--use_external_api', action='store_true', help='Use external LLM API for LLM judge (reads LLM_API_KEY env var)')
    parser.add_argument('--result_dir', default=None, help='Override result directory (default: result_debug)')
    parser.add_argument('--result_subdir', default='', help='Subdirectory under llm_name (e.g. "store")')
    args = parser.parse_args()

    seed_everything(args.seed)

    # ── Override RESULT_DIR if specified ─────────────────────────────────────
    if args.result_dir:
        RESULT_DIR = Path(args.result_dir).resolve()

    _LLM_BASE = RESULT_DIR / args.llm_name
    if args.result_subdir:
        _LLM_BASE = _LLM_BASE / args.result_subdir

    # ── Data layout: new-style (--data_root + --tier) or legacy ──────────────────
    # Note: _DATA_ROOT and _TIER are module-level variables; no 'global' needed in __main__
    if args.data_root:
        if not args.tier:
            print("Error: --tier is required when --data_root is specified")
            sys.exit(1)
        _DATA_ROOT = Path(args.data_root).resolve()
        _TIER      = args.tier
        print(f"[DataLayout] new-style: data_root={_DATA_ROOT}  tier={_TIER}")
    else:
        _DATA_ROOT = None
        _TIER      = None

    # ── Set result scan dir for dataset discovery ────────────────────────────
    if _TIER:
        _RESULT_SCAN_DIR = _LLM_BASE / f"{args.memory_name}_{args.emb_mode}" / _TIER
    else:
        _RESULT_SCAN_DIR = _LLM_BASE / f"{args.memory_name}_{args.emb_mode}"

    # Feature string for file naming (empty for now, can be extended in the future)
    feature_str = ''

    # Load LLM judge configuration if enabled (shared by llm_judge and retrieval_judge)
    judge_config = None
    if args.enable_llm_judge or args.enable_retrieval_judge:
        judge_config = {
            'name': os.environ.get('JUDGE_MODEL') or 'glm-5.2',
            'api_key': os.environ.get('JUDGE_API_KEY') or os.environ.get('LLM_API_KEY', ''),
            'base_url': os.environ.get('JUDGE_API_BASE') or os.environ.get('LLM_API_BASE') or 'http://localhost:8000/v1',
        }

        # Override with command-line arguments if provided
        if args.judge_model:
            judge_config['name'] = args.judge_model
        if args.judge_api_key:
            judge_config['api_key'] = args.judge_api_key
        if args.judge_api_base:
            judge_config['base_url'] = args.judge_api_base

        if not judge_config.get('name') or not judge_config.get('api_key'):
            print("Warning: LLM judge enabled but name or api_key is empty. "
                  "Use --judge_model/--judge_api_key or --use_external_api.")

        # Set prompt path (only used by llm_judge)
        judge_config['prompt_path'] = 'llm_judge.txt'

    # Determine which datasets to evaluate
    if args.all_datasets:
        datasets = get_available_datasets()
        if not datasets:
            print("No datasets found in dialog directory!")
            # return
        print(f"Found {len(datasets)} datasets. Evaluating all datasets...")
        print(f"Datasets: {', '.join(datasets)}")
    else:
        if args.data_name is None:
            print("Error: Either --data_name or --all_datasets must be specified")
            # return
        datasets = [args.data_name]

    # Store individual results for overall aggregation
    all_individual_results = []

    # Evaluate each dataset
    for data_name in datasets:
        data_name_underscore = data_name  # Keep original name for file naming
        
        print(f"\n{'='*80}")
        print(f"Evaluating dataset: {data_name}")
        print(f"{'='*80}")
        
        # Build file path with feature flags
        if _DATA_ROOT and _TIER:
            # New-style: <llm_base>/<memory>_<emb>/<tier>/<sample_id>/<sample_id>_results.json
            result_dir = _LLM_BASE / f"{args.memory_name}_{args.emb_mode}" / _TIER / data_name
        else:
            result_dir = _LLM_BASE / f"{args.memory_name}_{args.emb_mode}" / data_name
        result_dir.mkdir(parents=True, exist_ok=True)
        file_path = result_dir / f"{data_name_underscore}_results{feature_str}.json"
        
        if not file_path.exists():
            print(f"Warning: Result file not found: {file_path}")
            print("Skipping this dataset...")
            continue
        
        print(f"Loading results from: {file_path}")
        
        try:
            evaluate_result = main(
                str(file_path),
                enable_llm_judge=args.enable_llm_judge,
                judge_config=judge_config,
                eval_retrieval_metrics=args.eval_retrieval_metrics,
                eval_topk=args.eval_topk,
                num_workers=args.num_workers,
                enable_lexical_metrics=args.enable_lexical_metrics,
                enable_retrieval_judge=args.enable_retrieval_judge,
            )
            
            # Store individual result with dataset name
            evaluate_result['dataset_name'] = data_name
            all_individual_results.append(evaluate_result)
            
            # Save individual result (compact format)
            try:
                output_path = result_dir / f"{data_name_underscore}_evaluate_result{feature_str}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(evaluate_result, f, ensure_ascii=False, separators=(',', ':'))
                print(f"Individual evaluation results saved to: {output_path}")
            except Exception as e:
                print(f"Error saving individual result: {e}")
                
        except Exception as e:
            print(f"Error evaluating dataset {data_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


    # Compute overall statistics if evaluating multiple datasets
    if args.all_datasets and len(all_individual_results) > 1:
        print(f"\n{'='*80}")
        print("Computing Overall Statistics Across All Datasets")
        print(f"{'='*80}")

        # Aggregate all metrics across datasets
        overall_aggregated = aggregate_overall_metrics(all_individual_results, eval_retrieval_metrics=args.eval_retrieval_metrics)

        # Print overall results
        print("\n" + "="*80)
        print("Overall Results Across All Datasets:")
        print("="*80)
        for metric_name, metric_value in sorted(overall_aggregated['avg_overall_metrics_final'].items()):
            if isinstance(metric_value, (int, float)):
                print(f"Overall {metric_name}: {metric_value:.4f}")

        # Save overall results (overall_statistics first for quick access)
        overall_result = {
            'overall_statistics': overall_aggregated,
            'num_datasets': len(all_individual_results),
            'datasets': datasets,
            'individual_results': all_individual_results
        }
        
        try:
            if _DATA_ROOT and _TIER:
                overall_dir = _LLM_BASE / f"{args.memory_name}_{args.emb_mode}" / _TIER
            else:
                overall_dir = _LLM_BASE / f"{args.memory_name}_{args.emb_mode}"
            overall_dir.mkdir(parents=True, exist_ok=True)
            overall_output_path = overall_dir / f"ALL_evaluate_result{feature_str}.json"
            with open(overall_output_path, "w", encoding="utf-8") as f:
                json.dump(overall_result, f, ensure_ascii=False, separators=(',', ':'))
            print(f"\nOverall evaluation results saved to: {overall_output_path}")
        except Exception as e:
            print(f"Error saving overall result: {e}")
    elif args.all_datasets:
        print(f"\nOnly {len(all_individual_results)} dataset(s) evaluated. Skipping overall statistics.")
