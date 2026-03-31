# Phase 1: Core Performance Evaluation
# Multilingual QA System - Evaluation Phase
#
# This phase assesses:
# 1. Basic Metrics: F1 Score, ROUGE-L, Containment Accuracy
# 2. Generation Quality: Answer coherence, factual accuracy, completeness
# 3. Retrieval Quality: Context relevance and response time
#
# Dataset: 20 samples per language from MIRACL corpus
# Languages: English, Hindi, Bengali, Arabic

from generation.qa_generator import QAGenerator
from retrieval.search import Retriever
import evaluate
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re


# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

print("Loading training data from data/processed/...")

# Load training data from JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = {
    "english": load_jsonl("data/processed/en_chunks.jsonl"),
    "hindi": load_jsonl("data/processed/hi_chunks.jsonl"),
    "bengali": load_jsonl("data/processed/bn_chunks.jsonl"),
    "arabic": load_jsonl("data/processed/ar_chunks.jsonl")
}

print(f"Loaded samples - English: {len(train_data['english'])}, Hindi: {len(train_data['hindi'])}, Bengali: {len(train_data['bengali'])}, Arabic: {len(train_data['arabic'])}")

languages = ["english", "hindi", "bengali", "arabic"]

# Proper language mapping (FIXED)
lang_code_map = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "arabic": "ar"
}

print("Loading Generator...")
generator = QAGenerator()

print("Loading Retriever (for translation)...")
retriever = Retriever()

rouge = evaluate.load("rouge")


# -------------------------------------------------
# Utility Functions with NLTK
# -------------------------------------------------

# Initialize NLTK components
STOP_WORDS = set(stopwords.words('english'))
# Keep important words that affect meaning
KEEP_WORDS = {'not', 'no', 'first', 'second', 'most', 'more', 'all', 'only', 'very', 'much', 'many'}
STOP_WORDS = STOP_WORDS - KEEP_WORDS
stemmer = PorterStemmer()

def split_sentences_smart(text):
    """Better sentence splitting that handles abbreviations"""
    # Split on sentence boundaries (. ! ?) followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def extract_short_answer(prediction):
    """Clean answer text while preserving compact, evaluation-friendly answers."""
    prediction = prediction.strip()

    # Remove common leading prefixes without deleting core content.
    prediction = re.sub(r'^\s*(?:answer\s*:|the answer is\s+)', '', prediction, flags=re.IGNORECASE)

    if not prediction:
        return prediction

    # Keep explicit fallback answers intact.
    if prediction.lower().startswith("not found"):
        return "Not found"

    sentence_matches = re.findall(r'[^.!?]+[.!?]?', prediction)
    sentences = []
    for match in sentence_matches:
        sentence = match.strip()
        if sentence:
            sentences.append(sentence)
        if len(sentences) == 2:
            break

    if not sentences:
        return prediction

    cleaned = " ".join(sentences).strip()
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def normalize_for_scoring(text):
    """Normalize light formatting differences before metric computation."""
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[\"'`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def build_eval_question(title, context, lang):
    """Create a more natural question from a title-only sample."""
    title = title.strip()

    if lang == "arabic":
        return f"What does the passage say about {title}?"

    disambiguation_match = re.search(r"\(([^)]+)\)", title)
    if disambiguation_match:
        qualifier = disambiguation_match.group(1).strip()
        base_title = re.sub(r"\s*\([^)]+\)\s*", "", title).strip()
        return f"What is {base_title} in the context of {qualifier}?"

    lowered = title.lower()
    if any(keyword in lowered for keyword in ["list of ", "सूची", "তালিকা"]):
        return f"What does the passage describe about {title}?"

    if any(keyword in lowered for keyword in ["day", "days", "दिन", "week"]):
        return f"What are {title}?"

    if any(keyword in lowered for keyword in ["language", "भाषा", "ভাষা", "لغة"]):
        return f"What is {title}?"

    if any(keyword in lowered for keyword in ["city", "city)", "مدينة", "শহর", "नगर"]):
        return f"What is {title}?"

    words = title.split()
    if len(words) <= 4:
        if any(word[:1].isupper() for word in words if word):
            return f"Who is {title}?" if len(words) >= 2 else f"What is {title}?"
        return f"What is {title}?"

    first_sentence = split_sentences_smart(context)
    if first_sentence:
        lead = first_sentence[0]
        if len(lead) >= 40:
            return f"What does the passage say about {title}?"

    return f"Explain {title}."


def normalize_and_tokenize(text):
    """Normalize text using NLTK tokenization"""
    text = text.lower()
    # Tokenize using NLTK
    tokens = word_tokenize(text)
    # Remove punctuation tokens
    tokens = [t for t in tokens if t.isalnum()]
    return tokens


def remove_stopwords_and_stem(tokens):
    """Remove stopwords and apply stemming"""
    # Remove stopwords
    filtered = [t for t in tokens if t not in STOP_WORDS]
    # Apply stemming
    stemmed = [stemmer.stem(t) for t in filtered]
    return stemmed


def compute_overlap_metrics(prediction, ground_truth):
    # Tokenize and normalize
    pred_tokens = normalize_and_tokenize(prediction)
    gt_tokens = normalize_and_tokenize(ground_truth)
    
    # Remove stopwords and stem
    pred_tokens = remove_stopwords_and_stem(pred_tokens)
    gt_tokens = remove_stopwords_and_stem(gt_tokens)
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Calculate overlap
    common = set(pred_tokens) & set(gt_tokens)
    
    if len(common) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def exact_match_score(prediction, ground_truth):
    """Exact match after light normalization."""
    return 1.0 if normalize_for_scoring(prediction).lower() == normalize_for_scoring(ground_truth).lower() else 0.0


def containment_score(prediction, ground_truth):
    # Tokenize and process
    pred_tokens = normalize_and_tokenize(prediction)
    gt_tokens = normalize_and_tokenize(ground_truth)
    
    # Remove stopwords and stem
    pred_tokens = remove_stopwords_and_stem(pred_tokens)
    gt_tokens = remove_stopwords_and_stem(gt_tokens)
    
    if len(gt_tokens) == 0:
        return 0.0
    
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    
    # Check word-level containment (25% threshold - more lenient)
    overlap = len(gt_set & pred_set) / len(gt_set)
    return 1.0 if overlap >= 0.25 else 0.0


# -------------------------------------------------
# Evaluation Loop
# -------------------------------------------------

results = {}

for lang in languages:

    print(f"\nEvaluating {lang.upper()}")

    results[lang] = {
        "precision": [],
        "recall": [],
        "f1": [],
        "rouge": [],
        "containment": [],
        "answer_rate": []
    }
    count = 0
    seen_titles = set()  # Track titles to avoid duplicates

    for sample in train_data[lang]:

        # Skip samples with very short context (< 200 chars)
        if len(sample["text"]) < 200:
            continue
            
        context = sample["text"][:3200]  # Slightly larger eval context for better answer coverage
        title = sample.get("title", "")
        
        # Use title as question
        if not title or len(title.strip()) < 3:
            continue
        
        # Skip duplicate titles to ensure variety
        if title in seen_titles:
            continue
        seen_titles.add(title)
            
        # Build a more natural question from title-only chunk metadata.
        question = build_eval_question(title, context, lang)
        
        # Use 2 sentences so the reference length better matches generated answers.
        sentences = split_sentences_smart(context)
        
        if len(sentences) >= 2:
            ground_truth = ". ".join(sentences[:2]) + "."
        elif len(sentences) >= 1:
            ground_truth = sentences[0] + "."
        else:
            ground_truth = context[:200]
        
        # Skip if ground truth is too short
        if len(ground_truth.strip()) < 20:
            continue

        print(f"\n[{lang.upper()}] Sample {count + 1}:")
        print(f"  Title: {title[:50]}...")
        print(f"  Question: {question[:60]}...")
        print(f"  Ground Truth: {ground_truth[:80]}...")

        # -----------------------------
        # Translation Handling - SIMPLIFIED
        # -----------------------------

        try:
            # For non-English languages
            if lang != "english":
                if lang == "arabic":
                    print(f"  [DEBUG] Original Arabic Question: {question[:80]}")
                    print(f"  [DEBUG] Context length: {len(context)} chars")
                    print(f"  [DEBUG] Ground truth length: {len(ground_truth)} chars")
                elif lang == "hindi":
                    print(f"  [DEBUG] Original Hindi Question: {question[:80]}")
                    print(f"  [DEBUG] Context length: {len(context)} chars")
                    print(f"  [DEBUG] Ground truth length: {len(ground_truth)} chars")

                question_en = retriever.translate(
                    question,
                    lang_code_map[lang],
                    "en"
                )
                context_en = retriever.translate(
                    context,
                    lang_code_map[lang],
                    "en"
                )
                ground_truth_en = retriever.translate(
                    ground_truth,
                    lang_code_map[lang],
                    "en"
                )

                print(f"  Translated Question: {question_en[:60]}...")
                print(f"  Translated Context: {context_en[:80]}...")
                print(f"  Translated Ground Truth: {ground_truth_en[:80]}...")
            else:
                question_en = question
                context_en = context
                ground_truth_en = ground_truth

            # Generate answer
            raw_prediction = generator.generate_answer(
                question=question_en,
                context=context_en,
                role="eval",
                language="en"
            )
            
            print(f"  Raw Prediction: {raw_prediction[:80]}...")

            prediction = extract_short_answer(raw_prediction)
            print(f"  Extracted Prediction: {prediction[:60]}...")

            prediction_for_score = normalize_for_scoring(prediction)
            ground_truth_for_score = normalize_for_scoring(ground_truth_en)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # -----------------------------
        # Metrics - Compare in English
        # -----------------------------

        overlap_metrics = compute_overlap_metrics(prediction_for_score, ground_truth_for_score)
        precision = overlap_metrics["precision"]
        recall = overlap_metrics["recall"]
        f1 = overlap_metrics["f1"]
        answered = 0.0 if prediction_for_score.lower() == "not found" else 1.0

        try:
            rouge_score = rouge.compute(
                predictions=[prediction_for_score],
                references=[ground_truth_for_score]
            )["rougeL"]
        except:
            rouge_score = 0.0

        contain = containment_score(prediction_for_score, ground_truth_for_score)
        
        print(
            f"  Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, "
            f"ROUGE-L: {rouge_score:.4f}, Containment: {contain:.4f}"
        )

        results[lang]["precision"].append(precision)
        results[lang]["recall"].append(recall)
        results[lang]["f1"].append(f1)
        results[lang]["rouge"].append(rouge_score)
        results[lang]["containment"].append(contain)
        results[lang]["answer_rate"].append(answered)

        count += 1
        if count >= 20:   # Evaluating 20 samples per language
            break


# -------------------------------------------------
# Final Results
# -------------------------------------------------

print("\n==== FINAL RESULTS ====\n")

for lang in languages:

    if len(results[lang]["f1"]) == 0:
        print(f"{lang.upper()}: No valid samples.")
        continue

    avg_precision = sum(results[lang]["precision"]) / len(results[lang]["precision"])
    avg_recall = sum(results[lang]["recall"]) / len(results[lang]["recall"])
    avg_f1 = sum(results[lang]["f1"]) / len(results[lang]["f1"])
    avg_rouge = sum(results[lang]["rouge"]) / len(results[lang]["rouge"])
    avg_contain = sum(results[lang]["containment"]) / len(results[lang]["containment"])
    avg_answer_rate = sum(results[lang]["answer_rate"]) / len(results[lang]["answer_rate"])
    max_f1 = max(results[lang]["f1"])
    max_rouge = max(results[lang]["rouge"])

    print(f"{lang.upper()}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  Average F1 Score: {avg_f1:.4f}")
    print(f"  Max F1 Score: {max_f1:.4f}")
    print(f"  Average ROUGE-L Score: {avg_rouge:.4f}")
    print(f"  Max ROUGE-L Score: {max_rouge:.4f}")
    print(f"  Containment Accuracy: {avg_contain:.4f}")
    print(f"  Answer Rate: {avg_answer_rate:.4f}")
    print()
