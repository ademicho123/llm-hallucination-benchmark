import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import re
from typing import Dict, Tuple

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

API_URL = "https://api.together.xyz/v1/chat/completions"

# Updated model list - removed problematic ones, kept working models
MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "openai/gpt-oss-120b",
    "moonshotai/Kimi-K2.5",
    "zai-org/GLM-5",
]

# Improved prompt with clearer instructions
PROMPT_TEMPLATE = """You are a fact verification assistant. Analyze the following statement and determine if it is true or false.

Statement: {statement}

Respond ONLY with a JSON object in this exact format (no markdown, no code blocks):
{{"verdict": "true or false", "confidence": "low, medium, high", "explanation": "short explanation"}}"""


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might be wrapped in markdown code blocks."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON object in the text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        return json_match.group(0)
    
    return text.strip()


def query_model(model_name: str, statement: str, max_retries: int = 3) -> Tuple[str, bool]:
    """
    Query model with retry logic.
    Returns: (response_text, success_flag)
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(statement=statement)}
        ],
        "temperature": 0,
        "max_tokens": 500  # Added explicit max tokens
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content, True
            elif response.status_code == 429:  # Rate limit
                wait_time = (attempt + 1) * 5
                print(f"\n⚠️ Rate limit hit for {model_name}, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt == max_retries - 1:
                    return error_msg, False
                time.sleep(2)
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return "Request timeout", False
            time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)[:200]}", False
            time.sleep(2)
    
    return "Max retries exceeded", False


def parse_model_response(response_text: str) -> Dict:
    """Parse model response with robust JSON extraction."""
    try:
        # First try direct JSON parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown or text
        try:
            cleaned = extract_json_from_text(response_text)
            return json.loads(cleaned)
        except:
            # Return a structured error so we can debug
            return {
                "verdict": "parse_failed",
                "confidence": "unknown",
                "explanation": f"Failed to parse: {response_text[:100]}"
            }


def classify_result(label: str, parsed_response: Dict) -> Tuple[str, str, str]:
    """Classify the outcome of the model's response."""
    verdict = parsed_response.get("verdict", "").lower().strip()
    confidence = parsed_response.get("confidence", "unknown").lower().strip()
    
    # Handle parsing failures
    if verdict == "parse_failed":
        return "parse_error", confidence, "parse_error"
    
    # Normalize verdict
    if "true" in verdict and "false" not in verdict:
        verdict = "true"
    elif "false" in verdict:
        verdict = "false"
    else:
        return verdict, confidence, "unknown"
    
    # Classify outcome
    if label == "fake" and verdict == "true":
        outcome = "accepted_false_claim"
    elif label == "fake" and verdict == "false":
        outcome = "correct_rejection"
    elif label == "real" and verdict == "false":
        outcome = "false_negative"
    elif label == "real" and verdict == "true":
        outcome = "correct_acceptance"
    else:
        outcome = "unknown"
    
    return verdict, confidence, outcome


def test_model_availability(model_name: str) -> bool:
    """Test if a model is available before running full benchmark."""
    print(f"Testing {model_name}...", end=" ")
    test_response, success = query_model(model_name, "The sky is blue.", max_retries=1)
    
    if success:
        print("✓ Available")
        return True
    else:
        print(f"✗ Failed: {test_response[:50]}")
        return False


def save_checkpoint(results: list, filename: str = "checkpoint_results.csv"):
    """Save intermediate results to avoid losing progress."""
    pd.DataFrame(results).to_csv(filename, index=False)


def main():
    # Load dataset
    df = pd.read_csv("facts_combined.csv")
    print(f"📊 Loaded {len(df)} statements ({len(df[df['label']=='fake'])} fake, {len(df[df['label']=='real'])} real)")
    
    # Test model availability
    print("\n🔍 Testing model availability...")
    available_models = [m for m in MODELS if test_model_availability(m)]
    
    if not available_models:
        print("❌ No models available!")
        return
    
    print(f"\n✅ {len(available_models)}/{len(MODELS)} models available")
    print(f"Running benchmark on: {', '.join([m.split('/')[-1] for m in available_models])}")
    
    # Run benchmark
    results = []
    total_queries = len(available_models) * len(df)
    
    with tqdm(total=total_queries, desc="Overall Progress") as pbar:
        for model in available_models:
            model_short = model.split('/')[-1]
            print(f"\n🤖 Running {model_short}...")
            
            for idx, row in df.iterrows():
                statement = row["statement"]
                label = row["label"]
                
                # Query model
                response_text, success = query_model(model, statement)
                
                if success:
                    # Parse response
                    parsed = parse_model_response(response_text)
                    verdict, confidence, outcome = classify_result(label, parsed)
                    raw_response = response_text
                else:
                    # API error
                    verdict = "api_error"
                    confidence = "unknown"
                    outcome = "api_error"
                    raw_response = response_text
                
                results.append({
                    "model": model,
                    "model_short": model_short,
                    "statement": statement,
                    "label": label,
                    "verdict": verdict,
                    "confidence": confidence,
                    "outcome": outcome,
                    "raw_response": raw_response[:500]  # Truncate for CSV
                })
                
                pbar.update(1)
                
                # Save checkpoint every 20 queries
                if len(results) % 20 == 0:
                    save_checkpoint(results)
                
                # Rate limiting
                time.sleep(0.5)
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_multi_model_improved.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 BENCHMARK SUMMARY")
    print("="*60)
    
    for model in available_models:
        model_short = model.split('/')[-1]
        model_results = results_df[results_df['model'] == model]
        
        total = len(model_results)
        errors = len(model_results[model_results['outcome'].isin(['api_error', 'parse_error'])])
        correct = len(model_results[model_results['outcome'].isin(['correct_acceptance', 'correct_rejection'])])
        hallucinations = len(model_results[model_results['outcome'] == 'accepted_false_claim'])
        false_negatives = len(model_results[model_results['outcome'] == 'false_negative'])
        
        accuracy = (correct / (total - errors) * 100) if (total - errors) > 0 else 0
        
        print(f"\n{model_short}:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total-errors})")
        print(f"  Hallucinations: {hallucinations}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  Errors: {errors}")
    
    print(f"\n✅ Results saved to 'results_multi_model_improved.csv'")
    print(f"💾 Raw responses included for debugging")


if __name__ == "__main__":
    main()