import os
from tqdm import tqdm
import pandas as pd

def evaluate_mlm(model, masked_sentences, top_k=5):
    """
    Evaluate the model performance on masked token prediction
    
    Args:
        model: The EmbeddingModel instance with MLM capabilities
        masked_sentences: List of tuples (masked_sentence, expected_word)
        top_k: Consider prediction correct if expected word is in top_k predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = []
    correct_predictions = 0
    top_k_correct = 0
    
    for masked_sent, expected in tqdm(masked_sentences, desc="Evaluating MLM"):
        predictions = model.predict_masked_token(masked_sent, top_k=top_k)
        
        if not predictions:
            continue
            
        # Check if the top prediction matches the expected word
        top_prediction = predictions[0][0].replace("▁", "")  # Remove the underscore added by sentencepiece tokenizer
        is_correct = top_prediction.lower() == expected.lower()
        
        # Check if any of the top-k predictions match
        in_top_k = any(p[0].replace("▁", "").lower() == expected.lower() for p in predictions)
        
        if is_correct:
            correct_predictions += 1
        if in_top_k:
            top_k_correct += 1
            
        results.append({
            'masked_sentence': masked_sent,
            'expected': expected,
            'predicted': top_prediction,
            'correct': is_correct,
            'in_top_k': in_top_k,
            'confidence': predictions[0][1],
            'all_predictions': predictions
        })
    
    # Calculate metrics
    total = len(masked_sentences)
    accuracy = correct_predictions / total if total > 0 else 0
    top_k_accuracy = top_k_correct / total if total > 0 else 0
    
    metrics = {
        'total': total,
        'top_1_correct': correct_predictions,
        'top_k_correct': top_k_correct,
        'top_1_accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy,
        'results': results
    }
    
    return metrics

def print_evaluation_results(metrics):
    """Print evaluation results in a readable format"""
    print(f"\n===== MLM Evaluation Results =====")
    print(f"Total sentences evaluated: {metrics['total']}")
    print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.4f} ({metrics['top_1_correct']}/{metrics['total']})")
    print(f"Top-K Accuracy: {metrics['top_k_accuracy']:.4f} ({metrics['top_k_correct']}/{metrics['total']})")
    
    # Print some examples (correct and incorrect)
    print("\n----- Example Results -----")
    results = metrics['results']
    
    # Print some correct examples
    correct_examples = [r for r in results if r['correct']][:3]
    print("\nCorrect Predictions:")
    for i, example in enumerate(correct_examples, 1):
        print(f"{i}. Sentence: {example['masked_sentence']}")
        print(f"   Expected: {example['expected']}, Predicted: {example['predicted']} (Confidence: {example['confidence']:.4f})")
    
    # Print some incorrect examples
    incorrect_examples = [r for r in results if not r['correct']][:3]
    print("\nIncorrect Predictions:")
    for i, example in enumerate(incorrect_examples, 1):
        print(f"{i}. Sentence: {example['masked_sentence']}")
        print(f"   Expected: {example['expected']}, Predicted: {example['predicted']} (Confidence: {example['confidence']:.4f})")
        print(f"   All predictions: {', '.join([f'{p[0]} ({p[1]:.4f})' for p in example['all_predictions'][:3]])}")

def save_evaluation_results(metrics, output_dir):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total sentences: {metrics['total']}\n")
        f.write(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.4f} ({metrics['top_1_correct']}/{metrics['total']})\n")
        f.write(f"Top-K Accuracy: {metrics['top_k_accuracy']:.4f} ({metrics['top_k_correct']}/{metrics['total']})\n")
    
    # Save detailed results as CSV
    results_df = pd.DataFrame([
        {
            'masked_sentence': r['masked_sentence'],
            'expected': r['expected'],
            'predicted': r['predicted'],
            'correct': r['correct'],
            'in_top_k': r['in_top_k'],
            'confidence': r['confidence'],
            'top_predictions': str([f"{p[0]}:{p[1]:.4f}" for p in r['all_predictions'][:3]])
        }
        for r in metrics['results']
    ])
    
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False, encoding="utf-8")
    
    print(f"Evaluation results saved to {output_dir}")
