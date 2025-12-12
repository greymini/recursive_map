"""
Evaluation script for comparing extraction approaches
"""
import argparse
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator import load_reference_annotations, evaluate_extraction, compare_approaches
from config import OUTPUTS_DIR, REFERENCE_ANNOTATIONS_DIR


def evaluate_results(predicted_graph_path: str, reference_annotations_path: str):
    """
    Evaluate predicted graph against reference annotations.
    
    Args:
        predicted_graph_path: Path to predicted knowledge graph JSON
        reference_annotations_path: Path to reference annotations JSON
    """
    print(f"Loading predicted graph: {predicted_graph_path}")
    with open(predicted_graph_path, 'r', encoding='utf-8') as f:
        predicted_graph = json.load(f)
    
    print(f"Loading reference annotations: {reference_annotations_path}")
    reference_annotations = load_reference_annotations(reference_annotations_path)
    
    print("Calculating metrics...")
    metrics = evaluate_extraction(predicted_graph, reference_annotations)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nEntity Extraction:")
    print(f"  Precision: {metrics['entities']['precision']:.3f}")
    print(f"  Recall:    {metrics['entities']['recall']:.3f}")
    print(f"  F1 Score:  {metrics['entities']['f1']:.3f}")
    print(f"  TP: {metrics['entities']['true_positives']}, "
          f"FP: {metrics['entities']['false_positives']}, "
          f"FN: {metrics['entities']['false_negatives']}")
    
    print(f"\nRelationship Extraction:")
    print(f"  Precision: {metrics['relationships']['precision']:.3f}")
    print(f"  Recall:    {metrics['relationships']['recall']:.3f}")
    print(f"  F1 Score:  {metrics['relationships']['f1']:.3f}")
    print(f"  TP: {metrics['relationships']['true_positives']}, "
          f"FP: {metrics['relationships']['false_positives']}, "
          f"FN: {metrics['relationships']['false_negatives']}")
    
    print(f"\nOverall:")
    print(f"  Precision: {metrics['overall']['precision']:.3f}")
    print(f"  Recall:    {metrics['overall']['recall']:.3f}")
    print(f"  F1 Score:  {metrics['overall']['f1']:.3f}")
    print("="*60)
    
    # Save results
    results_path = predicted_graph_path.replace(".json", "_evaluation.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation results saved to {results_path}")


def compare_approaches_evaluation(zero_shot_path: str, few_shot_path: str,
                                  reference_annotations_path: str):
    """
    Compare zero-shot and few-shot approaches.
    
    Args:
        zero_shot_path: Path to zero-shot graph JSON
        few_shot_path: Path to few-shot graph JSON
        reference_annotations_path: Path to reference annotations JSON
    """
    print("Loading graphs and annotations...")
    
    with open(zero_shot_path, 'r', encoding='utf-8') as f:
        zero_shot_graph = json.load(f)
    
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        few_shot_graph = json.load(f)
    
    reference_annotations = load_reference_annotations(reference_annotations_path)
    
    print("Comparing approaches...")
    comparison = compare_approaches(zero_shot_graph, few_shot_graph, reference_annotations)
    
    print("\n" + "="*60)
    print("APPROACH COMPARISON")
    print("="*60)
    
    print("\nZero-Shot Approach:")
    print(f"  Entity F1:      {comparison['zero_shot']['entities']['f1']:.3f}")
    print(f"  Relationship F1: {comparison['zero_shot']['relationships']['f1']:.3f}")
    print(f"  Overall F1:    {comparison['zero_shot']['overall']['f1']:.3f}")
    
    print("\nFew-Shot + Self-Critique Approach:")
    print(f"  Entity F1:      {comparison['few_shot']['entities']['f1']:.3f}")
    print(f"  Relationship F1: {comparison['few_shot']['relationships']['f1']:.3f}")
    print(f"  Overall F1:    {comparison['few_shot']['overall']['f1']:.3f}")
    
    print("\nImprovement:")
    print(f"  Entity F1:      {comparison['improvement']['entity_f1']:+.3f}")
    print(f"  Relationship F1: {comparison['improvement']['relationship_f1']:+.3f}")
    print(f"  Overall F1:    {comparison['improvement']['overall_f1']:+.3f}")
    print("="*60)
    
    # Save comparison
    comparison_path = os.path.join(OUTPUTS_DIR, "approach_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison results saved to {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Graph Extraction")
    parser.add_argument("--predicted", "-p", 
                       help="Path to predicted knowledge graph JSON")
    parser.add_argument("--reference", "-r", required=True,
                       help="Path to reference annotations JSON")
    parser.add_argument("--zero-shot", 
                       help="Path to zero-shot graph JSON (for comparison)")
    parser.add_argument("--few-shot",
                       help="Path to few-shot graph JSON (for comparison)")
    
    args = parser.parse_args()
    
    if args.predicted:
        evaluate_results(args.predicted, args.reference)
    elif args.zero_shot and args.few_shot:
        compare_approaches_evaluation(args.zero_shot, args.few_shot, args.reference)
    else:
        print("Error: Either provide --predicted or both --zero-shot and --few-shot")
        parser.print_help()


if __name__ == "__main__":
    main()

