"""
Evaluation module for calculating precision, recall, and F1 scores
"""
import json
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

from knowledge_graph import fuzzy_match, normalize_entity_name, FUZZY_MATCH_THRESHOLD


def load_reference_annotations(file_path: str) -> Dict:
    """
    Load reference annotations from JSON file.
    
    Args:
        file_path: Path to reference annotations JSON file
        
    Returns:
        Dictionary with reference entities and relationships
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def match_entity(predicted: Dict, reference: Dict, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """
    Check if predicted entity matches reference entity.
    
    Args:
        predicted: Predicted entity dictionary
        reference: Reference entity dictionary
        threshold: Similarity threshold
        
    Returns:
        True if entities match
    """
    pred_name = predicted.get("name", "").strip()
    ref_name = reference.get("name", "").strip()
    pred_type = predicted.get("type", "").strip()
    ref_type = reference.get("type", "").strip()
    
    # Normalize names for comparison
    pred_name_norm = normalize_entity_name(pred_name)
    ref_name_norm = normalize_entity_name(ref_name)
    
    name_match = fuzzy_match(pred_name, ref_name) >= threshold
    
    # Type matching - handle variations (e.g., "Technology" vs "Operating System")
    # For now, require exact match, but could be made more flexible
    type_match = pred_type.lower() == ref_type.lower()
    
    return name_match and type_match


def match_relationship(predicted: Dict, reference: Dict, 
                      entity_mapping: Dict[str, str] = None,
                      threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """Check if predicted relationship matches reference relationship."""
    pred_source = predicted.get("source", "").strip()
    pred_target = predicted.get("target", "").strip()
    pred_type = predicted.get("type", "").strip()
    
    ref_source = reference.get("source", "").strip()
    ref_target = reference.get("target", "").strip()
    ref_type = reference.get("type", "").strip()
    
    # Normalize names for comparison
    pred_source_norm = normalize_entity_name(pred_source)
    pred_target_norm = normalize_entity_name(pred_target)
    ref_source_norm = normalize_entity_name(ref_source)
    ref_target_norm = normalize_entity_name(ref_target)
    
    source_match = fuzzy_match(pred_source, ref_source) >= threshold
    target_match = fuzzy_match(pred_target, ref_target) >= threshold
    
    # Type matching - handle variations (e.g., "develops" vs "develops_operating_system")
    # Check if types match or if one is a prefix/suffix of the other
    pred_type_lower = pred_type.lower()
    ref_type_lower = ref_type.lower()
    type_match = (pred_type_lower == ref_type_lower or 
                  pred_type_lower in ref_type_lower or 
                  ref_type_lower in pred_type_lower)
    
    return source_match and target_match and type_match


def calculate_entity_metrics(predicted_entities: List[Dict], 
                             reference_entities: List[Dict]) -> Dict:

    if not reference_entities:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(predicted_entities),
            "false_negatives": 0
        }
    
    if not predicted_entities:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(reference_entities)
        }
    
    true_positives = 0
    matched_references = set()
    
    for pred_entity in predicted_entities:
        for idx, ref_entity in enumerate(reference_entities):
            if idx not in matched_references and match_entity(pred_entity, ref_entity):
                true_positives += 1
                matched_references.add(idx)
                break
    
    false_positives = len(predicted_entities) - true_positives
    false_negatives = len(reference_entities) - true_positives
    
    precision = true_positives / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positives / len(reference_entities) if reference_entities else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def calculate_relationship_metrics(predicted_relationships: List[Dict],
                                    reference_relationships: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1 for relationship extraction."""
    if not reference_relationships:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(predicted_relationships),
            "false_negatives": 0
        }
    
    if not predicted_relationships:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(reference_relationships)
        }
    
    true_positives = 0
    matched_references = set()
    
    for pred_rel in predicted_relationships:
        for idx, ref_rel in enumerate(reference_relationships):
            if idx not in matched_references and match_relationship(pred_rel, ref_rel):
                true_positives += 1
                matched_references.add(idx)
                break
    
    false_positives = len(predicted_relationships) - true_positives
    false_negatives = len(reference_relationships) - true_positives
    
    precision = true_positives / len(predicted_relationships) if predicted_relationships else 0.0
    recall = true_positives / len(reference_relationships) if reference_relationships else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def evaluate_extraction(predicted_graph: Dict, reference_annotations: Dict) -> Dict:
    """Evaluate predicted knowledge graph against reference annotations."""
    # Extract entities and relationships from predicted graph
    predicted_entities = []
    for node in predicted_graph.get("nodes", []):
        predicted_entities.append({
            "name": node.get("label", ""),
            "type": node.get("type", "")
        })
    
    predicted_relationships = []
    for edge in predicted_graph.get("edges", []):
        # Get source and target labels
        source_label = next((n.get("label", "") for n in predicted_graph.get("nodes", [])
                            if n.get("id") == edge.get("source")), "")
        target_label = next((n.get("label", "") for n in predicted_graph.get("nodes", [])
                            if n.get("id") == edge.get("target")), "")
        
        predicted_relationships.append({
            "source": source_label,
            "target": target_label,
            "type": edge.get("relationship", "")
        })
    
    # Get reference entities and relationships
    reference_entities = reference_annotations.get("entities", [])
    reference_relationships = reference_annotations.get("relationships", [])
    
    # Calculate metrics
    entity_metrics = calculate_entity_metrics(predicted_entities, reference_entities)
    relationship_metrics = calculate_relationship_metrics(predicted_relationships, 
                                                           reference_relationships)
    
    return {
        "entities": entity_metrics,
        "relationships": relationship_metrics,
        "overall": {
            "precision": (entity_metrics["precision"] + relationship_metrics["precision"]) / 2,
            "recall": (entity_metrics["recall"] + relationship_metrics["recall"]) / 2,
            "f1": (entity_metrics["f1"] + relationship_metrics["f1"]) / 2
        }
    }


def compare_approaches(zero_shot_results: Dict, few_shot_results: Dict,
                       reference_annotations: Dict) -> Dict:
    """Compare zero-shot and few-shot extraction approaches."""
    zero_shot_metrics = evaluate_extraction(zero_shot_results, reference_annotations)
    few_shot_metrics = evaluate_extraction(few_shot_results, reference_annotations)
    
    return {
        "zero_shot": zero_shot_metrics,
        "few_shot": few_shot_metrics,
        "improvement": {
            "entity_f1": few_shot_metrics["entities"]["f1"] - zero_shot_metrics["entities"]["f1"],
            "relationship_f1": few_shot_metrics["relationships"]["f1"] - zero_shot_metrics["relationships"]["f1"],
            "overall_f1": few_shot_metrics["overall"]["f1"] - zero_shot_metrics["overall"]["f1"]
        }
    }

