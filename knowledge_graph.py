import json
from typing import Dict, List, Any
from difflib import SequenceMatcher
from collections import defaultdict

from config import CONFIDENCE_THRESHOLD, FUZZY_MATCH_THRESHOLD, ENTITY_TYPES, RELATIONSHIP_TYPES


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for comparison (remove punctuation, extra spaces)."""
    if not name:
        return ""
    # Remove trailing punctuation
    name = name.rstrip('.,;:!?')
    # Normalize whitespace
    name = ' '.join(name.split())
    return name.strip()


def fuzzy_match(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings."""
    # Normalize both strings
    norm1 = normalize_entity_name(str1).lower()
    norm2 = normalize_entity_name(str2).lower()
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Fuzzy match
    return SequenceMatcher(None, norm1, norm2).ratio()


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entities using fuzzy matching."""
    if not entities:
        return []
    
    # Filter by confidence threshold first
    filtered = [e for e in entities if e.get("confidence", 0) >= CONFIDENCE_THRESHOLD]
    
    if not filtered:
        return []
    
    # Group similar entities
    unique_entities = []
    seen_names = set()
    
    for entity in filtered:
        name = entity.get("name", "").strip()
        if not name:
            continue
        
        # Check for fuzzy matches
        is_duplicate = False
        for seen_name in seen_names:
            similarity = fuzzy_match(name, seen_name)
            if similarity >= FUZZY_MATCH_THRESHOLD:
                is_duplicate = True
                # Keep the one with higher confidence
                for ue in unique_entities:
                    if fuzzy_match(ue["name"], seen_name) >= FUZZY_MATCH_THRESHOLD:
                        if entity.get("confidence", 0) > ue.get("confidence", 0):
                            unique_entities.remove(ue)
                            unique_entities.append(entity)
                            seen_names.remove(seen_name)
                            seen_names.add(name)
                        break
                break
        
        if not is_duplicate:
            unique_entities.append(entity)
            seen_names.add(name)
    
    return unique_entities


def validate_relationships(relationships: List[Dict], entities: List[Dict]) -> List[Dict]:
    """Validate relationships: ensure source and target entities exist."""
    if not relationships or not entities:
        return []
    
    # Create entity name lookup
    entity_names = {e.get("name", "").strip().lower() for e in entities}
    
    # Filter relationships
    validated = []
    for rel in relationships:
        # Check confidence threshold
        if rel.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            continue
        
        # Check relationship type
        rel_type = rel.get("type", "")
        if rel_type not in RELATIONSHIP_TYPES:
            continue
        
        # Check if source and target exist (fuzzy match)
        source = rel.get("source", "").strip()
        target = rel.get("target", "").strip()
        
        source_exists = any(fuzzy_match(source.lower(), en) >= FUZZY_MATCH_THRESHOLD 
                          for en in entity_names)
        target_exists = any(fuzzy_match(target.lower(), en) >= FUZZY_MATCH_THRESHOLD 
                          for en in entity_names)
        
        if source_exists and target_exists:
            validated.append(rel)
    
    return validated


def build_graph(entities: List[Dict], relationships: List[Dict], 
                document_source: str = "") -> Dict:
    """Build knowledge graph JSON structure from entities and relationships."""
    # Deduplicate entities
    unique_entities = deduplicate_entities(entities)
    
    # Validate relationships
    validated_relationships = validate_relationships(relationships, unique_entities)
    
    # Build nodes
    nodes = []
    entity_name_to_id = {}
    
    for idx, entity in enumerate(unique_entities):
        node_id = f"n{idx + 1}"
        entity_name_to_id[entity.get("name", "").strip()] = node_id
        
        nodes.append({
            "id": node_id,
            "label": entity.get("name", "").strip(),
            "type": entity.get("type", "Unknown"),
            "confidence": entity.get("confidence", 0.0),
            "context": entity.get("context", ""),
            "source": document_source
        })
    
    # Build edges
    edges = []
    source_mapping = {}
    unmatched_relationships = []
    
    # Create normalized lookup for better matching
    normalized_entity_lookup = {}
    for name, eid in entity_name_to_id.items():
        normalized_name = normalize_entity_name(name).lower()
        if normalized_name not in normalized_entity_lookup:
            normalized_entity_lookup[normalized_name] = []
        normalized_entity_lookup[normalized_name].append((name, eid))
    
    for idx, rel in enumerate(validated_relationships):
        source_name = rel.get("source", "").strip()
        target_name = rel.get("target", "").strip()
        
        # Find entity IDs (fuzzy match) - improved matching
        source_id = None
        target_id = None
        best_source_match = 0.0
        best_target_match = 0.0
        best_source_name = None
        best_target_name = None
        
        # Normalize source and target names for matching
        source_normalized = normalize_entity_name(source_name).lower()
        target_normalized = normalize_entity_name(target_name).lower()
        
        # First try exact normalized match
        if source_normalized in normalized_entity_lookup:
            source_id = normalized_entity_lookup[source_normalized][0][1]
            best_source_name = normalized_entity_lookup[source_normalized][0][0]
            best_source_match = 1.0
        
        if target_normalized in normalized_entity_lookup:
            target_id = normalized_entity_lookup[target_normalized][0][1]
            best_target_name = normalized_entity_lookup[target_normalized][0][0]
            best_target_match = 1.0
        
        # If not found, try fuzzy matching
        if not source_id:
            for name, eid in entity_name_to_id.items():
                source_sim = fuzzy_match(source_normalized, normalize_entity_name(name).lower())
                if source_sim >= FUZZY_MATCH_THRESHOLD and source_sim > best_source_match:
                    source_id = eid
                    best_source_match = source_sim
                    best_source_name = name
        
        if not target_id:
            for name, eid in entity_name_to_id.items():
                target_sim = fuzzy_match(target_normalized, normalize_entity_name(name).lower())
                if target_sim >= FUZZY_MATCH_THRESHOLD and target_sim > best_target_match:
                    target_id = eid
                    best_target_match = target_sim
                    best_target_name = name
        
        if source_id and target_id:
            edge_id = f"e{idx + 1}"
            edge = {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "relationship": rel.get("type", ""),
                "confidence": rel.get("confidence", 0.0),
                "description": rel.get("description", ""),
                "document_source": document_source
            }
            edges.append(edge)
            
            # Add to source mapping
            source_mapping[edge_id] = rel.get("description", rel.get("source_chunk", ""))
        else:
            # Track unmatched relationships for debugging
            unmatched_relationships.append({
                "relationship": rel,
                "source_found": source_id is not None,
                "target_found": target_id is not None,
                "best_source_match": best_source_match,
                "best_target_match": best_target_match,
                "best_source_name": best_source_name,
                "best_target_name": best_target_name
            })
    
    # Add entity source mappings
    for node in nodes:
        # Find corresponding entity
        for entity in unique_entities:
            if entity.get("name", "").strip() == node["label"]:
                source_mapping[node["id"]] = entity.get("context", entity.get("source_chunk", ""))
                break
    
    graph = {
        "nodes": nodes,
        "edges": edges,
        "source_mapping": source_mapping,
        "metadata": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_unmatched_relationships": len(unmatched_relationships),
            "document_source": document_source
        }
    }
    
    # Print analysis
    print(f"\n=== Graph Building Analysis ===")
    print(f"Total entities: {len(unique_entities)}")
    print(f"Total relationships: {len(validated_relationships)}")
    print(f"Successfully connected edges: {len(edges)}")
    print(f"Unmatched relationships: {len(unmatched_relationships)}")
    
    if unmatched_relationships:
        print("\nUnmatched relationships details:")
        for umr in unmatched_relationships[:5]:  # Show first 5
            rel = umr["relationship"]
            print(f"  - Source: '{rel.get('source')}' (found: {umr['source_found']}, match: {umr['best_source_match']:.2f})")
            print(f"    Target: '{rel.get('target')}' (found: {umr['target_found']}, match: {umr['best_target_match']:.2f})")
            if umr['best_source_name']:
                print(f"    Best source match: '{umr['best_source_name']}'")
            if umr['best_target_name']:
                print(f"    Best target match: '{umr['best_target_name']}'")
    
    # Print connection statistics
    if edges:
        print(f"\n=== Edge Connection Statistics ===")
        node_connections = defaultdict(int)
        for edge in edges:
            node_connections[edge["source"]] += 1
            node_connections[edge["target"]] += 1
        
        print(f"Nodes with connections: {len(node_connections)}/{len(nodes)}")
        print(f"Average connections per node: {sum(node_connections.values()) / len(nodes) if nodes else 0:.2f}")
        print(f"Most connected node: {max(node_connections.items(), key=lambda x: x[1]) if node_connections else 'N/A'}")
    
    return graph


def merge_graphs(graphs: List[Dict]) -> Dict:
    """Merge multiple knowledge graphs into one."""
    if not graphs:
        return {"nodes": [], "edges": [], "source_mapping": {}, "metadata": {}}
    
    if len(graphs) == 1:
        return graphs[0]
    
    # Collect all nodes and edges
    all_nodes = []
    all_edges = []
    all_source_mappings = {}
    document_sources = set()
    
    for graph in graphs:
        all_nodes.extend(graph.get("nodes", []))
        all_edges.extend(graph.get("edges", []))
        all_source_mappings.update(graph.get("source_mapping", {}))
        doc_source = graph.get("metadata", {}).get("document_source", "")
        if doc_source:
            document_sources.add(doc_source)
    
    # Deduplicate nodes by label and type
    unique_nodes = []
    node_label_to_id = {}
    next_node_id = 1
    
    for node in all_nodes:
        label = node.get("label", "").strip()
        node_type = node.get("type", "")
        
        # Check for existing similar node
        existing_id = None
        for existing_label, existing_node_id in node_label_to_id.items():
            if (fuzzy_match(label.lower(), existing_label.lower()) >= FUZZY_MATCH_THRESHOLD and
                node_type == next((n for n in unique_nodes if n["id"] == existing_node_id), {}).get("type", "")):
                existing_id = existing_node_id
                # Update confidence if higher
                for un in unique_nodes:
                    if un["id"] == existing_id:
                        if node.get("confidence", 0) > un.get("confidence", 0):
                            un["confidence"] = node.get("confidence", 0)
                        break
                break
        
        if existing_id is None:
            new_id = f"n{next_node_id}"
            next_node_id += 1
            node_label_to_id[label] = new_id
            node["id"] = new_id
            unique_nodes.append(node)
    
    # Update edges with new node IDs
    next_edge_id = 1
    merged_edges = []
    
    for edge in all_edges:
        source_label = None
        target_label = None
        
        # Find source and target labels from original nodes
        for node in all_nodes:
            if node.get("id") == edge.get("source"):
                source_label = node.get("label", "")
            if node.get("id") == edge.get("target"):
                target_label = node.get("label", "")
        
        # Find new IDs
        source_id = None
        target_id = None
        
        for label, node_id in node_label_to_id.items():
            if source_label and fuzzy_match(source_label.lower(), label.lower()) >= FUZZY_MATCH_THRESHOLD:
                source_id = node_id
            if target_label and fuzzy_match(target_label.lower(), label.lower()) >= FUZZY_MATCH_THRESHOLD:
                target_id = node_id
        
        if source_id and target_id:
            # Check for duplicate edges
            is_duplicate = False
            for me in merged_edges:
                if (me.get("source") == source_id and 
                    me.get("target") == target_id and 
                    me.get("relationship") == edge.get("relationship")):
                    is_duplicate = True
                    # Update confidence if higher
                    if edge.get("confidence", 0) > me.get("confidence", 0):
                        me["confidence"] = edge.get("confidence", 0)
                    break
            
            if not is_duplicate:
                edge["id"] = f"e{next_edge_id}"
                next_edge_id += 1
                edge["source"] = source_id
                edge["target"] = target_id
                merged_edges.append(edge)
    
    # Update source mappings with new IDs
    merged_source_mapping = {}
    for old_id, mapping_text in all_source_mappings.items():
        # Try to find corresponding new ID
        if old_id.startswith("n"):
            # Node mapping
            for node in all_nodes:
                if node.get("id") == old_id:
                    label = node.get("label", "")
                    new_id = node_label_to_id.get(label)
                    if new_id:
                        merged_source_mapping[new_id] = mapping_text
                    break
        elif old_id.startswith("e"):
            # Edge mapping - keep original edge IDs if possible
            merged_source_mapping[old_id] = mapping_text
    
    merged_graph = {
        "nodes": unique_nodes,
        "edges": merged_edges,
        "source_mapping": merged_source_mapping,
        "metadata": {
            "num_nodes": len(unique_nodes),
            "num_edges": len(merged_edges),
            "document_sources": list(document_sources),
            "merged_from": len(graphs)
        }
    }
    
    return merged_graph


def analyze_graph_connections(graph: Dict) -> Dict:
    """Analyze graph connections and return detailed statistics."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    analysis = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "nodes_by_type": defaultdict(int),
        "edges_by_relationship": defaultdict(int),
        "node_connections": defaultdict(int),
        "isolated_nodes": [],
        "most_connected_nodes": [],
        "relationship_distribution": defaultdict(int)
    }
    
    # Count nodes by type
    for node in nodes:
        node_type = node.get("type", "Unknown")
        analysis["nodes_by_type"][node_type] += 1
    
    # Count edges by relationship type
    for edge in edges:
        rel_type = edge.get("relationship", "unknown")
        analysis["edges_by_relationship"][rel_type] += 1
        analysis["relationship_distribution"][rel_type] += 1
        
        # Count connections per node
        source_id = edge.get("source")
        target_id = edge.get("target")
        if source_id:
            analysis["node_connections"][source_id] += 1
        if target_id:
            analysis["node_connections"][target_id] += 1
    
    # Find isolated nodes (no connections)
    node_ids = {node["id"] for node in nodes}
    connected_node_ids = set(analysis["node_connections"].keys())
    analysis["isolated_nodes"] = [
        node for node in nodes 
        if node["id"] not in connected_node_ids
    ]
    
    # Find most connected nodes
    if analysis["node_connections"]:
        max_connections = max(analysis["node_connections"].values())
        analysis["most_connected_nodes"] = [
            {"node_id": node_id, "connections": count, "label": next(
                (n.get("label", node_id) for n in nodes if n["id"] == node_id), node_id
            )}
            for node_id, count in analysis["node_connections"].items()
            if count == max_connections
        ]
    
    return analysis


def print_graph_analysis(graph: Dict, output_file: str = None):
    """Print detailed graph analysis and optionally save to file."""
    analysis = analyze_graph_connections(graph)
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("KNOWLEDGE GRAPH ANALYSIS")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Basic statistics
    output_lines.append(f"Total Nodes: {analysis['total_nodes']}")
    output_lines.append(f"Total Edges: {analysis['total_edges']}")
    output_lines.append(f"Isolated Nodes: {len(analysis['isolated_nodes'])}")
    output_lines.append("")
    
    # Nodes by type
    output_lines.append("Nodes by Type:")
    for node_type, count in sorted(analysis["nodes_by_type"].items()):
        output_lines.append(f"  {node_type}: {count}")
    output_lines.append("")
    
    # Edges by relationship type
    output_lines.append("Edges by Relationship Type:")
    for rel_type, count in sorted(analysis["edges_by_relationship"].items()):
        output_lines.append(f"  {rel_type}: {count}")
    output_lines.append("")
    
    # Most connected nodes
    if analysis["most_connected_nodes"]:
        output_lines.append("Most Connected Nodes:")
        for node_info in analysis["most_connected_nodes"][:10]:  # Top 10
            output_lines.append(f"  {node_info['label']} ({node_info['node_id']}): {node_info['connections']} connections")
        output_lines.append("")
    
    # Isolated nodes
    if analysis["isolated_nodes"]:
        output_lines.append(f"Isolated Nodes ({len(analysis['isolated_nodes'])}):")
        for node in analysis["isolated_nodes"][:10]:  # First 10
            output_lines.append(f"  {node.get('label', node['id'])} ({node.get('type', 'Unknown')})")
        if len(analysis["isolated_nodes"]) > 10:
            output_lines.append(f"  ... and {len(analysis['isolated_nodes']) - 10} more")
        output_lines.append("")
    
    # Edge details
    output_lines.append("Edge Details:")
    edges = graph.get("edges", [])
    for edge in edges[:20]:  # First 20 edges
        source_node = next((n for n in graph.get("nodes", []) if n["id"] == edge.get("source")), None)
        target_node = next((n for n in graph.get("nodes", []) if n["id"] == edge.get("target")), None)
        source_label = source_node.get("label", edge.get("source")) if source_node else edge.get("source")
        target_label = target_node.get("label", edge.get("target")) if target_node else edge.get("target")
        rel_type = edge.get("relationship", "unknown")
        confidence = edge.get("confidence", 0.0)
        output_lines.append(f"  {source_label} --[{rel_type}]--> {target_label} (confidence: {confidence:.2f})")
    if len(edges) > 20:
        output_lines.append(f"  ... and {len(edges) - 20} more edges")
    output_lines.append("")
    
    output_lines.append("=" * 80)
    
    # Print to console
    analysis_text = "\n".join(output_lines)
    print(analysis_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        print(f"\nGraph analysis saved to {output_file}")


def save_graph(graph: Dict, output_path: str):
    """Save knowledge graph to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

