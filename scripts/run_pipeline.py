import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR
from document_processors import load_document, load_multiple_documents
from text_preprocessing import preprocess_text, save_preprocessing_output
from llm_orchestration import create_extraction_graph
from knowledge_graph import build_graph, save_graph, merge_graphs, print_graph_analysis
from visualizer import visualize_graph, export_graph_svg


def run_pipeline(document_path: str, approach: str = "zero_shot", 
                output_dir: str = OUTPUTS_DIR, visualize: bool = True):
    """Run the complete knowledge graph extraction pipeline."""
    print(f"Document: {document_path}")
    print(f"Approach: {approach}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load document(s)
    document_path = Path(document_path)
    
    if document_path.is_file():
        documents = [(load_document(str(document_path)))]
    elif document_path.is_dir():
        # Load all PDF and TXT files from directory
        pdf_files = list(document_path.glob("*.pdf"))
        txt_files = list(document_path.glob("*.txt"))
        all_files = pdf_files + txt_files
        
        if not all_files:
            print(f"No PDF or TXT files found in {document_path}")
            return
        
        documents = load_multiple_documents([str(f) for f in all_files])
    else:
        print(f"Error: {document_path} is not a valid file or directory")
        return
    
    if not documents:
        print("No documents loaded")
        return
    
    print(f"Loaded {len(documents)} document(s)")
    
    # Process each document
    graphs = []
    extraction_graph = create_extraction_graph()
    
    for idx, (text, metadata) in enumerate(documents):
        print(f"\nProcessing document {idx + 1}/{len(documents)}: {metadata.get('source', 'unknown')}")
        
        # Preprocess text
        cleaned_text, chunks = preprocess_text(text)
        print(f"Text chunks: {len(chunks)}")
        
        # Save preprocessing outputs
        doc_name = Path(metadata.get("source", f"doc_{idx + 1}")).stem
        save_preprocessing_output(cleaned_text, chunks, output_dir, doc_name)
        
        # Initialize state
        initial_state = {
            "document_id": f"doc_{idx + 1}",
            "document_content": cleaned_text,
            "document_source": metadata.get("source", ""),
            "chunks": chunks,
            "entity_extractions": [],
            "relationship_extractions": [],
            "validated_entities": [],
            "validated_relationships": [],
            "final_graph": {},
            "approach": approach
        }
        
        # Run extraction graph
        print("Running extraction workflow...")
        try:
            final_state = extraction_graph.invoke(initial_state)
        except Exception as e:
            print(f"Error in extraction workflow: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            continue
        
        # Get validated entities and relationships
        if approach == "few_shot_critique":
            entities = final_state.get("validated_entities", final_state.get("entity_extractions", []))
            relationships = final_state.get("validated_relationships", 
                                          final_state.get("relationship_extractions", []))
        else:
            entities = final_state.get("entity_extractions", [])
            relationships = final_state.get("relationship_extractions", [])
        
        print(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        
        # Build knowledge graph
        graph = build_graph(entities, relationships, metadata.get("source", ""))
        graphs.append(graph)
        
        # Save individual graph
        doc_name = Path(metadata.get("source", f"doc_{idx + 1}")).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"{doc_name}_{approach}_{timestamp}.json")
        save_graph(graph, output_file)
        print(f"Saved graph to {output_file}")
        
        # Print and save graph analysis
        analysis_file = output_file.replace(".json", "_analysis.txt")
        print_graph_analysis(graph, analysis_file)
        
        # Visualize individual graph
        if visualize and graph.get("nodes"):
            viz_file = output_file.replace(".json", ".png")
            visualize_graph(graph, viz_file)
    
    # Merge graphs if multiple documents
    if len(graphs) > 1:
        print(f"\nMerging {len(graphs)} graphs...")
        merged_graph = merge_graphs(graphs)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        merged_output = os.path.join(output_dir, 
                                    f"merged_{approach}_{timestamp}.json")
        save_graph(merged_graph, merged_output)
        print(f"Saved merged graph to {merged_output}")
        
        # Print and save merged graph analysis
        merged_analysis_file = merged_output.replace(".json", "_analysis.txt")
        print_graph_analysis(merged_graph, merged_analysis_file)
        
        if visualize and merged_graph.get("nodes"):
            viz_file = merged_output.replace(".json", ".png")
            visualize_graph(merged_graph, viz_file)
    
    print("\nPipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Pipeline")
    parser.add_argument("--document", "-d", required=True,
                       help="Path to document file or directory")
    parser.add_argument("--approach", "-a", default="zero_shot",
                       choices=["zero_shot", "few_shot_critique"],
                       help="Extraction approach")
    parser.add_argument("--output", "-o", default=OUTPUTS_DIR,
                       help="Output directory")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    run_pipeline(
        document_path=args.document,
        approach=args.approach,
        output_dir=args.output,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()

