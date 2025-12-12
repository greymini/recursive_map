"""
Streamlit UI for Knowledge Graph Extraction
"""
import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, CONFIDENCE_THRESHOLD
from document_processors import load_document, load_multiple_documents
from text_preprocessing import preprocess_text
from llm_orchestration import create_extraction_graph
from knowledge_graph import build_graph, save_graph, merge_graphs
from visualizer import visualize_graph
from evaluator import evaluate_extraction, load_reference_annotations


st.set_page_config(
    page_title="Knowledge Graph Extraction",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Knowledge Graph Extraction Pipeline")
st.markdown("Extract entities and relationships from documents using AI")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    approach = st.selectbox(
        "Extraction Approach",
        ["zero_shot", "few_shot_critique"],
        help="Zero-shot: Direct extraction. Few-shot: With examples and self-critique"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence score for entities/relationships"
    )
    
    visualize = st.checkbox("Generate Visualization", value=True)
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload PDF or TXT documents
    2. Select extraction approach
    3. Click "Extract Knowledge Graph"
    4. View results and export
    """)


# Main content area
tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "📊 Results", "📈 Evaluation"])

with tab1:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        
        if st.button("🚀 Extract Knowledge Graph", type="primary"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, 
                                                    suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        temp_files.append(tmp.name)
                
                try:
                    # Load documents
                    status_text.text("Loading documents...")
                    progress_bar.progress(10)
                    
                    documents = load_multiple_documents(temp_files)
                    
                    if not documents:
                        st.error("Failed to load documents")
                        st.stop()
                    
                    status_text.text(f"Processing {len(documents)} document(s)...")
                    progress_bar.progress(20)
                    
                    # Preprocess
                    all_chunks = []
                    all_metadata = []
                    for text, metadata in documents:
                        cleaned_text, chunks = preprocess_text(text)
                        all_chunks.extend(chunks)
                        all_metadata.append(metadata)
                    
                    progress_bar.progress(30)
                    
                    # Initialize extraction graph
                    extraction_graph = create_extraction_graph()
                    
                    # Process documents
                    graphs = []
                    for idx, (text, metadata) in enumerate(documents):
                        status_text.text(f"Extracting from document {idx + 1}/{len(documents)}...")
                        progress_bar.progress(30 + (idx + 1) * 30 // len(documents))
                        
                        cleaned_text, chunks = preprocess_text(text)
                        
                        initial_state = {
                            "document_id": f"doc_{idx + 1}",
                            "document_content": cleaned_text,
                            "document_source": metadata.get("source", uploaded_files[idx].name),
                            "chunks": chunks,
                            "entity_extractions": [],
                            "relationship_extractions": [],
                            "validated_entities": [],
                            "validated_relationships": [],
                            "final_graph": {},
                            "approach": approach
                        }
                        
                        try:
                            final_state = extraction_graph.invoke(initial_state)
                            
                            if approach == "few_shot_critique":
                                entities = final_state.get("validated_entities", 
                                                          final_state.get("entity_extractions", []))
                                relationships = final_state.get("validated_relationships",
                                                              final_state.get("relationship_extractions", []))
                            else:
                                entities = final_state.get("entity_extractions", [])
                                relationships = final_state.get("relationship_extractions", [])
                            
                            graph = build_graph(entities, relationships, 
                                              metadata.get("source", uploaded_files[idx].name))
                            graphs.append(graph)
                            
                        except Exception as e:
                            st.error(f"Error processing document {idx + 1}: {str(e)}")
                            continue
                    
                    progress_bar.progress(90)
                    
                    # Merge graphs if multiple documents
                    if len(graphs) > 1:
                        status_text.text("Merging graphs...")
                        final_graph = merge_graphs(graphs)
                    else:
                        final_graph = graphs[0] if graphs else {}
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Store in session state
                    st.session_state['knowledge_graph'] = final_graph
                    st.session_state['graphs'] = graphs
                    st.session_state['approach'] = approach
                    
                    st.success("Knowledge graph extraction completed!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    # Clean up temp files
                    for tmp_file in temp_files:
                        try:
                            os.unlink(tmp_file)
                        except:
                            pass


with tab2:
    st.header("Extraction Results")
    
    if 'knowledge_graph' not in st.session_state:
        st.info("Please upload documents and extract knowledge graph first.")
    else:
        graph = st.session_state['knowledge_graph']
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", len(graph.get("nodes", [])))
        with col2:
            st.metric("Edges", len(graph.get("edges", [])))
        with col3:
            st.metric("Approach", st.session_state.get('approach', 'unknown'))
        with col4:
            avg_confidence = sum(n.get("confidence", 0) for n in graph.get("nodes", [])) / len(graph.get("nodes", [])) if graph.get("nodes") else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Visualization
        if visualize and graph.get("nodes"):
            st.subheader("Graph Visualization")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    visualize_graph(graph, tmp.name)
                    st.image(tmp.name)
                    os.unlink(tmp.name)
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
        
        # Entities table
        st.subheader("Entities")
        if graph.get("nodes"):
            entities_data = []
            for node in graph["nodes"]:
                entities_data.append({
                    "ID": node["id"],
                    "Label": node["label"],
                    "Type": node["type"],
                    "Confidence": f"{node.get('confidence', 0):.2f}",
                    "Context": node.get("context", "")[:100] + "..." if len(node.get("context", "")) > 100 else node.get("context", "")
                })
            
            df_entities = pd.DataFrame(entities_data)
            st.dataframe(df_entities, use_container_width=True, hide_index=True)
        else:
            st.info("No entities extracted")
        
        # Relationships table
        st.subheader("Relationships")
        if graph.get("edges"):
            relationships_data = []
            for edge in graph["edges"]:
                source_label = next((n["label"] for n in graph["nodes"] 
                                   if n["id"] == edge["source"]), edge["source"])
                target_label = next((n["label"] for n in graph["nodes"] 
                                   if n["id"] == edge["target"]), edge["target"])
                
                relationships_data.append({
                    "ID": edge["id"],
                    "Source": source_label,
                    "Target": target_label,
                    "Relationship": edge["relationship"],
                    "Confidence": f"{edge.get('confidence', 0):.2f}",
                    "Description": edge.get("description", "")[:100] + "..." if len(edge.get("description", "")) > 100 else edge.get("description", "")
                })
            
            df_relationships = pd.DataFrame(relationships_data)
            st.dataframe(df_relationships, use_container_width=True, hide_index=True)
        else:
            st.info("No relationships extracted")
        
        # Source mapping
        with st.expander("Source Text Mapping"):
            if graph.get("source_mapping"):
                for item_id, text in list(graph["source_mapping"].items())[:10]:  # Show first 10
                    st.text(f"{item_id}: {text[:200]}...")
            else:
                st.info("No source mapping available")
        
        # Export options
        st.subheader("Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_str = json.dumps(graph, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            if graph.get("nodes"):
                entities_df = pd.DataFrame([{
                    "ID": n["id"],
                    "Label": n["label"],
                    "Type": n["type"],
                    "Confidence": n.get("confidence", 0)
                } for n in graph["nodes"]])
                csv_entities = entities_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Entities CSV",
                    data=csv_entities,
                    file_name=f"entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if graph.get("edges"):
                relationships_df = pd.DataFrame([{
                    "ID": e["id"],
                    "Source": next((n["label"] for n in graph["nodes"] if n["id"] == e["source"]), e["source"]),
                    "Target": next((n["label"] for n in graph["nodes"] if n["id"] == e["target"]), e["target"]),
                    "Relationship": e["relationship"],
                    "Confidence": e.get("confidence", 0)
                } for e in graph["edges"]])
                csv_relationships = relationships_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Relationships CSV",
                    data=csv_relationships,
                    file_name=f"relationships_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


with tab3:
    st.header("Evaluation")
    
    st.info("Upload reference annotations to evaluate extraction quality")
    
    reference_file = st.file_uploader(
        "Upload Reference Annotations (JSON)",
        type=['json']
    )
    
    if reference_file and 'knowledge_graph' in st.session_state:
        try:
            reference_data = json.load(reference_file)
            graph = st.session_state['knowledge_graph']
            
            from evaluator import evaluate_extraction
            metrics = evaluate_extraction(graph, reference_data)
            
            st.subheader("Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Entity Precision", f"{metrics['entities']['precision']:.3f}")
                st.metric("Entity Recall", f"{metrics['entities']['recall']:.3f}")
                st.metric("Entity F1", f"{metrics['entities']['f1']:.3f}")
            
            with col2:
                st.metric("Relationship Precision", f"{metrics['relationships']['precision']:.3f}")
                st.metric("Relationship Recall", f"{metrics['relationships']['recall']:.3f}")
                st.metric("Relationship F1", f"{metrics['relationships']['f1']:.3f}")
            
            with col3:
                st.metric("Overall Precision", f"{metrics['overall']['precision']:.3f}")
                st.metric("Overall Recall", f"{metrics['overall']['recall']:.3f}")
                st.metric("Overall F1", f"{metrics['overall']['f1']:.3f}")
            
            # Detailed metrics
            with st.expander("Detailed Metrics"):
                st.json(metrics)
                
        except Exception as e:
            st.error(f"Error evaluating: {str(e)}")
    elif 'knowledge_graph' not in st.session_state:
        st.warning("Please extract a knowledge graph first")

