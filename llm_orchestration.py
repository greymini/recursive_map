import json
import re
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import google.generativeai as genai

from config import (
    GOOGLE_API_KEY, MODEL_NAME, CONFIDENCE_THRESHOLD,
    ENTITY_TYPES, RELATIONSHIP_TYPES
)


# Define the state schema
class GraphState(TypedDict):
    document_id: str
    document_content: str
    document_source: str
    chunks: List[str]
    entity_extractions: List[Dict]
    relationship_extractions: List[Dict]
    validated_entities: List[Dict]
    validated_relationships: List[Dict]
    final_graph: Dict
    approach: str  # "zero_shot" or "few_shot_critique"


def initialize_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Google Gemini LLM."""
    if not GOOGLE_API_KEY:
        error_msg = (
            "GOOGLE_API_KEY not set!"
        )
        raise ValueError(error_msg)
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY
    )


def extract_json_from_response(response: str, debug: bool = False) -> Dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not response or not isinstance(response, str):
        if debug:
            print(f"DEBUG: Empty or non-string response: {type(response)}")
        return {}
    
    original_response = response
    
    # Remove markdown code blocks if present
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()
    
    if debug:
        print(f"DEBUG: Response after cleaning (first 500 chars): {response[:500]}")
    

    json_match = None
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_match = response[start_idx:i+1]
                break
    
    if json_match:
        if debug:
            print(f"DEBUG: Found JSON match (first 200 chars): {json_match[:200]}")
        try:
            return json.loads(json_match)
        except json.JSONDecodeError as e:
            if debug:
                print(f"DEBUG: JSON decode error: {e}")
                print(f"DEBUG: Error at position {e.pos if hasattr(e, 'pos') else 'unknown'}")
                print(f"DEBUG: Error message: {e.msg if hasattr(e, 'msg') else str(e)}")
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                json_match = re.sub(r',\s*}', '}', json_match)
                json_match = re.sub(r',\s*]', ']', json_match)
                # Fix single quotes to double quotes (common LLM mistake)
                json_match = re.sub(r"'([^']*)':", r'"\1":', json_match)
                json_match = re.sub(r":\s*'([^']*)'", r': "\1"', json_match)
                return json.loads(json_match)
            except json.JSONDecodeError as e2:
                if debug:
                    print(f"DEBUG: JSON decode error after fixing: {e2}")
                    print(f"DEBUG: Problematic JSON (first 500 chars): {json_match[:500]}")
                # Return empty dict instead of raising
                pass
    
    # If no JSON found, try parsing entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        if debug:
            print(f"DEBUG: Failed to parse entire response as JSON: {e}")
            print(f"DEBUG: Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
            print(f"DEBUG: Response length: {len(response)}")
            print(f"DEBUG: Response (first 1000 chars): {response[:1000]}")
        # Try one more time with common fixes
        try:
            fixed_response = response
            # Remove trailing commas
            fixed_response = re.sub(r',\s*}', '}', fixed_response)
            fixed_response = re.sub(r',\s*]', ']', fixed_response)
            # Fix single quotes
            fixed_response = re.sub(r"'([^']*)':", r'"\1":', fixed_response)
            fixed_response = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_response)
            return json.loads(fixed_response)
        except:
            pass
        return {}


def extract_entities_zero_shot(state: GraphState) -> GraphState:
    """Extract entities using zero-shot approach."""
    llm = initialize_llm()
    chunks = state["chunks"]
    entities = []
    
    entity_types_str = ", ".join(ENTITY_TYPES)
    
    prompt_template = """Extract all entities from the following text.

For each entity, identify:
- name: Entity name
- type: One of [{entity_types}]
- context: Brief description or context where it appears
- confidence: Your confidence (0.0-1.0)

Return ONLY valid JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "context": "...", "confidence": 0.0}}
  ]
}}

TEXT:
{text_chunk}"""

    for chunk in chunks:
        prompt = prompt_template.format(entity_types=entity_types_str, text_chunk=chunk)
        
        try:
            response = llm.invoke(prompt)
            
            # Handle different response types from LangChain
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            elif isinstance(response, str):
                content = response
            else:
                # Try to get string representation
                content = str(response)
            
            if not content:
                print("Warning: Empty response from LLM")
                continue
            
            # Debug mode for first chunk if no entities found
            debug_mode = len(entities) == 0 and chunk == chunks[0]
            result = extract_json_from_response(content, debug=debug_mode)
            
            if "entities" in result and isinstance(result["entities"], list):
                for entity in result["entities"]:
                    if isinstance(entity, dict) and "name" in entity:
                        entity["source_chunk"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
                        entities.append(entity)
            else:
                print(f"Warning: No valid entities found in response. Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                if debug_mode:
                    print(f"DEBUG: Full response content (first 1000 chars): {content[:1000]}")
        except Exception as e:
            print(f"Error extracting entities from chunk: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    state["entity_extractions"] = entities
    return state


def extract_entities_few_shot(state: GraphState) -> GraphState:
    """Extract entities using few-shot approach with examples."""
    llm = initialize_llm()
    chunks = state["chunks"]
    entities = []
    
    entity_types_str = ", ".join(ENTITY_TYPES)
    
    prompt_template = """Extract all entities from the following text with examples from diverse domains:

EXAMPLE 1 (Business/Technology):
Text: "Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California."
Output: {{"entities": [
  {{"name": "Apple Inc.", "type": "Organization", "context": "Technology company", "confidence": 0.98}},
  {{"name": "Steve Jobs", "type": "Person", "context": "Founder", "confidence": 0.95}},
  {{"name": "Cupertino, California", "type": "Location", "context": "Company location", "confidence": 0.92}}
]}}

EXAMPLE 2 (Literature/Arts):
Text: "Jane Austen wrote Pride and Prejudice, which was published in London in 1813."
Output: {{"entities": [
  {{"name": "Jane Austen", "type": "Person", "context": "Author", "confidence": 0.98}},
  {{"name": "Pride and Prejudice", "type": "Product", "context": "Novel", "confidence": 0.96}},
  {{"name": "London", "type": "Location", "context": "Publication location", "confidence": 0.94}}
]}}

EXAMPLE 3 (Science/History):
Text: "The discovery of penicillin by Alexander Fleming at St. Mary's Hospital revolutionized medicine in 1928."
Output: {{"entities": [
  {{"name": "penicillin", "type": "Product", "context": "Antibiotic drug", "confidence": 0.97}},
  {{"name": "Alexander Fleming", "type": "Person", "context": "Scientist and discoverer", "confidence": 0.96}},
  {{"name": "St. Mary's Hospital", "type": "Organization", "context": "Research institution", "confidence": 0.93}},
  {{"name": "1928", "type": "Event", "context": "Discovery year", "confidence": 0.90}}
]}}

For each entity, identify:
- name: Entity name
- type: One of [{entity_types}]
- context: Brief description
- confidence: Your confidence (0.0-1.0)

Extract from this text:
{text_chunk}

Return ONLY valid JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "context": "...", "confidence": 0.0}}
  ]
}}"""

    for chunk in chunks:
        prompt = prompt_template.format(entity_types=entity_types_str, text_chunk=chunk)
        
        try:
            response = llm.invoke(prompt)
            
            # Handle different response types from LangChain
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            elif isinstance(response, str):
                content = response
            else:
                # Try to get string representation
                content = str(response)
            
            if not content:
                print("Warning: Empty response from LLM")
                continue
            
            # Debug mode for first chunk if no entities found
            debug_mode = len(entities) == 0 and chunk == chunks[0]
            result = extract_json_from_response(content, debug=debug_mode)
            
            if "entities" in result and isinstance(result["entities"], list):
                for entity in result["entities"]:
                    if isinstance(entity, dict) and "name" in entity:
                        entity["source_chunk"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
                        entities.append(entity)
            else:
                print(f"Warning: No valid entities found in response. Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                if debug_mode:
                    print(f"DEBUG: Full response content (first 1000 chars): {content[:1000]}")
        except Exception as e:
            print(f"Error extracting entities from chunk: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    state["entity_extractions"] = entities
    return state


def extract_relationships_zero_shot(state: GraphState) -> GraphState:
    """Extract relationships using zero-shot approach."""
    llm = initialize_llm()
    chunks = state["chunks"]
    relationships = []
    
    relationship_types_str = ", ".join(RELATIONSHIP_TYPES)
    
    prompt_template = """Extract relationships between entities from the following text.

For each relationship, identify:
- source: Source entity name
- target: Target entity name
- type: One of [{relationship_types}]
- description: Sentence describing the relationship
- confidence: Your confidence (0.0-1.0)

Return ONLY valid JSON:
{{
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "...", "confidence": 0.0}}
  ]
}}

TEXT:
{text_chunk}"""

    for chunk in chunks:
        prompt = prompt_template.format(relationship_types=relationship_types_str, text_chunk=chunk)
        
        try:
            response = llm.invoke(prompt)
            
            # Handle different response types from LangChain
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            elif isinstance(response, str):
                content = response
            else:
                # Try to get string representation
                content = str(response)
            
            if not content:
                print("Warning: Empty response from LLM")
                continue
            
            # Debug mode for first chunk if no relationships found
            debug_mode = len(relationships) == 0 and chunk == chunks[0]
            result = extract_json_from_response(content, debug=debug_mode)
            
            if "relationships" in result and isinstance(result["relationships"], list):
                for rel in result["relationships"]:
                    if isinstance(rel, dict) and "source" in rel and "target" in rel:
                        rel["source_chunk"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
                        relationships.append(rel)
            else:
                print(f"Warning: No valid relationships found in response. Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                if debug_mode:
                    print(f"DEBUG: Full response content (first 1000 chars): {content[:1000]}")
        except Exception as e:
            print(f"Error extracting relationships from chunk: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    state["relationship_extractions"] = relationships
    return state


def extract_relationships_few_shot(state: GraphState) -> GraphState:
    """Extract relationships using few-shot approach with examples."""
    llm = initialize_llm()
    chunks = state["chunks"]
    relationships = []
    
    relationship_types_str = ", ".join(RELATIONSHIP_TYPES)
    
    prompt_template = """Extract relationships between entities with examples from diverse domains:

EXAMPLE 1 (Business/Employment):
Text: "John Smith works at Apple Inc. as a software engineer."
Output: {{"relationships": [{{"source": "John Smith", "target": "Apple Inc.", "type": "works_at", "description": "John Smith works at Apple Inc. as a software engineer", "confidence": 0.95}}]}}

EXAMPLE 2 (Literature/Publishing):
Text: "HarperCollins published The Great Gatsby, which was written by F. Scott Fitzgerald in 1925."
Output: {{"relationships": [
  {{"source": "F. Scott Fitzgerald", "target": "The Great Gatsby", "type": "develops", "description": "F. Scott Fitzgerald wrote The Great Gatsby", "confidence": 0.96}},
  {{"source": "HarperCollins", "target": "The Great Gatsby", "type": "develops", "description": "HarperCollins published The Great Gatsby", "confidence": 0.92}}
]}}

EXAMPLE 3 (History/Geography):
Text: "Napoleon Bonaparte was born in Corsica, France, and led the French Empire during the Napoleonic Wars."
Output: {{"relationships": [
  {{"source": "Napoleon Bonaparte", "target": "Corsica, France", "type": "located_in", "description": "Napoleon Bonaparte was born in Corsica, France", "confidence": 0.94}},
  {{"source": "Napoleon Bonaparte", "target": "French Empire", "type": "works_at", "description": "Napoleon Bonaparte led the French Empire", "confidence": 0.93}},
  {{"source": "Napoleon Bonaparte", "target": "Napoleonic Wars", "type": "uses", "description": "Napoleon Bonaparte led during the Napoleonic Wars", "confidence": 0.90}}
]}}

EXAMPLE 4 (Science/Collaboration):
Text: "Marie Curie collaborated with her husband Pierre Curie at the University of Paris to discover radium."
Output: {{"relationships": [
  {{"source": "Marie Curie", "target": "Pierre Curie", "type": "collaboratesWith", "description": "Marie Curie collaborated with her husband Pierre Curie", "confidence": 0.95}},
  {{"source": "Marie Curie", "target": "University of Paris", "type": "works_at", "description": "Marie Curie worked at the University of Paris", "confidence": 0.93}},
  {{"source": "Marie Curie", "target": "radium", "type": "develops", "description": "Marie Curie discovered radium", "confidence": 0.94}}
]}}

For each relationship, identify:
- source: Source entity name
- target: Target entity name
- type: One of [{relationship_types}]
- description: Sentence describing the relationship
- confidence: Your confidence (0.0-1.0)

Extract from this text:
{text_chunk}

Return ONLY valid JSON:
{{
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "...", "confidence": 0.0}}
  ]
}}"""

    for chunk in chunks:
        prompt = prompt_template.format(relationship_types=relationship_types_str, text_chunk=chunk)
        
        try:
            response = llm.invoke(prompt)
            
            # Handle different response types from LangChain
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            elif isinstance(response, str):
                content = response
            else:
                # Try to get string representation
                content = str(response)
            
            if not content:
                print("Warning: Empty response from LLM")
                continue
            
            # Debug mode for first chunk if no relationships found
            debug_mode = len(relationships) == 0 and chunk == chunks[0]
            result = extract_json_from_response(content, debug=debug_mode)
            
            if "relationships" in result and isinstance(result["relationships"], list):
                for rel in result["relationships"]:
                    if isinstance(rel, dict) and "source" in rel and "target" in rel:
                        rel["source_chunk"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
                        relationships.append(rel)
            else:
                print(f"Warning: No valid relationships found in response. Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                if debug_mode:
                    print(f"DEBUG: Full response content (first 1000 chars): {content[:1000]}")
        except Exception as e:
            print(f"Error extracting relationships from chunk: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    state["relationship_extractions"] = relationships
    return state


def self_critique(state: GraphState) -> GraphState:
    """LLM validates and critiques extracted entities and relationships."""
    llm = initialize_llm()
    
    entities = state.get("entity_extractions", [])
    relationships = state.get("relationship_extractions", [])
    
    # Validate and clean entities/relationships before serialization
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []
    
    # Filter out any invalid entries
    valid_entities = []
    for e in entities:
        if isinstance(e, dict) and "name" in e:
            valid_entities.append(e)
    
    valid_relationships = []
    for r in relationships:
        if isinstance(r, dict) and "source" in r and "target" in r:
            valid_relationships.append(r)
    
    entities = valid_entities
    relationships = valid_relationships
    
    # Serialize to JSON safely
    try:
        entities_json = json.dumps(entities, indent=2, ensure_ascii=False)
        relationships_json = json.dumps(relationships, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        print(f"Error serializing entities/relationships for self-critique: {e}")
        # Fallback to original extractions
        state["validated_entities"] = entities
        state["validated_relationships"] = relationships
        return state
    
    prompt = f"""Review these extracted entities and relationships for issues:
- Duplicates or near-duplicates
- Invalid entity types (must be one of: {', '.join(ENTITY_TYPES)})
- Relationships where source/target don't exist in entities
- Items with low confidence that should be removed
- Contradictions with source text

ENTITIES:
{entities_json}

RELATIONSHIPS:
{relationships_json}

Return corrected JSON with only valid extractions:
{{
  "entities": [...],
  "relationships": [...]
}}"""

    try:
        response = llm.invoke(prompt)
        
        # Handle different response types from LangChain
        if hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'text'):
            content = response.text
        elif isinstance(response, str):
            content = response
        else:
            # Try to get string representation
            content = str(response)
        
        if not content:
            print("Warning: Empty response from LLM in self-critique")
            state["validated_entities"] = entities
            state["validated_relationships"] = relationships
            return state
        
        result = extract_json_from_response(content, debug=True)
        
        if "entities" in result and isinstance(result["entities"], list):
            state["validated_entities"] = result["entities"]
        else:
            print("Warning: No valid entities in self-critique result, using original")
            state["validated_entities"] = entities
        
        if "relationships" in result and isinstance(result["relationships"], list):
            state["validated_relationships"] = result["relationships"]
        else:
            print("Warning: No valid relationships in self-critique result, using original")
            state["validated_relationships"] = relationships
    except Exception as e:
        print(f"Error in self-critique: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # Fallback to original extractions
        state["validated_entities"] = entities
        state["validated_relationships"] = relationships
    
    return state


def should_use_few_shot(state: GraphState) -> str:
    """Conditional edge: determine if few-shot approach should be used."""
    return "few_shot" if state["approach"] == "few_shot_critique" else "zero_shot"


def create_extraction_graph() -> StateGraph:
    """Create the LangGraph workflow for knowledge graph extraction."""
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("extract_entities_zero_shot", extract_entities_zero_shot)
    workflow.add_node("extract_entities_few_shot", extract_entities_few_shot)
    workflow.add_node("extract_relationships_zero_shot", extract_relationships_zero_shot)
    workflow.add_node("extract_relationships_few_shot", extract_relationships_few_shot)
    workflow.add_node("self_critique", self_critique)
    
    # Set entry point
    workflow.set_entry_point("extract_entities_zero_shot")
    
    # Add conditional edges for entity extraction
    workflow.add_conditional_edges(
        "extract_entities_zero_shot",
        should_use_few_shot,
        {
            "zero_shot": "extract_relationships_zero_shot",
            "few_shot": "extract_entities_few_shot"
        }
    )
    
    workflow.add_edge("extract_entities_few_shot", "extract_relationships_few_shot")
    
    # Add conditional edges for relationship extraction
    workflow.add_conditional_edges(
        "extract_relationships_zero_shot",
        should_use_few_shot,
        {
            "zero_shot": END,
            "few_shot": "self_critique"
        }
    )
    
    workflow.add_edge("extract_relationships_few_shot", "self_critique")
    workflow.add_edge("self_critique", END)
    
    return workflow.compile()

