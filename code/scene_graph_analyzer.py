"""
Scene Graph Analysis for Planning Problems

This module provides functionality for analyzing planning problems using scene graphs
and evaluating solution options through GPT-4 Vision.

License: MIT (or your preferred license)
"""

import base64
import logging
import os
import re
import time
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any

import requests
from PIL import Image
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
MIN_SCENE_GRAPH_LENGTH = 20
MIN_JSON_LENGTH = 30


# ============================================================================
# Image and Encoding Utilities
# ============================================================================

def load_image(image_file: str) -> Image.Image:
    """
    Load an image from file path or URL.
    
    Args:
        image_file: Path to image file or URL
        
    Returns:
        PIL Image in RGB format
    """
    if isinstance(image_file, str) and (
        image_file.startswith('http://') or image_file.startswith('https://')
    ):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ============================================================================
# GPT Client Management
# ============================================================================

def initialize_gpt_client() -> OpenAI:
    """
    Initialize OpenAI GPT client using environment variable.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If API key is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with your OpenAI API key."
        )
    
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
    return client


def gpt_inference(
    client: OpenAI,
    image_path: str,
    query: str,
    history: Optional[List] = None
) -> str:
    """
    Make an inference using the GPT-4 Vision model.
    
    Args:
        client: OpenAI client instance
        image_path: Path to the image file
        query: Text query/prompt
        history: Conversation history (not used currently)
        
    Returns:
        Model response text
    """
    base64_image = encode_image_to_base64(image_path)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }]
        )
        response_text = response.choices[0].message.content
        time.sleep(3)  # Rate limiting
        return response_text
    except Exception as e:
        logger.error(f"Error in GPT inference: {e}")
        time.sleep(3)
        return ""


# ============================================================================
# Scene Graph Extraction Functions
# ============================================================================

def extract_scene_graph(response: str, max_attempts: int = DEFAULT_MAX_RETRIES) -> Tuple[str, str]:
    """
    Extract initial and target scene graphs from GPT response.
    
    Args:
        response: GPT model response text
        max_attempts: Maximum extraction attempts
        
    Returns:
        Tuple of (initial_scene_graph, target_scene_graph)
    """
    # Define possible variations of scene graph identifiers
    initial_variants = [
        "initial_scene_graph", "Initial_scene_graph", "INITIAL_SCENE_GRAPH",
        "initial scene graph", "Initial Scene Graph", "initialSceneGraph"
    ]
    target_variants = [
        "target_scene_graph", "Target_scene_graph", "TARGET_SCENE_GRAPH",
        "target scene graph", "Target Scene Graph", "targetSceneGraph",
        "final_scene_graph", "Final_scene_graph", "finalSceneGraph"
    ]
    
    for attempt in range(max_attempts):
        try:
            # Find first occurrence of initial variant
            initial_idx = -1
            initial_variant_found = ""
            for variant in initial_variants:
                idx = response.find(variant)
                if idx != -1 and (initial_idx == -1 or idx < initial_idx):
                    initial_idx = idx
                    initial_variant_found = variant
            
            # Find first occurrence of target variant
            target_idx = -1
            target_variant_found = ""
            for variant in target_variants:
                idx = response.find(variant)
                if idx != -1 and (target_idx == -1 or idx < target_idx):
                    target_idx = idx
                    target_variant_found = variant
            
            # Check if both found and in correct order
            if initial_idx != -1 and target_idx != -1 and initial_idx < target_idx:
                initial_scene_graph = response[initial_idx:target_idx].strip()
                target_scene_graph = response[target_idx:].strip()
                
                # Validate length
                if (len(initial_scene_graph) > MIN_SCENE_GRAPH_LENGTH and 
                    len(target_scene_graph) > MIN_SCENE_GRAPH_LENGTH):
                    logger.info(
                        f"Successfully extracted scene graphs using variants: "
                        f"'{initial_variant_found}' and '{target_variant_found}'"
                    )
                    return initial_scene_graph, target_scene_graph
                else:
                    logger.warning(
                        f"Extracted scene graphs too short: "
                        f"{len(initial_scene_graph)}, {len(target_scene_graph)}"
                    )
            else:
                if initial_idx == -1:
                    logger.warning("No initial scene graph variant found")
                if target_idx == -1:
                    logger.warning("No target scene graph variant found")
                if initial_idx >= target_idx:
                    logger.warning("Initial scene graph appears after target")
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
    
    logger.error("Failed to extract valid scene graphs after all attempts")
    return "", ""


def extract_scene_graph_json(response: str, max_attempts: int = DEFAULT_MAX_RETRIES) -> Tuple[str, str]:
    """
    Extract scene graphs using JSON pattern matching.
    
    Args:
        response: GPT model response text
        max_attempts: Maximum extraction attempts
        
    Returns:
        Tuple of (initial_scene_graph, target_scene_graph)
    """
    json_start_patterns = [
        '"initial_scene_graph":', '"initialSceneGraph":', 
        '{"initial_scene_graph":', '{"initialSceneGraph":'
    ]
    json_end_patterns = [
        '"target_scene_graph":', '"targetSceneGraph":', 
        '"final_scene_graph":', '"finalSceneGraph":'
    ]
    
    for attempt in range(max_attempts):
        try:
            # Find start of JSON
            start_idx = -1
            for pattern in json_start_patterns:
                idx = response.find(pattern)
                if idx != -1 and (start_idx == -1 or idx < start_idx):
                    start_idx = idx
            
            # Find where target/final scene graph begins
            target_idx = -1
            for pattern in json_end_patterns:
                idx = response.find(pattern, start_idx if start_idx != -1 else 0)
                if idx != -1 and (target_idx == -1 or idx < target_idx):
                    target_idx = idx
            
            if start_idx != -1 and target_idx != -1:
                initial_json = response[start_idx:target_idx].strip()
                
                # Find end of response
                end_markers = ['"output":', '"metrics":', '"evaluations":', '}]}']
                end_idx = len(response)
                for marker in end_markers:
                    idx = response.find(marker, target_idx + 20)
                    if idx != -1 and idx < end_idx:
                        end_idx = idx
                
                target_json = response[target_idx:end_idx].strip()
                
                # Validate JSON-like content
                if len(initial_json) > MIN_JSON_LENGTH and len(target_json) > MIN_JSON_LENGTH:
                    logger.info("Successfully extracted scene graphs using JSON pattern matching")
                    return initial_json, target_json
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"JSON extraction attempt {attempt + 1} failed: {e}")
    
    return "", ""


def robust_scene_graph_extraction(
    response: str,
    max_attempts: int = DEFAULT_MAX_RETRIES
) -> Tuple[str, str]:
    """
    Combined approach that tries multiple extraction methods.
    
    Args:
        response: GPT model response text
        max_attempts: Maximum extraction attempts
        
    Returns:
        Tuple of (initial_scene_graph, target_scene_graph)
    """
    logger.info("Attempting standard scene graph extraction...")
    initial, target = extract_scene_graph(response, max_attempts)
    
    if initial and target:
        return initial, target
    
    logger.info("Standard extraction failed, trying JSON pattern matching...")
    initial, target = extract_scene_graph_json(response, max_attempts)
    
    if initial and target:
        return initial, target
    
    logger.error("All extraction methods failed")
    return "", ""


def extract_intermediate_scene_graph(
    response: str,
    max_attempts: int = DEFAULT_MAX_RETRIES
) -> str:
    """
    Extract intermediate scene graph from response.
    
    Args:
        response: GPT model response text
        max_attempts: Maximum extraction attempts
        
    Returns:
        Intermediate scene graph string
    """
    intermediate_variants = [
        "intermediate_scene_graph", "Intermediate_scene_graph",
        "intermediate scene graph", "intermediateSceneGraph",
        "current_scene_graph", "Current_scene_graph", "currentSceneGraph"
    ]
    
    for attempt in range(max_attempts):
        try:
            # Find first occurrence of intermediate variant
            intermediate_idx = -1
            variant_found = ""
            for variant in intermediate_variants:
                idx = response.find(variant)
                if idx != -1 and (intermediate_idx == -1 or idx < intermediate_idx):
                    intermediate_idx = idx
                    variant_found = variant
            
            if intermediate_idx != -1:
                intermediate_scene_graph = response[intermediate_idx:].strip()
                
                if len(intermediate_scene_graph) > MIN_SCENE_GRAPH_LENGTH:
                    logger.info(
                        f"Successfully extracted intermediate scene graph "
                        f"using variant: '{variant_found}'"
                    )
                    return intermediate_scene_graph
                else:
                    logger.warning(
                        f"Extracted intermediate scene graph too short: "
                        f"{len(intermediate_scene_graph)}"
                    )
            else:
                logger.warning("No intermediate scene graph variant found")
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
    
    logger.error("Failed to extract valid intermediate scene graph")
    return ""


def robust_intermediate_extraction(
    response: str,
    max_attempts: int = DEFAULT_MAX_RETRIES
) -> str:
    """
    Combined approach for intermediate scene graph extraction.
    
    Args:
        response: GPT model response text
        max_attempts: Maximum extraction attempts
        
    Returns:
        Intermediate scene graph string
    """
    # Try standard extraction
    intermediate = extract_intermediate_scene_graph(response, max_attempts)
    
    if intermediate:
        return intermediate
    
    # Try JSON pattern matching
    logger.info("Standard extraction failed, trying JSON pattern matching...")
    try:
        json_patterns = [
            '"intermediate_scene_graph":', '"intermediateSceneGraph":', 
            '"current_scene_graph":', '"currentSceneGraph":'
        ]
        
        start_idx = -1
        for pattern in json_patterns:
            idx = response.find(pattern)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx
        
        if start_idx != -1:
            end_markers = [
                '"output":', '"metrics":', '"evaluations":', '}]}',
                '"target_scene_graph":'
            ]
            end_idx = len(response)
            for marker in end_markers:
                idx = response.find(marker, start_idx + 20)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            
            intermediate_json = response[start_idx:end_idx].strip()
            
            if len(intermediate_json) > MIN_JSON_LENGTH:
                logger.info(
                    "Successfully extracted intermediate scene graph "
                    "using JSON pattern matching"
                )
                return intermediate_json
    except Exception as e:
        logger.error(f"JSON intermediate extraction failed: {e}")
    
    logger.error("All intermediate extraction methods failed")
    return ""


# ============================================================================
# Score Extraction
# ============================================================================

def extract_similarity_scores(json_strings: List[str]) -> List[Tuple[int, int]]:
    """
    Extract similarity scores from JSON response strings.
    
    Args:
        json_strings: List of JSON response strings
        
    Returns:
        List of tuples (option_index, similarity_score)
    """
    scores = []
    pattern = r'"similarity with target scene graph":\s*(\d+)'
    
    for idx, json_str in enumerate(json_strings):
        match = re.search(pattern, json_str)
        if match:
            score = int(match.group(1))
            scores.append((idx, score))
            logger.debug(f"Extracted score {score} for option {idx}")
        else:
            logger.warning(f"No similarity score found for option {idx}")
    
    return scores


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_planning_problem(
    image_path: str,
    step_dict: Dict[str, Any],
    prompts: Dict[str, str],
    max_retries: int = DEFAULT_MAX_RETRIES
) -> Dict[str, Any]:
    """
    Analyze a planning problem using scene graphs and GPT-4 Vision.
    
    Args:
        image_path: Path to the composite image showing initial and goal states
        step_dict: Dictionary containing 'previous_steps' and 'options'
        prompts: Dictionary of prompt templates for the specific domain
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary containing analysis results and scores
    """
    # Initialize GPT client
    try:
        client = initialize_gpt_client()
    except ValueError as e:
        logger.error(f"Failed to initialize GPT client: {e}")
        return {"error": str(e)}
    
    # Step 1: Generate initial and target scene graphs
    logger.info("Step 1: Generating initial and target scene graphs...")
    prompt_1 = prompts.get("info", "") + prompts["scene_graph"]
    
    initial_scene_graph = ""
    target_scene_graph = ""
    retry_count = 0
    
    while (not initial_scene_graph or not target_scene_graph) and retry_count < max_retries:
        logger.info(f"Attempt {retry_count + 1} to generate scene graphs...")
        scene_graph_response = gpt_inference(client, image_path, prompt_1, [])
        initial_scene_graph, target_scene_graph = robust_scene_graph_extraction(
            scene_graph_response
        )
        
        if not initial_scene_graph or not target_scene_graph:
            logger.warning(
                f"Failed to extract scene graphs on attempt {retry_count + 1}. Retrying..."
            )
            retry_count += 1
            time.sleep(DEFAULT_RETRY_DELAY)
    
    if not initial_scene_graph or not target_scene_graph:
        logger.error("Failed to extract scene graphs after maximum retries")
        return {
            "error": "Failed to extract scene graphs",
            "scene_graph_response": scene_graph_response,
            "state_transition_response": "",
            "option_evaluation_response": [],
            "scores": []
        }
    
    # Step 2: Generate intermediate scene graph
    logger.info("Step 2: Generating intermediate scene graph...")
    previous_steps = step_dict.get("previous_steps", "")
    prompt_2 = prompts["state_transition"].format(
        previous_steps=previous_steps,
        start_scene_graph=initial_scene_graph
    )
    
    intermediate_scene_graph = ""
    retry_count = 0
    
    while not intermediate_scene_graph and retry_count < max_retries:
        logger.info(f"Attempt {retry_count + 1} to generate intermediate scene graph...")
        state_transition_response = gpt_inference(client, image_path, prompt_2, [])
        intermediate_scene_graph = robust_intermediate_extraction(state_transition_response)
        
        if not intermediate_scene_graph:
            logger.warning(
                f"Failed to extract intermediate scene graph on attempt {retry_count + 1}. "
                "Retrying..."
            )
            retry_count += 1
            time.sleep(DEFAULT_RETRY_DELAY)
    
    if not intermediate_scene_graph:
        logger.error("Failed to extract intermediate scene graph after maximum retries")
        return {
            "error": "Failed to extract intermediate scene graph",
            "scene_graph_response": scene_graph_response,
            "state_transition_response": state_transition_response,
            "option_evaluation_response": [],
            "scores": []
        }
    
    # Step 3: Evaluate all options
    logger.info("Step 3: Evaluating solution options...")
    option_eval_responses = []
    options = step_dict.get("options", [])
    
    for option_idx, steps_to_simulate in enumerate(options):
        logger.info(f"Evaluating option {option_idx + 1}/{len(options)}...")
        
        prompt_option = prompts["option_evaluation"].format(
            intermediate_scene_graph=intermediate_scene_graph,
            target_scene_graph=target_scene_graph,
            steps_to_simulate=steps_to_simulate,
            intermediate=intermediate_scene_graph,
            target=target_scene_graph
        )
        
        option_evaluation_response = ""
        retry_count = 0
        
        while not option_evaluation_response and retry_count < max_retries:
            logger.debug(
                f"Attempt {retry_count + 1} to evaluate option {option_idx + 1}..."
            )
            option_evaluation_response = gpt_inference(
                client, image_path, prompt_option, []
            )
            
            if not option_evaluation_response or \
               "option_evaluations" not in option_evaluation_response.lower():
                logger.warning(
                    f"Failed to get proper option evaluation on attempt "
                    f"{retry_count + 1}. Retrying..."
                )
                retry_count += 1
                time.sleep(DEFAULT_RETRY_DELAY)
            else:
                break
        
        option_eval_responses.append(option_evaluation_response)
    
    # Extract scores
    scores = extract_similarity_scores(option_eval_responses)
    
    logger.info(f"Analysis complete. Scores: {scores}")
    
    return {
        "scene_graph_response": scene_graph_response,
        "state_transition_response": state_transition_response,
        "option_evaluation_response": option_eval_responses,
        "scores": scores,
        "initial_scene_graph": initial_scene_graph,
        "target_scene_graph": target_scene_graph,
        "intermediate_scene_graph": intermediate_scene_graph
    }