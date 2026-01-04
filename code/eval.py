"""
Vision-Language Model Evaluation Framework

This script evaluates various vision-language models on different reasoning tasks.
It supports multiple model architectures and dataset types.

License: MIT (or your preferred license)
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, Tuple, Callable, Any

from models import (
    cogvlm_load, cogvlm_inference,
    internvlm_load, internvlm_inference,
    qwenvlm_load, qwenvlm_inference,
    llamavis_load, llamavis_inference,
    gpt_load, gpt_inference, gpt_text_inference,
    januspro_load, januspro_inference,
    qwq_load, qwq_inference,
    qwen_load, qwen_inference,
    llama_load, llama_inference,
    gemini_load, gemini_inference
)
from prompts import Prompts


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Evaluate vision-language models on reasoning tasks'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the model results JSON'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        required=True,
        help='Name of the file for storing results JSON'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the model to evaluate'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='robovqa',
        help='Dataset to use for evaluation'
    )
    parser.add_argument(
        '--prompt_type',
        type=str,
        default='detect',
        help='Type of prompt to use'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='Number of options for multiple choice questions'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=100,
        help='Maximum number of samples to process'
    )
    parser.add_argument(
        '--sleep_interval',
        type=float,
        default=5.0,
        help='Sleep interval between API calls (seconds)'
    )
    
    return parser.parse_args()


def get_model_functions(model_name: str) -> Tuple[Callable, Callable]:
    """
    Get the load and inference functions for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (load_function, inference_function)
        
    Raises:
        ValueError: If model name is not recognized
    """
    model_registry = {
        'cogvlm': (cogvlm_load, cogvlm_inference),
        'cogvlm_graph': (cogvlm_load, cogvlm_inference),
        'cogvlm_nraph': (cogvlm_load, cogvlm_inference),
        'internvlm': (internvlm_load, internvlm_inference),
        'internvlm_graph': (internvlm_load, internvlm_inference),
        'internvlm_nraph': (internvlm_load, internvlm_inference),
        'internvlm_wo_graph': (internvlm_load, internvlm_inference),
        'qwenvlm': (qwenvlm_load, qwenvlm_inference),
        'qwenvlm_graph': (qwenvlm_load, qwenvlm_inference),
        'qwenvlm_nraph': (qwenvlm_load, qwenvlm_inference),
        'llamavis': (llamavis_load, llamavis_inference),
        'gpt': (gpt_load, gpt_inference),
        'gpt_graph': (gpt_load, gpt_inference),
        'gpt_nraph': (gpt_load, gpt_inference),
        'gpt_text': (gpt_load, gpt_text_inference),
        'gpt_wo_graph': (gpt_load, gpt_inference),
        'januspro': (januspro_load, januspro_inference),
        'januspro_graph': (januspro_load, januspro_inference),
        'januspro_nraph': (januspro_load, januspro_inference),
        'qwq': (qwq_load, qwq_inference),
        'qwen': (qwen_load, qwen_inference),
        'llama': (llama_load, llama_inference),
        'gemini': (gemini_load, gemini_inference)
    }
    
    for key, functions in model_registry.items():
        if model_name.startswith(key):
            return functions
    
    raise ValueError(f"Unknown model name: {model_name}")




def get_prompt_template(model_name: str) -> str:
    """
    Get the appropriate prompt template based on model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Prompt template string
    """
    if 'graph' in model_name:
        return Prompts['graph']
    elif 'nraph' in model_name:
        return Prompts['nraph']
    elif 'wo_graph' in model_name:
        return Prompts['wo_graph']
    else:
        return Prompts['basic']




def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Dict[str, Any],
    inference_func: Callable,
    prompt_template: str,
    args: argparse.Namespace
) -> list:
    """
    Run evaluation on the dataset.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer/processor
        dataset: Preprocessed dataset
        inference_func: Inference function
        prompt_template: Prompt template to use
        args: Command-line arguments
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for idx, (item_id, item_data) in enumerate(dataset.items()):
        if idx >= args.max_samples:
            logger.info(f"Reached maximum samples limit: {args.max_samples}")
            break
        
        image_path = item_data['file_name']
        planning_prompt = item_data['prompt']
        
        # Format the prompt
        full_prompt = prompt_template.format(prompt=planning_prompt)
        
        try:
            # Run inference
            response = inference_func(
                model,
                tokenizer,
                image_path,
                full_prompt,
                []
            )
            
            # Save result
            result = {
                'goal': response,
                'image': image_path,
                'gt': item_data,
                'correct_answer': item_data.get('correct_option', None)
            }
            results.append(result)
            
            logger.info(f"Processed sample {idx + 1}/{min(len(dataset), args.max_samples)}")
            
            # Sleep to avoid rate limits
            time.sleep(args.sleep_interval)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            continue
        
        # Save intermediate results
        output_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_{args.output_name}_{args.dataset}_k{args.k}.json"
        )
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation with model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    
    try:
        # Get model functions
        load_func, inference_func = get_model_functions(args.model_name)
        
        # Get dataset info
        #dataset_path, preprocess_func = get_dataset_info(args.dataset)
        
        # Get prompt template
        prompt_template = get_prompt_template(args.model_name)
        
        # Load dataset
        dataset = json.load(open(f'datasets/{args.dataset}_metadata.json', 'r'))
        
        # Load model
        logger.info("Loading model...")
        model, tokenizer = load_func()
        logger.info("Model loaded successfully")
        
        # Run evaluation
        results = evaluate_model(
            model,
            tokenizer,
            dataset,
            inference_func,
            prompt_template,
            args
        )
        
        # Save final results
        output_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_{args.output_name}_{args.dataset}_k{args.k}_final.json"
        )
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {output_path}")
        logger.info(f"Total samples processed: {len(results)}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()