"""
Model Loading and Inference Functions

This module provides loading and inference functions for various vision-language models.
API keys should be set as environment variables.

Required environment variables:
- OPENAI_API_KEY: For GPT models
- GOOGLE_API_KEY: For Gemini models
- AZURE_OPENAI_KEY: For Azure OpenAI (if using Azure)
- AZURE_OPENAI_ENDPOINT: Azure endpoint (if using Azure)

License: MIT (or your preferred license)
"""

import base64
import io
import logging
import os
import time
from typing import Tuple, List, Optional, Any

import torch
import transformers
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

# Third-party imports (model-specific)
try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
    JANUS_AVAILABLE = True
except ImportError:
    JANUS_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 448
DEFAULT_MAX_NUM_IMAGES = 12


# ============================================================================
# Utility Functions
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_transform(input_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """
    Build image transformation pipeline.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transformation
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int
) -> Tuple[int, int]:
    """
    Find the closest aspect ratio from target ratios.
    
    Args:
        aspect_ratio: Current aspect ratio
        target_ratios: List of target aspect ratios
        width: Image width
        height: Image height
        image_size: Target image size
        
    Returns:
        Best matching aspect ratio tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = DEFAULT_MAX_NUM_IMAGES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    use_thumbnail: bool = False
) -> List[Image.Image]:
    """
    Dynamically preprocess image into multiple tiles.
    
    Args:
        image: PIL Image
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles
        image_size: Size of each tile
        use_thumbnail: Whether to add thumbnail
        
    Returns:
        List of processed images
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image_internvl(
    image_path: str,
    input_size: int = DEFAULT_IMAGE_SIZE,
    max_num: int = DEFAULT_MAX_NUM_IMAGES
) -> torch.Tensor:
    """
    Load image for InternVL model.
    
    Args:
        image_path: Path to image file
        input_size: Target image size
        max_num: Maximum number of image tiles
        
    Returns:
        Processed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ============================================================================
# Gemini Model
# ============================================================================

def gemini_load() -> Tuple[Any, None]:
    """
    Load Gemini model.
    
    Returns:
        Tuple of (model, None)
        
    Raises:
        ValueError: If API key is not set or package not available
    """
    if not GOOGLE_GENAI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it with your Google AI Studio API key."
        )
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("Gemini model loaded successfully")
    return model, None


def gemini_inference(
    model: Any,
    processor: None,
    image_path: str,
    query: str,
    history: List
) -> str:
    """
    Run inference with Gemini model.
    
    Args:
        model: Gemini model
        processor: Not used
        image_path: Path to image
        query: Text query
        history: Conversation history (not used)
        
    Returns:
        Model response text
    """
    try:
        img = Image.open(image_path)
        response = model.generate_content([query, img])
        time.sleep(10)  # Rate limiting
        logger.debug(f"Gemini response: {response.text}")
        return response.text
    except Exception as e:
        logger.error(f"Gemini inference error: {str(e)}")
        time.sleep(10)
        return ""


# ============================================================================
# OpenAI GPT Models
# ============================================================================

def gpt_load() -> Tuple[Any, None]:
    """
    Load OpenAI GPT model client.
    
    Returns:
        Tuple of (client, None)
        
    Raises:
        ValueError: If API key is not set or package not available
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not installed")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with your OpenAI API key."
        )
    
    client = openai.OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
    return client, None


def gpt_inference(
    client: Any,
    processor: None,
    image_path: str,
    query: str,
    history: List
) -> str:
    """
    Run inference with GPT-4V model.
    
    Args:
        client: OpenAI client
        processor: Not used
        image_path: Path to image
        query: Text query
        history: Conversation history (not used)
        
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
        logger.error(f"GPT inference error: {str(e)}")
        time.sleep(3)
        return ""


def gpt_text_inference(
    client: Any,
    processor: None,
    image_path: str,
    query: str,
    history: List
) -> str:
    """
    Run text-only inference with GPT model.
    
    Args:
        client: OpenAI client
        processor: Not used
        image_path: Not used
        query: Text query
        history: Conversation history (not used)
        
    Returns:
        Model response text
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }]
        )
        response_text = response.choices[0].message.content
        time.sleep(3)
        return response_text
    except Exception as e:
        logger.error(f"GPT text inference error: {str(e)}")
        time.sleep(3)
        return ""


# ============================================================================
# InternVL Model
# ============================================================================

def internvlm_load() -> Tuple[Any, Any]:
    """
    Load InternVL model.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = 'OpenGVLab/InternVL2-26B'
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    logger.info("InternVL model loaded successfully")
    return model, tokenizer


def internvlm_inference(
    model: Any,
    tokenizer: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """
    Run inference with InternVL model.
    
    Args:
        model: InternVL model
        tokenizer: Model tokenizer
        image_path: Path to image
        query: Text query
        history: Conversation history
        
    Returns:
        Model response text
    """
    pixel_values = load_image_internvl(image_path, max_num=12)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    question = '<image>\n' + query
    
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


# ============================================================================
# CogVLM Model
# ============================================================================

def cogvlm_load() -> Tuple[Any, Any]:
    """
    Load CogVLM model.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = "THUDM/cogvlm2-llama3-chat-19B"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine dtype based on GPU capability
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device).eval()
    
    logger.info("CogVLM model loaded successfully")
    return model, tokenizer


def cogvlm_inference(
    model: Any,
    tokenizer: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """
    Run inference with CogVLM model.
    
    Args:
        model: CogVLM model
        tokenizer: Model tokenizer
        image_path: Path to image
        query: Text query
        history: Conversation history
        
    Returns:
        Model response text
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    
    image = Image.open(image_path).convert('RGB')
    
    # Build conversation with history
    old_prompt = ''
    for old_query, response in history:
        old_prompt += f"{old_query} {response}\n"
    
    full_query = old_prompt + f"USER: {query} ASSISTANT:"
    
    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=full_query,
        history=history,
        images=[image],
        template_version='chat'
    )
    
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_dtype)]]
    }
    
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
    
    return response


# ============================================================================
# Qwen2-VL Model
# ============================================================================

def qwenvlm_load() -> Tuple[Any, Any]:
    """
    Load Qwen2-VL model.
    
    Returns:
        Tuple of (model, processor)
    """
    if not QWEN_VL_UTILS_AVAILABLE:
        logger.warning("qwen_vl_utils not available, some features may not work")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    logger.info("Qwen2-VL model loaded successfully")
    return model, processor


def qwenvlm_inference(
    model: Any,
    processor: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """
    Run inference with Qwen2-VL model.
    
    Args:
        model: Qwen2-VL model
        processor: Model processor
        image_path: Path to image
        query: Text query
        history: Conversation history (not used)
        
    Returns:
        Model response text
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": query},
        ],
    }]
    
    # Prepare inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    if QWEN_VL_UTILS_AVAILABLE:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = None, None
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


# ============================================================================
# Qwen Text Models
# ============================================================================

def qwen_load() -> Tuple[Any, Any]:
    """Load Qwen text model."""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Qwen model loaded successfully")
    return model, tokenizer


def qwen_inference(
    model: Any,
    tokenizer: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """Run inference with Qwen text model."""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=2000)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def qwq_load() -> Tuple[Any, Any]:
    """Load QwQ reasoning model."""
    model_name = "Qwen/QwQ-32B-Preview"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("QwQ model loaded successfully")
    return model, tokenizer


def qwq_inference(
    model: Any,
    tokenizer: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """Run inference with QwQ reasoning model."""
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=2000)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ============================================================================
# LLaMA Models
# ============================================================================

def llama_load() -> Tuple[Any, None]:
    """Load LLaMA text model."""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    logger.info("LLaMA model loaded successfully")
    return pipeline, None


def llama_inference(
    model: Any,
    tokenizer: None,
    image_path: str,
    query: str,
    history: List
) -> str:
    """Run inference with LLaMA text model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    
    outputs = model(messages, max_new_tokens=2000)
    return outputs[0]["generated_text"][-1]["content"]


def llamavis_load() -> Tuple[Any, Any]:
    """Load LLaMA Vision model."""
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    logger.info("LLaMA Vision model loaded successfully")
    return model, processor


def llamavis_inference(
    model: Any,
    processor: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """Run inference with LLaMA Vision model."""
    image = Image.open(image_path)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]
    }]
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=1200)
    return processor.decode(output[0])


# ============================================================================
# Janus Pro Model
# ============================================================================

def januspro_load() -> Tuple[Any, Any]:
    """
    Load Janus Pro model.
    
    Returns:
        Tuple of (model, processor)
        
    Raises:
        ImportError: If janus package not available
    """
    if not JANUS_AVAILABLE:
        raise ImportError("janus package not installed")
    
    model_path = "deepseek-ai/Janus-Pro-7B"
    
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    logger.info("Janus Pro model loaded successfully")
    return vl_gpt, vl_chat_processor


def januspro_inference(
    vl_gpt: Any,
    vl_chat_processor: Any,
    image_path: str,
    query: str,
    history: List
) -> str:
    """Run inference with Janus Pro model."""
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{query}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    tokenizer = vl_chat_processor.tokenizer
    pil_images = load_pil_images(conversation)
    
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer