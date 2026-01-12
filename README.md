# CosPlan (Data coming soon)

## ❗️ Dataset Coming Soon
> **Note:** The specific datasets and metadata files required to run these evaluations are currently being prepared for public release. 
>
> **Please check back soon for the dataset upload!** 
> *Ensure you have the `datasets/` directory populated before running the evaluation scripts.*

## 📂 File Structure

*   **`eval.py`**  
    Main execution file for evaluating a model. It handles data loading, model initialization, and the evaluation loop.
*   **`models.py`**  
    Model-related utilities. Currently supports loading and inference for **InternVLM**, **JanusPro**, **CogVLM**, **GPT**, **Llama-Vision**, and others.
*   **`prompts.py`**  
    Contains the basic prompts used for standard evaluation.
*   **`scene_graph_analyzer.py`**  
    Analyzes scene graphs for Scene Graph Integration (SGI) tasks.
*   **`sgi_prompts.py`**  
    Contains specialized prompts used for SGI.

## 🚀 Usage

### Prerequisites
Ensure you have the necessary dependencies installed and that your `datasets/` folder contains the required `_metadata.json` files corresponding to your dataset argument.

### Running an Evaluation
To run the evaluation script, use the following command:

```bash
python eval.py \
  --output_dir ./results \
  --output_name janus_run_v1 \
  --model_name januspro \
  --dataset robovqa \
  --max_samples 10 \
  --sleep_interval 2.0
