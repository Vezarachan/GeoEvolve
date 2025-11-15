import os
import re
from typing import Dict, List
from ruamel.yaml import YAML

yaml = YAML()

def load_config(config_path: str) -> dict:
    """
    Load config.yaml file
    :param config_path:
    :return:
    """
    with open(config_path) as f:
        config = yaml.load(f)
    return config

def dump_config(config_path: str, config: dict):
    """
    Dump config.yaml file
    :param config_path:
    :param config:
    :return:
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

def save_round_level_logs(log_path: str, round_i: int, knowledge_needed: Dict, knowledge_retrieved: List[str], prompt_updated: str, metrics: Dict[str, float], name: str):
    """
    Save round level logs
    :param log_path:
    :param round_i:
    :param knowledge_needed:
    :param knowledge_retrieved:
    :param prompt_updated:
    :param metrics:
    :param name:
    :return:
    """
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    knowledge_str = '\n'.join(knowledge_retrieved)
    with open(f'{log_path}/geoevolve_{name}.log', 'a', encoding='utf-8') as f:
        f.write(f'Round {round_i}\n')
        f.write(f'metric: {metrics}\n')
        f.write(f'knowledge needed:\n {knowledge_needed}\n')
        f.write(f'knowledge retrieved:\n {knowledge_str}\n')
        f.write(f'prompt updated:\n {prompt_updated}\n\n')
        f.write('====================================================================================================')

def clean_markdown_labels_in_prompt(text: str) -> str:
    """
    Clean markdown labels in prompt
    :param text:
    :return:
    """
    text = re.sub(r"```[a-zA-Z]*\n?|```", "", text)
    text = text.replace("```", "")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    text.strip()
    return text

if __name__ == '__main__':
    # config = load_config('../examples/config.yaml')
    # print(config['prompt']['system_message'])
    # config['prompt']['system_message'] = 'Hello, World!'
    # dump_config('../examples/config.yaml', config)

    prompt = '''
    ```plaintext
    You are an expert in GIS and Python programming, tasked with enhancing the Ordinary Kriging algorithm for geospatial interpolation. Your focus is on algorithmic improvements, new operators, and structured output generation. Follow the instructions below:
    
    1. **Algorithmic Improvements**:
       - Implement alternative variogram models (spherical, Matérn) to capture diverse spatial structures.
       - Integrate cross-validation techniques using variable bin sizes during variogram fitting to optimize spatial autocorrelation handling.
       - Introduce a weighted least squares approach in variogram parameter estimation to enhance fitting accuracy.
       - Enable automatic selection of the best-fit variogram model based on the empirical variogram's characteristics.
    
    2. **New Operators**:
       - Create a "model selection" operator to analyze and select optimal variogram models based on empirical variogram attributes.
       - Develop a "variogram validation" operator for diagnostic assessments, including residual analysis and visual tools.
    
    3. **Structured Output**:
       - Refine the Ordinary Kriging algorithm to achieve improved RMSE performance over the current implementation.
       - Document the evolutionary search process, recording outcomes, model selection rationale, and parameter adjustments.
    
    **Constraints**:
    - Do not modify data preprocessing steps within the `run_kfold()` function.
    - All enhancements must maintain compatibility with the existing `ordinary_kriging()` structure.
    
    **Expected Outputs**:
    - A refined Ordinary Kriging algorithm demonstrating improved RMSE performance, documented changes, and validation results against the original implementation.
    ```
    '''

    print(clean_markdown_labels_in_prompt(prompt))