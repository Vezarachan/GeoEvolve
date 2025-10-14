import json
from typing import List, Dict, Any

from langchain_core.runnables import RunnableSerializable
from openai import OpenAI

client = OpenAI()


def analyze_evolved_code(code: str, metrics: Dict, max_queries: int = 3) -> Dict[str, Any]:
    """
    Analyze the evolved code and its metrics
    :param code:
    :param metrics:
    :param max_queries:
    :return:
    """
    prompt = f'''
    You are an expert in Geography and Computer Science who is evaluating the evolved code from OpenEvolve (algorithm evolution framework).
    Here is the current algorithm code:
    {code}
    
    Here are the metrics for current algorithm code:
    {metrics}
    
    
    Task:
    1. Identify missing or problematic knowledge.
    2. Suggest search queries for retrieving useful geographical knowledge.
    3. Indicate whether theories or code examples are needed.
    
    You are required to respond strictly in JSON format:
    1. Do not include any markdown, code blocks, explanations, or extra text. 
    Only return valid JSON.
    Example format:
    {{
        "key": "value"
    }}
    2. Please give a maximum of {max_queries} search queries.
    
    Now, generate the requested JSON according to the instructions below:
    {{
      "missing_or_problematic_knowledge": "...",
      "search_queries": ["...", "...", "..."]
      "need_code_examples": true/false,
      "need_geographical_theory": true/false
    }}
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    print(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)


def retrieve_geo_knowledge_via_rag(rag_chain: RunnableSerializable, query: str) -> List[str]:
    """
    Retrieve geographical knowledge via Rag chain
    :param rag_chain:
    :param query:
    :return:
    """
    result = rag_chain.invoke(query)
    return result


def generate_geo_knowledge_informed_prompt(current_prompt: str, current_code: str, raw_knowledge: str,
                                           max_tokens: int = 500) -> str:
    """
    Generate geographical knowledge informed prompt
    :param current_prompt:
    :param current_code:
    :param raw_knowledge:
    :param max_tokens:
    :return:
    """
    knowledge_str = '\n'.join(raw_knowledge)
    prompt = f'''
    You are an expert in Geography and Computer Science, are now guiding OpenEvolve for geospatial algorithm evolution. Here are some important related information:
    current prompt:
    {current_prompt}
    
    Current code:
    {current_code}
    
    Relevant geographical knowledge:
    {knowledge_str}
    
    Task:
    1. Suggest algorithmic fixes or improvements.
    2. Propose new operators or parameters for OpenEvolve.
    3. Design a structured prompt that includes:
       - Evolutionary search guidance
       - Constraints
       - Expected outputs
    4. Make sure that the generated prompt is concise and not too long (<= {max_tokens} tokens).
    
    Output the final optimized prompt as plain text only (no explanation). Do not use Markdown code fences (```), do not include backslash escapes like '\n', '\t', or escaped quotes like '\"'. Output the prompt as normal readable text.
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content


def generate_prompt_without_geo_knowledge(current_prompt: str, current_code: str, max_tokens: int = 500) -> str:
    prompt = f'''
    You are an expert in Geography and Computer Science, are now guiding OpenEvolve for geospatial algorithm evolution. Here are some important related information:
    current prompt:
    {current_prompt}

    Current code:
    {current_code}

    Task:
    1. Suggest algorithmic fixes or improvements.
    2. Propose new operators or parameters for OpenEvolve.
    3. Design a structured prompt that includes:
        - Evolutionary search guidance
        - Constraints
        - Expected outputs
    4. Make sure that the generated prompt is concise and not too long (<= {max_tokens} tokens).

    Output the final optimized prompt as plain text only (no explanation). Do not use Markdown code fences (```), do not include backslash escapes like '\n', '\t', or escaped quotes like '\"'. Output the prompt as normal readable text.
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content
