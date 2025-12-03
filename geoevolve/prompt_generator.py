from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models import BaseChatModel


def analyze_evolved_code(llm: BaseChatModel, code: str, metrics: Dict, max_queries: int = 3) -> Dict[str, Any]:
    """
    Analyze the evolved code and its metrics
    :param llm: LLM Model
    :param code: Code to be analyzed
    :param metrics: Metrics of current code to be analyzed
    :param max_queries: Max number of knowledge retrieval queries
    :return:
    """
    code_analyser_template = '''
    You are an expert in Geography and Computer Science who is evaluating the evolved code from OpenEvolve (algorithm evolution framework).
    Here is the current algorithm code:
    {code}

    Here are the metrics for current algorithm code:
    {metrics}

    Task:
    1. Identify missing or problematic knowledge.
    2. Suggest search queries for retrieving useful geographical knowledge.
    3. Indicate whether new theories or code examples are needed.
    4. If either new theories or new code examples are needed, include the "new_geo_knowledge_to_fetch" field with "keyword" and "category". If neither is needed, omit this field entirely.

    You must respond strictly in valid JSON (no markdown, no explanations).
    Give at most {max_queries} search queries.

    Example behavior:
    - “If new knowledge is required:”
    {{
    "missing_or_problematic_knowledge": "lacking spatial interpolation theory",
    "search_queries": ["kriging spatial uncertainty", "geostatistical interpolation python"],
    "need_new_geographical_theory": true,
    "new_geo_knowledge_to_fetch": {{
        "keyword": "kriging spatial interpolation examples",
        "category": "geostatistical methods"
      }}
    }}

    - "If no new knowledge is needed"
    {{
    "missing_or_problematic_knowledge": "minor parameter tuning issue",
    "search_queries": ["parameter optimization for GIS algorithms"],
    "need_new_geographical_theory": false
    }}
    '''
    code_analyser_prompt = PromptTemplate(input_variables=['code', 'metrics', 'max_queries'],
                                          template=code_analyser_template)
    code_analyser_chain = (
            code_analyser_prompt
            | llm
            | JsonOutputParser()
    )
    return code_analyser_chain.invoke({'code': code, 'metrics': metrics, 'max_queries': max_queries})


def retrieve_geo_knowledge_via_rag(rag_chain: RunnableSerializable, query: str) -> List[str]:
    """
    Retrieve geographical knowledge via Rag chain
    :param rag_chain: GeoKnowRAG chain
    :param query: GeoKnow query
    :return:
    """
    geokg_rag_result = rag_chain.invoke(query)
    return geokg_rag_result


def generate_geo_knowledge_informed_prompt(llm: BaseChatModel, current_prompt: str, current_code: str,
                                           raw_knowledge: str,
                                           max_tokens: int = 400) -> str:
    """
    Generate geographical knowledge informed prompt
    :param llm: LLM Model
    :param current_prompt: Current prompt
    :param current_code: Current code
    :param raw_knowledge: Raw knowledge
    :param max_tokens: Max number of tokens for retrieved knowledge
    :return:
    """
    knowledge_str = '\n'.join(raw_knowledge)
    geo_informed_generator_template = '''
    You are a Geography and Computer Science expert guiding OpenEvolve in evolving geospatial algorithms.

    Context:
    current prompt:
    {current_prompt}

    Current code:
    {current_code}

    Relevant geographical knowledge:
    {knowledge_str}

    Your task:
    1. Generate a complete OpenEvolve prompt that clearly defines:
        - The algorithmic optimization goal (e.g., improving spatial accuracy, computational efficiency, or robustness)
        - The geographical and computational expertise relevant to the task
        - The optimization space, listing parameters, operators, or configurations OpenEvolve should explore
        - The constraints (accuracy, performance, memory, spatial consistency, runtime limits)
        - The evolutionary strategy (stages of parameter exploration and refinement)
        - The performance targets and success indicators (e.g., RMSE reduction, speedup ratio)
        - The expected output format for OpenEvolve (concise structured prompt only)
    2. The generated prompt should use a clear hierarchical structure with headings such as:
        - “You are an expert in Geography, Statistics, and Computer Science.
        - “Your Expertise”
        - “Optimization Space”
        - “Key Constraints”
        - “Evolutionary Strategy”
        - “Performance Targets”
        - “Success Indicators”
        - “Expected Output”
    3. Keep the prompt concise, readable, and <= {max_tokens} tokens.
    4. Output plain text only — strictly no Markdown code fences, no quotes, no escape sequences, no backslashes, and no hidden characters.
    5. Use normal readable formatting (newlines and indentation allowed).
    6. Do not include explanations or commentary — output only the final prompt text suitable for direct file writing or API input.

    Output:
    A single, fully formatted prompt ready for OpenEvolve input tailored to geospatial algorithm evolution.
    '''
    geo_informed_prompt = PromptTemplate(input_variables=['current_prompt', 'current_code', 'knowledge_str'],
                                         template=geo_informed_generator_template)
    geo_informed_chain = (
            geo_informed_prompt
            | llm
            | StrOutputParser()
    )
    return geo_informed_chain.invoke(
        {'current_prompt': current_prompt, 'current_code': current_code, 'knowledge_str': knowledge_str,
         'max_tokens': max_tokens})


def generate_prompt_without_geo_knowledge(llm: BaseChatModel, current_prompt: str, current_code: str,
                                          max_tokens: int = 400) -> str:
    """
    Generate a prompt without geographical knowledge.
    :param llm:
    :param current_prompt:
    :param current_code:
    :param max_tokens:
    :return:
    """
    no_geo_informed_template = '''
    You are a Geography and Computer Science expert guiding OpenEvolve in evolving geospatial algorithms.

    Context:
    current prompt:
    {current_prompt}

    Current code:
    {current_code}

    Your task:
    1. Generate a complete OpenEvolve prompt that clearly defines:
        - The algorithmic optimization goal (e.g., improving spatial accuracy, computational efficiency, or robustness)
        - The geographical and computational expertise relevant to the task
        - The optimization space, listing parameters, operators, or configurations OpenEvolve should explore
        - The constraints (accuracy, performance, memory, spatial consistency, runtime limits)
        - The evolutionary strategy (stages of parameter exploration and refinement)
        - The performance targets and success indicators (e.g., RMSE reduction, speedup ratio)
        - The expected output format for OpenEvolve (concise structured prompt only)
    2. The generated prompt should use a clear hierarchical structure with headings such as:
        - “You are an expert in Geography, Statistics, and Computer Science.
        - “Your Expertise”
        - “Optimization Space”
        - “Key Constraints”
        - “Evolutionary Strategy”
        - “Performance Targets”
        - “Success Indicators”
        - “Expected Output”
    3. Keep the prompt concise, readable, and <= {max_tokens} tokens.
    4. Output plain text only — strictly no Markdown code fences, no quotes, no escape sequences, no backslashes, and no hidden characters.
    5. Use normal readable formatting (newlines and indentation allowed).
    6. Do not include explanations or commentary — output only the final prompt text suitable for direct file writing or API input.

    Output:
    A single, fully formatted prompt ready for OpenEvolve input tailored to geospatial algorithm evolution.
    '''
    no_geo_informed_prompt = PromptTemplate(input_variables=['current_prompt', 'current_code'],
                                            template=no_geo_informed_template)
    no_geo_informed_chain = (
            no_geo_informed_prompt
            | llm
            | StrOutputParser()
    )
    return no_geo_informed_chain.invoke(
        {'current_prompt': current_prompt, 'current_code': current_code, 'max_tokens': max_tokens})


if __name__ == '__main__':
    result = analyze_evolved_code('', '', '')
    print(result)