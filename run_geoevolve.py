import argparse
import asyncio
import os
from geoevolve import GeoEvolve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--initial_program_path',
        type=str,
        required=True,
        help='The path of the initial program'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='The path of the config file (config.yaml)'
    )
    parser.add_argument(
        '--evaluator_path',
        type=str,
        required=True,
        help='The path of the evaluator'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='The path of the evolved results'
    )
    parser.add_argument(
        '--working_dir',
        type=str,
        default='../geoevolve_storage',
        help='Geographical knowledge Retrieval Augmented Generation (GeoRAG) working directory',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./geoevolve_logs',
        help='The path of the log file'
    )
    parser.add_argument(
        '--total_rounds',
        type=int,
        default=10,
        help='The number of rounds to run'
    )
    parser.add_argument(
        '--num_iterations_per_round',
        type=int,
        default=10,
        help='The number of iterations per round'
    )
    parser.add_argument(
        '--openai_api_key',
        type=str,
        help='The API key for OpenAI'
    )
    parser.add_argument(
        '--gemini_api_key',
        type=str,
        help='The API key for Gemini'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default='',
        help='The log file name'
    )
    parser.add_argument(
        '--github_token',
        type=str,
        help='GitHub token',
        default='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'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        help='The name of the embedding model',
        default='openrouter-openai/text-embedding-3-large'
    )
    parser.add_argument(
        '--llm_model',
        type=str,
        help='The name of the llm model',
        default='openrouter-openai/gpt-4.1'
    )
    parser.add_argument(
        '--max_arxiv_papers',
        type=int,
        help='The maximum number of arxiv papers allowed',
        default=3
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        help='The number of chunk overlap allowed',
        default=50
    )

    config = parser.parse_args()

    initial_program_path = config.initial_program_path
    config_path = config.config_path
    evaluator_path = config.evaluator_path
    output_path = config.output
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    working_dir = config.working_dir
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    log_dir = config.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    rounds = config.total_rounds
    num_iterations_per_round = config.num_iterations_per_round
    openai_api_key = config.openai_api_key
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    gemini_api_key = config.gemini_api_key
    if gemini_api_key:
        os.environ['GEMINI_API_KEY'] = gemini_api_key
    log_name = config.log_name
    embedding_model_name = config.embedding_model
    llm_model_name = config.llm_model
    max_arxiv_papers = config.max_arxiv_papers
    chunk_overlap = config.chunk_overlap


    evolver = GeoEvolve(
        initial_program_file=initial_program_path,
        evaluator_file=evaluator_path,
        config_path=config_path,
        output_path=output_path,
        rag_working_dir=working_dir,
        log_dir=log_dir,
        log_name=log_name,
        max_arxiv_papers=max_arxiv_papers,
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        chunk_overlap=chunk_overlap
    )

    asyncio.run(evolver.evolve(rounds=rounds, iterations_per_round=num_iterations_per_round))

