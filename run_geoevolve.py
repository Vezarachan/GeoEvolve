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
        default='../outputs',
        help='The path of the evolved results'
    )
    parser.add_argument(
        '--working_dir',
        type=str,
        default='../geoevolve_storage',
        help='Geographical knowledge Retrieval Augmented Generation (GeoRAG) working directory',
    )
    parser.add_argument(
        '--rag_log_dir',
        type=str,
        default='../geoevolve_logs',
        help='The path of the rag log file'
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
        '--is_compressed',
        type=bool,
        default=False,
        help='Whether the prompt is compressed as a refined prompt'
    )
    parser.add_argument(
        '--openai_api_key',
        type=str,
        help='The API key for OpenAI'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default='',
        help='The log file name'
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
    rag_log_dir = config.rag_log_dir
    if not os.path.exists(rag_log_dir):
        os.mkdir(rag_log_dir)
    rounds = config.total_rounds
    num_iterations_per_round = config.num_iterations_per_round
    is_compressed = config.is_compressed
    openai_api_key = config.openai_api_key
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    log_name = config.log_name

    evolver = GeoEvolve(
        initial_program_file=initial_program_path,
        evaluator_file=evaluator_path,
        config_path=config_path,
        output_path=output_path,
        rag_working_dir=working_dir,
        rag_log_dir=rag_log_dir,
        is_compressed=is_compressed,
        log_name=log_name
    )

    asyncio.run(evolver.evolve(rounds=rounds, iterations_per_round=num_iterations_per_round))
