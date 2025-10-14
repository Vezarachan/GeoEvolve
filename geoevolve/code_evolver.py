import os
import tqdm
from openevolve import OpenEvolve
from openevolve.database import Program
from geoevolve.geo_knowledge_rag import GeoKnowledgeRAG
from geoevolve.prompt_generator import analyze_evolved_code, retrieve_geo_knowledge_via_rag, \
    generate_geo_knowledge_informed_prompt, generate_prompt_without_geo_knowledge
from geoevolve.utils import load_config, dump_config, save_round_level_logs, clean_markdown_labels_in_prompt


class GeoEvolve:
    """
    GeoEvolve: Automatic GeoAI Modeling with Multi-Agent Large Language Models.
    """
    def __init__(self, initial_program_file: str, evaluator_file: str, config_path: str, output_path: str,
                 rag_working_dir: str = './geoevolve_storage', rag_log_dir: str = '../geoevolve_logs',
                 log_name: str = 'geocp', is_compressed: bool = False):
        self.initial_program_file = initial_program_file
        self.evaluator_file = evaluator_file
        self.output_path = output_path
        self.config_path = config_path
        self.log_dir = rag_log_dir
        self.rag = GeoKnowledgeRAG(persist_dir=rag_working_dir, is_compressed=is_compressed)
        self.log_name = log_name

    async def _run_iterations(self, evolver: OpenEvolve, num_iterations: int) -> Program:
        """
        Run openevolve for num_iterations iterations.
        :param evolver:
        :param num_iterations:
        :return:
        """
        best = await evolver.run(iterations=num_iterations)
        return best

    async def evolve(self, rounds: int, iterations_per_round: int = 10):
        """
        Run GeoEvolve
        :param rounds:
        :param iterations_per_round:
        :return:
        """
        for r in tqdm.tqdm(range(rounds), desc="==> Round", leave=True, dynamic_ncols=True):
            if not os.path.exists(f'{self.output_path}/round_{r + 1}'):
                os.mkdir(f'{self.output_path}/round_{r + 1}')
            if r == 0:
                evolver = OpenEvolve(
                    initial_program_path=self.initial_program_file,
                    evaluation_file=self.evaluator_file,
                    config_path=self.config_path,
                    output_dir=f'{self.output_path}/round_{r + 1}')
            else:
                evolver = OpenEvolve(
                    initial_program_path=f'{self.output_path}/round_{r}/best/best_program.py',
                    evaluation_file=self.evaluator_file,
                    config_path=self.config_path,
                    output_dir=f'{self.output_path}/round_{r + 1}')
            best_program = await self._run_iterations(evolver, num_iterations=iterations_per_round)
            code = best_program.code
            metrics = best_program.metrics
            config = load_config(self.config_path)
            current_prompt = config['prompt']['system_message']
            # Analysis evolved code
            knowledge_needed = analyze_evolved_code(code, metrics)
            # RAG knowledge retrieval
            queries = knowledge_needed['search_queries']
            # is_classic_code_needed = knowledge_needed['need_code_examples']
            # is_geographical_theory_needed = knowledge_needed['need_geographical_theory']
            # retrieve_new_geo_knowledge(...)
            # get rag chain
            rag_chain = self.rag.make_rag_chain()
            # retrieve geographical knowledge
            all_knowledge = []
            for query in queries:
                knowledge = retrieve_geo_knowledge_via_rag(rag_chain, query)
                all_knowledge.append(knowledge)
            # update prompt
            updated_prompt = generate_geo_knowledge_informed_prompt(current_prompt, code, all_knowledge)
            # updated_prompt = generate_prompt_without_geo_knowledge(current_prompt, code)
            updated_prompt = clean_markdown_labels_in_prompt(updated_prompt)
            save_round_level_logs(self.log_dir, r, '', '', updated_prompt, metrics, self.log_name)
            config['prompt']['system_message'] = updated_prompt
            dump_config(self.config_path, config)
