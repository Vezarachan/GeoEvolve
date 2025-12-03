import os
import tqdm
from openevolve import OpenEvolve
from openevolve.database import Program
from ruamel.yaml.scalarstring import LiteralScalarString
from geoevolve.llm import get_embeddings, get_llm
from geoevolve.geo_knowledge_rag import GeoKnowledgeRAG, obtain_new_geo_knowledge_from_outside
from geoevolve.prompt_generator import analyze_evolved_code, retrieve_geo_knowledge_via_rag, \
    generate_geo_knowledge_informed_prompt
from geoevolve.utils import load_config, dump_config, save_round_level_logs, clean_markdown_labels_in_prompt


class GeoEvolve:
    """
    geoevolve: Automatic GeoAI Modeling with Multi-Agent Large Language Models.
    """

    def __init__(self,
                 initial_program_file: str,
                 evaluator_file: str,
                 config_path: str, output_path: str,
                 rag_working_dir: str = './geoevolve_storage',
                 log_dir: str = './geoevolve_logs',
                 geo_knowledge_dir: str = './geo_knowledge',
                 log_name: str = 'geoevolve',
                 max_arxiv_papers: int = 3,
                 embedding_model_name: str = 'openrouter-openai/text-embedding-3-large',
                 llm_model_name: str = 'openrouter-openai/gpt-4.1',
                 chunk_overlap: int = 50):
        self.initial_program_file = initial_program_file
        self.evaluator_file = evaluator_file
        self.output_path = output_path
        self.config_path = config_path
        self.log_dir = log_dir
        self.llm_model = get_llm(llm_model_name)
        self.rag = GeoKnowledgeRAG(persist_dir=rag_working_dir,
                                   rag_embedding_model_name=embedding_model_name,
                                   rag_llm_model_name=llm_model_name,
                                   chunk_overlap=chunk_overlap)
        self.rag_chain = self.rag.make_rag_chain()
        self.log_name = log_name
        self.geo_knowledge_dir = geo_knowledge_dir
        self.max_arxiv_papers = max_arxiv_papers


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
        Run geoevolve
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
                config = load_config(self.config_path)
            else:
                evolver = OpenEvolve(
                    initial_program_path=f'{self.output_path}/round_{r}/best/best_program.py',
                    evaluation_file=self.evaluator_file,
                    config_path=f'{self.output_path}/config_round_{r}.yaml',
                    output_dir=f'{self.output_path}/round_{r + 1}')
                config = load_config(f'{self.output_path}/config_round_{r}.yaml')
            best_program = await self._run_iterations(evolver, num_iterations=iterations_per_round)
            if r == rounds - 1:
                break
            code = best_program.code
            metrics = best_program.metrics

            current_prompt = config['prompt']['system_message']
            # Analysis evolved code
            knowledge_needed = analyze_evolved_code(self.llm_model, code, metrics)
            # RAG knowledge retrieval
            queries = knowledge_needed['search_queries']
            # Check if new geographical knowledge should be introduced
            is_new_geographical_theory_needed = knowledge_needed['need_new_geographical_theory']
            if is_new_geographical_theory_needed:
                new_knowledge_info = knowledge_needed['new_geo_knowledge_to_fetch']
                new_knowledge_keyword = new_knowledge_info['keyword']
                new_knowledge_category = new_knowledge_info['category']
                obtain_new_geo_knowledge_from_outside(rag=self.rag,
                                                      keyword=new_knowledge_keyword,
                                                      category=new_knowledge_category,
                                                      geo_knowledge_dir=self.geo_knowledge_dir,
                                                      max_arxiv_papers=self.max_arxiv_papers)
            # get rag chain
            # retrieve geographical knowledge
            all_knowledge = []
            for query in queries:
                knowledge = retrieve_geo_knowledge_via_rag(self.rag_chain, query)
                all_knowledge.append(knowledge)
            # update prompt
            updated_prompt = generate_geo_knowledge_informed_prompt(self.llm_model, current_prompt, code, all_knowledge)
            # updated_prompt = generate_prompt_without_geo_knowledge(current_prompt, code)
            updated_prompt = clean_markdown_labels_in_prompt(updated_prompt)

            save_round_level_logs(self.log_dir, r, knowledge_needed, all_knowledge, updated_prompt, metrics,
                                  self.log_name)
            config['prompt']['system_message'] = LiteralScalarString(updated_prompt)
            dump_config(f'{self.output_path}/config_round_{r + 1}.yaml', config)