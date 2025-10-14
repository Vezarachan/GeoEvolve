import asyncio
import os

from langchain_core.documents import Document
from geoevolve import GeoKnowledgeRAG, GeoEvolve


def make_geo_know_db(geo_knowledge_dir: str, persist_dir: str, max_length: int = 1000,
                     collection_name: str = 'geo_knowledge_db', chunk_size: int = 300, chunk_overlap: int = 50,
                     is_compressed: bool = False):
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    geokg_rag = GeoKnowledgeRAG(persist_dir=persist_dir,
                                collection_name=collection_name,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                is_compressed=is_compressed)
    docs = []
    for category in os.listdir(geo_knowledge_dir):
        category_path = os.path.join(geo_knowledge_dir, category)
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        knowledge = f.read()
                        if knowledge == '':
                            continue
                        if len(knowledge) > max_length:
                            chunks = [knowledge[i:i + max_length] for i in
                                      range(0, len(knowledge), max_length)]
                            chunked_docs = [Document(page_content=chunk,
                                                     metadata={'category': category, 'name': file.split('.')[0]})
                                            for chunk in chunks]
                            docs.extend(chunked_docs)
                        else:
                            doc = Document(page_content=knowledge,
                                           metadata={'category': category, 'name': file.split('.')[0]})
                            docs.append(doc)
    geokg_rag.add_document_to_db(docs)


def run_geo_evolution(
        initial_program_file: str, evaluator_file: str, config_path: str, output_path: str,
        rounds: int = 15, iterations_per_round: int = 15, rag_working_dir: str = './geoevolve_storage',
        rag_log_dir: str = '../geoevolve_logs', log_name: str = 'geocp', is_compressed: bool = False,
        openai_api_key: str = None
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(rag_log_dir):
        os.makedirs(rag_log_dir)
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key

    evolver = GeoEvolve(
        initial_program_file=initial_program_file,
        evaluator_file=evaluator_file,
        config_path=config_path,
        output_path=output_path,
        rag_working_dir=rag_working_dir,
        rag_log_dir=rag_log_dir,
        is_compressed=is_compressed,
        log_name=log_name
    )
    asyncio.run(evolver.evolve(rounds=rounds, iterations_per_round=iterations_per_round))
