import asyncio
import os

from langchain_core.documents import Document
from geoevolve import GeoKnowledgeRAG, GeoEvolve


def initialize_or_get_geo_know_db(persist_dir: str,
                          collection_name: str = 'geo_knowledge_db',
                          chunk_size: int = 300,
                          chunk_overlap: int = 50,
                          embedding_model_name: str = 'text-embedding-3-large',
                          llm_model_name: str = 'gpt-4.1') -> GeoKnowledgeRAG:
    print(embedding_model_name)
    print(llm_model_name)
    geokg_rag = GeoKnowledgeRAG(persist_dir=persist_dir,
                                collection_name=collection_name,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                rag_embedding_model_name=embedding_model_name,
                                rag_llm_model_name=llm_model_name)
    return geokg_rag


def import_knowledge_into_geo_know_db(geo_knowledge_dir: str,
                     persist_dir: str,
                     max_length: int = 1000,
                     collection_name: str = 'geo_knowledge_db',
                     chunk_size: int = 300,
                     chunk_overlap: int = 50,
                     embedding_model_name: str = 'text-embedding-3-large',
                     llm_model_name: str = 'gpt-4.1', ):
    """

    :param geo_knowledge_dir: GeoKnowledge directory
    :param persist_dir: GeoKnowledge Dataset Persist directory
    :param max_length: Max Text Length
    :param collection_name: Collection Name
    :param chunk_size: Chunk Size
    :param chunk_overlap: Chunk Overlap
    :param embedding_model_name: Embedding Model Name
    :param llm_model_name: LLM Model Name
    :return:
    """
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    geokg_rag = GeoKnowledgeRAG(persist_dir=persist_dir,
                                collection_name=collection_name,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                rag_embedding_model_name=embedding_model_name,
                                rag_llm_model_name=llm_model_name)
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
        initial_program_file: str,
        evaluator_file: str,
        config_path: str,
        output_path: str,
        rounds: int = 15,
        iterations_per_round: int = 15,
        rag_working_dir: str = './geoevolve_storage',
        log_dir: str = './geoevolve_logs',
        log_name: str = 'algorithm',
        embedding_model_name: str = 'text-embedding-3-large',
        llm_model_name: str = 'gpt-4.1',
        max_arxiv_papers: int = 3,
        chunk_overlap: int = 50,
):
    """
    :param initial_program_file: Initial Program File
    :param evaluator_file: Evaluator File
    :param config_path: Config Path
    :param output_path: Output Path
    :param rounds: Total outer loop rounds
    :param iterations_per_round: Iterations per round
    :param rag_working_dir: GeoKnowRAG Working Directory
    :param log_dir: Log Directory
    :param log_name: Log File Name
    :param embedding_model_name: Embedding Model Name
    :param llm_model_name: LLM Model Name
    :param max_arxiv_papers: Max ArXiv papers
    :param chunk_overlap: Chunk Overlap
    :return:
    """

    if not(os.path.isdir(rag_working_dir)):
        os.makedirs(rag_working_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    evolver = GeoEvolve(
        initial_program_file=initial_program_file,
        evaluator_file=evaluator_file,
        config_path=config_path,
        output_path=output_path,
        rag_working_dir=rag_working_dir,
        log_dir=log_dir,
        log_name=log_name,
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        max_arxiv_papers=max_arxiv_papers,
        chunk_overlap=chunk_overlap
    )
    asyncio.run(evolver.evolve(rounds=rounds, iterations_per_round=iterations_per_round))
