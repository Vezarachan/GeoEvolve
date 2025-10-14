import argparse
import json
import os
from geoevolve import GeoKnowledgeRAG, save_wiki_pages, save_arxiv_papers, save_github_codes
from langchain_core.documents import Document


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--working_dir',
        type=str,
        default='../geoevolve_storage',
        help='Geographical knowledge RAG working directory'
    )
    parser.add_argument(
        '--geo_knowledge_dir',
        type=str,
        default='../geo_knowledge',
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=300,
        help='Chunk size'
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=50,
        help='Chunk overlap'
    )
    parser.add_argument(
        '--topics_file',
        type=str,
        default='./topics.json',
        help='Geographical knowledge topics'
    )
    parser.add_argument(
        '--github_token',
        type=str,
        help='GitHub token',
        # default='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'
    )
    parser.add_argument(
        '--max_arxiv_papers',
        type=int,
        default=3,
        help='Maximum number of papers to consider'
    )
    parser.add_argument(
        '--max_repos',
        type=int,
        default=3,
        help='Maximum number of repos to consider'
    )
    parser.add_argument(
        '--collect_knowledge',
        type=bool,
        default=False,
        help='Whether to collect knowledge',
        required=True
    )
    parser.add_argument(
        '--add_knowledge',
        type=bool,
        default=False,
        help='Whether to add knowledge',
        required=True
    )

    config = parser.parse_args()

    topics_file = config.topics_file
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)


    working_dir = config.working_dir
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    geo_knowledge_dir = config.geo_knowledge_dir

    github_token = config.github_token

    max_arxiv_papers = config.max_arxiv_papers

    max_repos = config.max_repos

    collect_knowledge = config.collect_knowledge
    add_knowledge = config.add_knowledge

    geokg_rag = GeoKnowledgeRAG(persist_dir=working_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    knowledge_db = {}

    if collect_knowledge:
        print('Collecting Geo Knowledge ...')
        print(topics)
        for category, queries in topics.items():
            if not os.path.exists(f'{geo_knowledge_dir}/{category}'):
                os.mkdir(f'{geo_knowledge_dir}/{category}')
            print(f'Category: {category}')
            for query in queries:
                save_wiki_pages(query, db_path=geo_knowledge_dir, category=category)
                save_arxiv_papers(query, max_results=max_arxiv_papers, db_path=geo_knowledge_dir, category=category)
                save_github_codes(query, max_repos=max_repos, token=github_token, db_path=geo_knowledge_dir,
                                  category=category)

    if add_knowledge:
        print('Adding knowledge into RAG ...')
        docs = []
        for category in os.listdir(geo_knowledge_dir):
            category_path = os.path.join(geo_knowledge_dir, category)
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            knowledge = f.read()
                            max_length = 1000
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




