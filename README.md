from unicodedata import category

# GeoEvolve
> GeoEvolve aims to accelerate geospatial model discovery by the power of large language models.

## CLI Usage
```shell
pip install geoevolve

export OPENAI_API_KEY='your-openai-api-key' && source ~/.zshrc

# Optional If you already have a GeoKnowRAG
# Build GeoKnowRAG
python ./build_geo_knowledge_db.py --geo_knowledge_dir ./geo_knowledge --working_dir ../geoevolve_storage --chunk_size 300 --chunk_overlap 50 --topic_file ./topics.json --add_knowledge True --collect_knowledge True --github_token your-github-token --embedding_model text-embedding-3-large --llm_model gpt-4.1

# Run GeoEvolve
python ./run_geoevolve.py --initial_program_path path-to-initial_program --evaluator_path path-to-evaluator --config_path path-to-config --total_rounds 10 --num_iterations_per_round 10 --output path-to-output --log_name your_program_name --embedding_model text-embedding-3-large --llm_model gpt-4.1;
```

## Library Usage
```python
import os
from geoevolve import save_wiki_pages, save_arxiv_papers, save_github_codes, initialize_or_get_geo_know_db, import_knowledge_into_geo_know_db, run_geo_evolution

os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
# Initialize an Empty GeoKnowRAG with Chroma
geokg_rag = initialize_or_get_geo_know_db(persist_dir='your_geoevolve_storage',
                                   embedding_model_name='text-embedding-3-large',
                                   llm_model_name='gpt-4.1')

# Run GeoEvolve
run_geo_evolution(initial_program_file='your-initial-program-path',
                  evaluator_file='your-evaluator-file',
                  config_path='your-config-path',
                  rounds=15,
                  iterations_per_round=15,
                  output_path='your-output-path',
                  log_name='your-program-name',
                  embedding_model_name='text-embedding-3-large',
                  llm_model_name='gpt-4.1')

```

## Import Knowledge into GeoKnowRAG
```python
from geoevolve import initialize_or_get_geo_know_db, import_knowledge_into_geo_know_db
from langchain_core.documents import Document

# You can import knowledge from a well-structured geographical knowledge directory
# The structure like geo_knowledge/{category}/{knowledge}.txt

import_knowledge_into_geo_know_db(geo_knowledge_dir='your_geo_knowledge_dir',
                                  persist_dir='your_geoevolve_storage',
                                  collection_name='your_collection_name',
                                  embedding_model_name='text-embedding-3-large',
                                  llm_model_name='gpt-4.1')

# You can also just import a document into the GeoKnowRAG
rag = initialize_or_get_geo_know_db(persist_dir='your_geoevolve_storage',
                                   embedding_model_name='text-embedding-3-large',
                                   llm_model_name='gpt-4.1')

knowledge = '...'
category = '...'
title = '...'

max_length = 1000
docs = []
if knowledge != '':
    if len(knowledge) > max_length:
        chunks = [knowledge[i:i + max_length] for i in
                  range(0, len(knowledge), max_length)]
        chunked_docs = [Document(page_content=chunk,
                                 metadata={'category': category, 'name': title})
                        for chunk in chunks]
        docs.extend(chunked_docs)
    else:
        doc = Document(page_content=knowledge,
                       metadata={'category': category, 'name': title})
        docs.append(doc)
rag.add_document_to_db(docs)
```

**GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models** <br>
Peng Luo, Xiayin Lou, Yu Zheng, Zhuo Zheng, Stefano Ermon <br>