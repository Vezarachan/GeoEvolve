# GeoEvolve
> GeoEvolve aims to accelerate geospatial model discovery by the power of large language models.

## CLI Usage
```shell
pip install geoevolve

export OPENAI_API_KEY='your-openai-api-key'

# Optional If you already have a GeoKnowRAG
# Build GeoKnowRAG
python ./build_geo_knowledge_db.py --geo_knowledge_dir ./geo_knowledge --working_dir ../geoevolve_storage --chunk_size 300 --chunk_overlap 50 --topic_file ./topics.json --add_knowledge True --collect_knowledge True --github_token your-github-token --embedding_model text-embedding-3-large --llm_model gpt-4.1

# Run GeoEvolve
python ./run_geoevolve.py --initial_program_path path-to-initial_program --evaluator_path path-to-evaluator --config_path path-to-config --total_rounds 10 --num_iterations_per_round 10 --output path-to-output --log_name your_program_name --embedding_model text-embedding-3-large --llm_model gpt-4.1;
```

## Library Usage
```python
import os
from geoevolve import save_wiki_pages, save_arxiv_papers, save_github_codes, make_geo_know_db, run_geo_evolution

# Collect GeoKnowledge
topics = {
    'giscience_theory': [
    'Absolute vs relative vs relational space',
    'Cognitive geography',
    'Representation of scale in GIS'
  ],
  'spatial_modeling': [
    'Agent-based models in geography',
    'Spatial interaction models',
    'Gravity model in geography',
    'Entropy maximization models',
    'Complexity theory in geography'
  ]
}

for category, queries in topics.items():
    if not os.path.exists(f'./geo_knowledge/{category}'):
        os.mkdir(f'./geo_knowledge/{category}')
    print(f'Category: {category}')
    for query in queries:
        save_wiki_pages(query, db_path='./geo_knowledge', category=category)
        save_arxiv_papers(query, max_results=3, db_path='./geo_knowledge', category=category)
        save_github_codes(query, max_repos=3, token='token', db_path='./geo_knowledge',
                                  category=category)
        
# Build GeoKnowRAG with Chroma
make_geo_know_db(geo_knowledge_dir='./geo_knowledge', 
                 persist_dir='./geoevolve_storage',
                 embedding_model_name='text-embedding-3-large',
                 llm_model_name='gpt-4.1',)


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

**GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models** <br>
Peng Luo, Xiayin Lou, Yu Zheng, Zhuo Zheng, Stefano Ermon <br>