# GeoEvolve
> GeoEvolve aims to accelerate geospatial model discovery by the power of large language models.

## CLI Usage
```shell
pip install geoevolve

export OPENAI_API_KEY='your-openai-api-key'

python ./build_geo_knowledge_db.py --geo_knowledge_dir ./geo_knowledge --working_dir ../geoevolve_storage --chunk_size 500 --chunk_overlap 50 --topic_file ./topics.json --add_knowledge True --collect_knowledge True --is_compressed False --github_token your-github-token

python ./run_geoevolve.py --initial_program_path path-to-initial_program --evaluator_path path-to-evaluator --config_path path-to-config --total_rounds 15 --num_iterations_per_round 15 --is_compressed True --output path-to-output --log_name geoevolve
```

## Library Usage
```python
import os
from geoevolve import save_wiki_pages, save_arxiv_papers, save_github_codes, make_geo_know_db, run_geo_evolution

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
    
make_geo_know_db('./geo_knowledge', './geoevolve_storage')

run_geo_evolution(initial_program_file='your-initial-program-path',
                  evaluator_file='your-evaluator-file',
                  config_path='your-config-path',
                  rounds=15,
                  iterations_per_round=15,
                  output_path='your-output-path')

```

This repository hosts the code base for the paper

**GeoXCP: uncertainty quantification of spatial explanations in explainable AI** <br>
Xiayin Lou, Peng Luo, Ziqi Li, Song Gao, Liqiu Meng <br>
International Journal of Geographical Information Science <br>
