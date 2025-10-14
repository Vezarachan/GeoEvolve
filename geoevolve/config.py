import os
from dataclasses import dataclass, field
from typing import Set

@dataclass
class GeoEvolveConfig:
    # Geospatial problem to solve
    initial_program_path: str = ''
    config_path: str = ''
    evaluator_path: str = ''
    # LLM
    model_name: str = 'gpt-4o-mini'
    embedding_name: str = 'text-embedding-3-small'
    temperature: float = 0.7
    # GeoKnowRAG
    persist_path: str = './geoevolve_storage'
    github_token: str = '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'
    code_suffix: Set[str] = field(default_factory=lambda: {'.py', '.js', '.java', '.cpp', '.r', '.f90'})
    github_repo_temp_path: str = './github_temp'
    max_arxiv_papers: int = 5

    def to_dict(self):
        return {
            'initial_program_path': self.initial_program_path,
            'config_path': self.config_path,
            'evaluator_path': self.evaluator_path,
            'model_name': self.model_name,
            'embedding_name': self.embedding_name,
            'temperature': self.temperature,
            'github_token': self.github_token,
            'code_suffix': self.code_suffix,
            'persist_path': self.persist_path,
            'github_repo_temp_path': self.github_repo_temp_path,
            'max_arxiv_papers': self.max_arxiv_papers,
        }

default_config = GeoEvolveConfig()
