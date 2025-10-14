import os
import re
import shutil
from typing import List, Dict, Any
import requests
import wikipediaapi
import arxiv
import pymupdf
from git import Repo
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain.retrievers import ContextualCompressionRetriever
from wikipediaapi import WikipediaPage
from tqdm import tqdm
from more_itertools import batched


class GeoKnowledgeRAG:
    """
    Geographical Knowledge Retrival Augmented Generation
    """
    def __init__(self, persist_dir: str, collection_name: str = 'geo_knowledge_db', embedding_model: str = 'text-embedding-3-large', llm_model: str = 'gpt-4o-mini', chunk_size: int = 300, chunk_overlap: int = 50, is_compressed: bool = False):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=512)
        self.embeddings = embeddings
        self.db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_dir)
        self.memory = MemorySaver()
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = self.db.as_retriever(search_kwargs={'k': 4})
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
        self.is_compressed = is_compressed

    def add_document_to_db(self, doc, max_batch_size: int = 5461):
        """
        Add document to database
        :param doc:
        :param max_batch_size:
        :return:
        """
        chunks = self.splitter.split_documents(doc)
        for batched_chunk in tqdm(batched(chunks, max_batch_size), desc='Saving to Chroma'):
            self.db.add_documents(batched_chunk)

    def add_text_to_db(self, text: str, max_batch_size: int = 5461):
        """
        Add text to database
        :param text:
        :param max_batch_size:
        :return:
        """
        chunks = self.splitter.split_text(text)
        for batched_chunk in tqdm(batched(chunks, max_batch_size), desc='Saving to Chroma'):
            self.db.add_texts(batched_chunk)

    def generate_queries(self) -> RunnableSerializable:
        """
        Generate multiple search queries based on the question for better geographical knowledge retrieval
        :return:
        """
        template = '''You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (5 queries) separated by newlines:'''
        prompt_perspectives = ChatPromptTemplate.from_template(template=template)

        queries = (
            prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split('\n'))
        )
        return queries

    def reciprocal_rank_fusion(self, results: List[List], k=60):
        """
        Reciprocal rank fusion (RRF): a method for combining multiple result sets with different relevance indicators into a single result set.
        :param results:
        :param k:
        :return:
        """
        fused_scores = {}
        doc_contents = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_id = id(doc)
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (rank + k)
                doc_contents[doc_id] = doc

        fused_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_contents[doc_id], fused_scores[doc_id]) for doc_id, _ in fused_docs]

    def make_rag_chain(self) -> RunnableSerializable:
        """
        Construct a chain of consecutive steps in a Retrieval-Augmented Generation (RAG) system that are executed
        at inference time to provide a Large Language Model (LLM) with relevant, external information for generating
        more accurate and factually grounded responses.

                 -> Query 1             -> Document 1
        Question -> Query 2 -> Database -> Document 2 -> RRF -> Refined Answer
                 -> Query 3             -> Document 3
        :return:
        """
        if self.is_compressed:
            retrieval_chain_rag_fusion = self.generate_queries() | self.compression_retriever.map() | self.reciprocal_rank_fusion
        else:
            retrieval_chain_rag_fusion = self.generate_queries() | self.retriever.map() | self.reciprocal_rank_fusion
        template = '''Answer the following question based on this context:
        
        {context}
        
        Question: {question}
        '''
        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = (
            {'context': retrieval_chain_rag_fusion, 'question': RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain


def fetch_wikipedia_page(title: str) -> WikipediaPage:
    """
    Fetch wikipedia page
    :param title:
    :return:
    """
    wiki = wikipediaapi.Wikipedia(user_agent='geoevolve', language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    wiki_page = wiki.page(title)
    return wiki_page

def fetch_arxiv_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Fetch Arxiv Paper Metadata
    :param query:
    :param max_results:
    :return:
    """
    results = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=f'all:{query} OR ti:{query} OR abs:{query}', max_results=max_results,
                              sort_by=arxiv.SortCriterion.Relevance)
        search_results = client.results(search)
        for result in search_results:
            meta = {
                'id': result.get_short_id(),
                'title': result.title,
                'authors': [a.name for a in result.authors],
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'published': result.published.isoformat()
            }

            try:
                pdf_response = requests.get(result.pdf_url, timeout=30)
                if pdf_response.status_code == 200:
                    meta['pdf_bytes'] = pdf_response.content
                else:
                    meta['pdf_bytes'] = None
            except Exception as e:
                meta['pdf_bytes'] = None
                print(f'Download arxiv pdf failed: {str(e)}')
            results.append(meta)
    except arxiv.UnexpectedEmptyPageError:
        print('Arxiv paper not found')
    return results

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes
    :param pdf_bytes:
    :return:
    """
    if not pdf_bytes:
        return ''
    with pymupdf.Document(stream=pdf_bytes) as doc:
        text = '\n'.join([page.get_text() for page in doc])
    return text

def search_github_repos(query: str, max_repos: int = 3, token: str = None) -> List[Dict[str, Any]]:
    """
    Search github repository related to the topic
    :param query:
    :param max_repos:
    :param token:
    :return:
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    params = {'q': f'{query}', 'sort': 'stars', 'order': 'desc', 'per_page': max_repos}
    r = requests.get("https://api.github.com/search/repositories", params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'Github search failed: {r.status_code} {r.text}')
    data = r.json()
    code_repos = []
    for item in data.get('items', []):
        code_repos.append({'full_name': item['full_name'], 'html_url': item['html_url'], 'clone_url': item['clone_url']})
    return code_repos

def clone_repo(repo_url: str, dest_dir: str):
    """
    Clone repository from github to local directory
    :param repo_url:
    :param dest_dir:
    :return:
    """
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    Repo.clone_from(repo_url, dest_dir)

def gather_code_text(repo_path: str) -> str:
    """
    Gather code text from github repository
    :param repo_path:
    :return:
    """
    texts = []
    for root, dirs, files in os.walk(repo_path):
        for f in files:
            if any(f.lower().endswith(s) for s in ['.py', '.js', '.java', '.cpp', '.r', '.f90']):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as code_f:
                        text = code_f.read()
                        texts.append(text)
                except Exception as e:
                    continue
    return '\n\n'.join(texts)

def safe_filename(s: str) -> str:
    """
    Change filename so that the file can be saved safely
    :param s:
    :return:
    """
    return re.sub(r"[^\w\-_. ]", "_", s)[:200]

def save_wiki_pages(topic: str, db_path: str, category: str):
    """
    Save wiki pages to .txt file
    :param topic:
    :param db_path:
    :param category:
    :return:
    """
    wiki_page = fetch_wikipedia_page(topic)
    if wiki_page.exists():
        wiki_title = wiki_page.title
        wiki_doc = wiki_page.text
        with open(f'{db_path}/{category}/{wiki_title}.txt', 'w', encoding='utf-8', errors='ignore') as f:
            f.write(wiki_doc)

def save_arxiv_papers(query: str, max_results: int, db_path: str, category: str):
    """
    Save Arxiv Paper to .txt file
    :param query:
    :param max_results:
    :param db_path:
    :param category:
    :return:
    """
    papers = fetch_arxiv_papers(query, max_results)
    if len(papers) == 0:
        return
    for p in papers:
        title = p.get('title')
        text = p.get('summary', '')
        if p.get('pdf_bytes'):
            pdf_text = extract_text_from_pdf_bytes(p.get('pdf_bytes'))
            if len(pdf_text) > len(text):
                text = pdf_text
        with open(f'{db_path}/{category}/{safe_filename(title)}.txt', 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text)

def save_github_codes(query: str, max_repos: int, token: str = None, db_path: str = None, category: str = None):
    """
    Save github codes to .txt file
    :param query:
    :param max_repos:
    :param token:
    :param db_path:
    :param category:
    :return:
    """
    repos = search_github_repos(query, max_repos, token=token)
    for r in repos:
        name = r['full_name'].replace('/', '_')
        dest = os.path.join('../github_temp', name)
        try:
            clone_repo(r["clone_url"], dest)
            code_text = gather_code_text(dest)
            with open(f'{db_path}/{category}/{safe_filename(name)}.txt', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(code_text)
            print(f"[GITHUB] {r['full_name']}")
        except Exception as e:
            print(f"[GITHUB] clone failed {r['full_name']}: {e}")


if __name__ == '__main__':
    rag = GeoKnowledgeRAG(persist_dir='../geoevolve_storage')
    # papers = fetch_arxiv_papers('kriging', max_results=10)
    #
    # print(papers[0]['title'])
    # for p in papers:
    #     text = p.get('summary', '')
    #     if p.get('pdf_bytes'):
    #         pdf_text = extract_text_from_pdf_bytes(p.get('pdf_bytes'))
    #         if len(pdf_text) > len(text):
    #             text = pdf_text
    #     rag.add_document_to_db(text)
    rag_chain = rag.make_rag_chain()
    response = rag_chain.invoke('What are the different variants of Kriging interpolation?')
    print(response)
    # save_arxiv_papers('Spatial autocorrelation', 3, '../geo_knowledge', 'spatial_statistics')






