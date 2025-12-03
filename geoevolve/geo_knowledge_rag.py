import os
import re
import shutil
from typing import List, Dict, Any
import requests
import wikipediaapi
import arxiv
import pymupdf
from git import Repo
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from wikipediaapi import WikipediaPage
from tqdm import tqdm
from more_itertools import batched
from geoevolve.llm import get_embeddings, get_llm


class GeoKnowledgeRAG:
    """
    Geographical Knowledge Retrival Augmented Generation
    """

    def __init__(self,
                 persist_dir: str,
                 rag_embedding_model_name: str = 'text-embedding-3-large',
                 rag_llm_model_name: str = 'gpt-4.1',
                 collection_name: str = 'geo_knowledge_db',
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        self.llm = get_llm(rag_llm_model_name)
        self.embeddings = get_embeddings(rag_embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size,
                                                                             chunk_overlap=chunk_overlap)
        self.embeddings = get_embeddings(rag_embedding_model_name)
        self.db = Chroma(collection_name=collection_name,
                         embedding_function=self.embeddings,
                         persist_directory=persist_dir)
        self.memory = MemorySaver()
        self.retriever = self.db.as_retriever(search_kwargs={'k': 4})

    def add_document_to_db(self, docs: List[Document], max_batch_size: int = 5461):
        """
        Add document to database
        :param docs:
        :param max_batch_size:
        :return:
        """
        chunks = self.splitter.split_documents(docs)
        for batched_chunk in tqdm(batched(chunks, max_batch_size), desc='Saving to Chroma'):
            list_batched_chunk = [*batched_chunk]
            self.db.add_documents(list_batched_chunk)

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
        retrieval_chain_rag_fusion = self.generate_queries() | self.retriever.map() | self.reciprocal_rank_fusion
        template = '''Answer the following question based on this context:

        {context}

        Question: {question}
        '''
        prompt = ChatPromptTemplate.from_template(template)
        geokg_rag_chain = (
                {'context': retrieval_chain_rag_fusion, 'question': RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        return geokg_rag_chain


def fetch_wikipedia_page(title: str) -> None | WikipediaPage:
    """
    Fetch wikipedia page
    :param title:
    :return:
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent='geoevolve',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    try:
        page = wiki.page(title)
        return page
    except requests.exceptions.Timeout:
        # 超时了 → 返回 None，而不是让程序崩溃
        return None
    except Exception:
        # 其他异常也返回 None
        return None


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
                'pdf_url': f'https://arxiv.org/pdf/{result.get_short_id()}',
                'published': result.published.isoformat()
            }
            print(meta)

            try:
                pdf_response = requests.get(f'https://arxiv.org/pdf/{result.get_short_id()}', timeout=30)
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
    except arxiv.HTTPError:
        print('Page request error')
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
    Search GitHub repository related to the topic
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
        code_repos.append(
            {'full_name': item['full_name'], 'html_url': item['html_url'], 'clone_url': item['clone_url']})
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
            if any(f.lower().endswith(s) for s in ['.py', '.js', '.java', '.cpp', '.r', '.f90', '.md']):
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
    Save GitHub codes to .txt file
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
        print(name)
        dest = os.path.join('../github_temp', name)
        try:
            clone_repo(r["clone_url"], dest)
            code_text = gather_code_text(dest)
            with open(f'{db_path}/{category}/{safe_filename(name)}.txt', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(code_text)
            print(f"[GITHUB] {r['full_name']}")
            shutil.rmtree(dest)
        except Exception as e:
            print(f"[GITHUB] clone failed {r['full_name']}: {e}")


def direct_add_document_to_db(rag: GeoKnowledgeRAG, knowledge: str, title: str, category: str, max_length: int = 1000):
    """
    Direct add document to geographical knowledge database
    :param rag:
    :param knowledge:
    :param title:
    :param category:
    :param max_length:
    :return:
    """
    docs = []
    if knowledge == '':
        return
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


def add_wiki_pages(topic: str, category: str, rag: GeoKnowledgeRAG, geo_knowledge_dir: str = None):
    """
    Add wiki pages to geographical knowledge database
    :param geo_knowledge_dir:
    :param is_save_file:
    :param rag:
    :param topic:
    :param category:
    :return:
    """
    wiki_page = fetch_wikipedia_page(topic)
    if wiki_page is None:
        return
    if wiki_page.exists():
        wiki_title = wiki_page.title
        wiki_doc = wiki_page.text
        if geo_knowledge_dir is not None:
            if not os.path.exists(f'{geo_knowledge_dir}/{category}'):
                os.mkdir(f'{geo_knowledge_dir}/{category}')
            with open(f'{geo_knowledge_dir}/{category}/{wiki_title}.txt', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(wiki_doc)
        direct_add_document_to_db(rag=rag, knowledge=wiki_doc, category=category, title=wiki_title)


def add_arxiv_papers(query: str, max_results: int, category: str, rag: GeoKnowledgeRAG, geo_knowledge_dir: str = None):
    """
    Add Arxiv Paper to geographical knowledge database
    :param rag:
    :param query:
    :param max_results:
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
            if pdf_text == '':
                continue
        if geo_knowledge_dir is not None:
            if not os.path.exists(f'{geo_knowledge_dir}/{category}'):
                os.mkdir(f'{geo_knowledge_dir}/{category}')
            with open(f'{geo_knowledge_dir}/{category}/{safe_filename(title)}.txt', 'w', encoding='utf-8',
                      errors='ignore') as f:
                f.write(text)
        # direct_add_document_to_db(rag=rag, knowledge=text, category=category, title=title)


def add_github_codes(query: str, max_repos: int, rag: GeoKnowledgeRAG, token: str = None, category: str = None):
    """
    Add GitHub codes to geographical knowledge database
    :param rag:
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
            direct_add_document_to_db(rag=rag, knowledge=code_text, category=category, title=name)
            shutil.rmtree(dest, ignore_errors=True)
        except Exception as e:
            print(f"[GITHUB] add failed {r['full_name']}: {e}")


def obtain_new_geo_knowledge(rag: GeoKnowledgeRAG, is_new_code_examples_needed: bool,
                             is_new_geographical_theory_needed: bool, keyword: str, category: str,
                             geo_knowledge_dir: str, github_token: str, max_repos: int = 3, max_arxiv_papers: int = 3):
    if is_new_code_examples_needed and github_token:
        add_github_codes(query=keyword, max_repos=max_repos, token=github_token, category=category, rag=rag)
    if is_new_geographical_theory_needed:
        add_wiki_pages(topic=keyword, category=category, rag=rag, geo_knowledge_dir=geo_knowledge_dir)
        add_arxiv_papers(query=keyword, max_results=max_arxiv_papers, category=category, rag=rag, geo_knowledge_dir=geo_knowledge_dir)


def obtain_new_geo_knowledge_from_outside(rag: GeoKnowledgeRAG, keyword: str, category: str, geo_knowledge_dir: str,
                                          max_arxiv_papers: int = 3):
    add_wiki_pages(topic=keyword, category=category, rag=rag, geo_knowledge_dir=geo_knowledge_dir)
    add_arxiv_papers(query=keyword, max_results=max_arxiv_papers, category=category, rag=rag,
                     geo_knowledge_dir=geo_knowledge_dir)


if __name__ == '__main__':
    embedding_model = get_embeddings('gemini-embedding-001')
    llm_model = get_llm('gemini-2.5-flash')
    rag = GeoKnowledgeRAG(persist_dir='./geoevolve_storage_re', rag_embedding_model_name=embedding_model, rag_llm_model_name=llm_model)
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
    response = rag_chain.invoke('What is semivariogram?')
    # print(response)
    # save_arxiv_papers('Spatial autocorrelation', 3, '../geo_knowledge', 'spatial_statistics')
    # add_wiki_pages(topic="Tobler's first law of geography", category='spatial-theory', rag=rag, geo_knowledge_dir='../geo_knowledge')
    # add_arxiv_papers(query='conformal prediction', category='uncertainty', rag=rag, geo_knowledge_dir='../geo_knowledge', max_results=5)





