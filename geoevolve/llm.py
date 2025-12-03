import os
from typing import Set

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

SUPPORTED_LLM_BACKENDS: Set[str] = {'OpenAI', 'Gemini', 'Anthropic', 'Groq', 'OpenRouter', 'Custom'}
SUPPORTED_EMBEDDING_BACKENDS: Set[str] = {'OpenAI', 'Gemini', 'OpenRouter'}


def get_embeddings(embedding_model: str, source: str = None) -> Embeddings:
    if embedding_model is None:
        embedding_model = 'text-embedding-small'
    if source is None:
        env_source = os.getenv('EMBEDDING_SOURCE')
        if env_source in SUPPORTED_LLM_BACKENDS:
            source = env_source
        else:
            if embedding_model[:15] == 'text-embedding-':
                source = 'OpenAI'
            elif embedding_model[:7] == 'gemini-':
                source = 'Gemini'
            elif embedding_model[:11] == 'openrouter-':
                source = 'OpenRouter'

    if source == 'OpenAI':
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.')
        return OpenAIEmbeddings(model=embedding_model, chunk_size=512, api_key=os.getenv('OPENAI_API_KEY'))
    elif source == 'Gemini':
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.')
        return GoogleGenerativeAIEmbeddings(model=f'models/{embedding_model}',
                                            google_api_key=os.getenv('GEMINI_API_KEY'))
    elif source == 'OpenRouter':
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.')

        return OpenAIEmbeddings(model=embedding_model[11:],
                                base_url='https://openrouter.ai/api/v1',
                                chunk_size=512,
                                api_key=os.getenv('OPENROUTER_API_KEY'),
                                check_embedding_ctx_length=False)
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'Gemini'"
        )

def get_llm(model: str = None,
            temperature: float = None,
            base_url: str = None,
            stop_sequences: list[str] = None,
            source: str = None,
            api_key: str = None,) -> BaseChatModel:
    if model is None:
        model = 'gpt-4o-mini'
    if temperature is None:
        temperature = 0.7
    if api_key is None:
        api_key = 'EMPTY'
    if source is None:
        env_source = os.getenv('LLM_SOURCE')
        if env_source in SUPPORTED_LLM_BACKENDS:
            source = env_source
        else:
            if model[:7] == 'claude-':
                source = 'Anthropic'
            elif model[:4] == 'gpt-':
                source = 'OpenAI'
            elif model[:7] == 'gemini-':
                source = 'Gemini'
            elif 'groq' in model.lower():
                source = 'groq'
            elif model[:11] == 'openrouter-':
                source = 'OpenRouter'
            else:
                raise ValueError('Unable to determine model source. Please specify a valid model.')

    if source == 'OpenAI':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.')
        return ChatOpenAI(model=model, temperature=temperature, stop_sequences=stop_sequences, max_tokens=8192)
    elif source == 'Gemini':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.'
            )
        return ChatOpenAI(model=model,
                          temperature=temperature,
                          stop_sequences=stop_sequences,
                          api_key=os.getenv('GEMINI_API_KEY'),
                          base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
    # elif source == 'Gemini':
    #     try:
    #         from langchain_google_genai import ChatGoogleGenerativeAI
    #     except ImportError:
    #         raise ImportError('langchain-google-genai package is required for OpenAI models. Install with: pip install langchain-google-genai.')
    #     return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=os.getenv('GEMINI_API_KEY'))
    elif source == 'Anthropic':
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                'langchain-anthropic package is required for OpenAI models. Install with: pip install langchain_anthropic.'
            )
        return ChatAnthropic(model=model,
                             temperature=temperature,
                             max_tokens=8192,
                             stop_sequences=stop_sequences)
    elif source == 'Groq':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.'
            )
        return ChatOpenAI(model=model,
                          temperature=temperature,
                          api_key=os.getenv('GROQ_API_KEY'),
                          base_url='https://api.groq.com/openai/v1',
                          stop_sequences=stop_sequences)
    elif source == 'OpenRouter':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.'
            )
        llm = ChatOpenAI(model=model[11:],
                         temperature=temperature,
                         max_tokens=8192,
                         stop_sequences=stop_sequences,
                         base_url='https://openrouter.ai/api/v1',
                         api_key=os.getenv('OPENROUTER_API_KEY'))
        return llm
    elif source == 'Custom':
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                'langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai.'
            )
        assert base_url is not None, 'base_url must be provided for customly served LLMs'
        llm = ChatOpenAI(model=model,
                         temperature=temperature,
                         max_tokens=8192,
                         stop_sequences=stop_sequences,
                         base_url=base_url,
                         api_key=api_key)
        return llm
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'Anthropic', 'Gemini', 'Groq'"
        )

if __name__ == '__main__':
    print(os.getenv('OPENROUTER_API_KEY'))

