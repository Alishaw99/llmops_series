
import os
import sys
import json
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_groq import ChatGroq

from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException


PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


class ApiKeyManager:
    """
    Loads available API keys from an ECS JSON secret or
    individual environment variables.

    Validates only the keys needed by selected providers.
    """

    def __init__(self, required_keys=None):
        self.api_keys = {}
        required_keys = set(required_keys or [])

        # Optional ECS secret containing JSON credentials
        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "ECS API secret must be a JSON object"
                    )

                self.api_keys.update(parsed)
                log.info("Loaded API keys from ECS secret")

            except (json.JSONDecodeError, ValueError) as e:
                log.warning(
                    "Failed to parse ECS API secret",
                    error=str(e)
                )

        # Load available provider keys from environment
        for key in PROVIDER_KEYS.values():
            if not self.api_keys.get(key):
                value = os.getenv(key)

                if value:
                    self.api_keys[key] = value
                    log.info("API key loaded", key_name=key)

        # Validate required credentials only
        missing = [
            key for key in required_keys
            if not self.api_keys.get(key)
        ]

        if missing:
            log.error(
                "Missing required API keys",
                missing_keys=missing
            )
            raise DocumentPortalException(
                f"Missing required API keys: {missing}",
                sys
            )

        # Never log API key values, even partially
        log.info(
            "API key configuration validated",
            required_keys=sorted(required_keys)
        )

    def get(self, key: str) -> str:
        value = self.api_keys.get(key)

        if not value:
            raise KeyError(
                f"API key '{key}' is not configured"
            )

        return value


class ModelLoader:
    """
    Loads embedding and chat models according to YAML
    configuration and environment variables.
    """

    def __init__(self):

        # Step 1: Load environment variables
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        # Step 2: Load YAML configuration first
        self.config = load_config()

        log.info(
            "YAML config loaded",
            config_keys=list(self.config.keys())
        )

        # Step 3: Determine selected providers
        embedding_config = self.config["embedding_model"]

        embedding_provider = embedding_config.get(
            "provider", "openai"
        ).lower()

        self.llm_provider = os.getenv(
            "LLM_PROVIDER", "openai"
        ).lower()

        if embedding_provider not in PROVIDER_KEYS:
            raise ValueError(
                f"Unsupported embedding provider: "
                f"{embedding_provider}"
            )

        if self.llm_provider not in self.config["llm"]:
            raise ValueError(
                f"LLM provider '{self.llm_provider}' "
                "not found in YAML config"
            )

        # Step 4: Determine required credentials
        required_keys = {
            PROVIDER_KEYS[embedding_provider],
            PROVIDER_KEYS[self.llm_provider]
        }

        # Step 5: Initialize API key manager
        self.api_key_mgr = ApiKeyManager(
            required_keys=required_keys
        )

        log.info(
            "ModelLoader initialized",
            embedding_provider=embedding_provider,
            llm_provider=self.llm_provider
        )

    def load_embeddings(self):
        """
        Load the configured embedding model.
        """
        try:
            config = self.config["embedding_model"]

            provider = config.get(
                "provider", "openai"
            ).lower()

            model_name = config["model_name"]

            log.info(
                "Loading embedding model",
                provider=provider,
                model=model_name
            )

            if provider == "openai":
                return OpenAIEmbeddings(
                    model=model_name,
                    api_key=self.api_key_mgr.get(
                        "OPENAI_API_KEY"
                    )
                )

            elif provider == "google":
                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get(
                        "GOOGLE_API_KEY"
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported embedding provider: {provider}"
                )

        except Exception as e:
            log.error(
                "Error loading embedding model",
                error=str(e)
            )
            raise DocumentPortalException(
                "Failed to load embedding model",
                sys
            ) from e

    def load_llm(self):
        """
        Load the configured chat model.
        """
        try:
            provider_key = self.llm_provider

            llm_config = self.config["llm"][provider_key]

            provider = llm_config.get(
                "provider", provider_key
            ).lower()

            if provider != provider_key:
                raise ValueError(
                    "LLM provider configuration mismatch"
                )

            model_name = llm_config["model_name"]

            temperature = llm_config.get(
                "temperature", 0.2
            )

            max_tokens = llm_config.get(
                "max_output_tokens", 2048
            )

            log.info(
                "Loading LLM",
                provider=provider,
                model=model_name
            )

            if provider == "openai":
                return ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get(
                        "OPENAI_API_KEY"
                    ),
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            elif provider == "google":
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get(
                        "GOOGLE_API_KEY"
                    ),
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )

            elif provider == "groq":
                return ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get(
                        "GROQ_API_KEY"
                    ),
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            else:
                raise ValueError(
                    f"Unsupported LLM provider: {provider}"
                )

        except Exception as e:
            log.error(
                "Error loading LLM",
                error=str(e)
            )
            raise DocumentPortalException(
                "Failed to load LLM",
                sys
            ) from e


if __name__ == "__main__":

    loader = ModelLoader()

    # Test embeddings
    embeddings = loader.load_embeddings()

    vector = embeddings.embed_query(
        "What is Agentic AI?"
    )

    print("Embedding dimensions:", len(vector))

    # Test LLM
    llm = loader.load_llm()

    response = llm.invoke(
        "Explain Retrieval Augmented Generation briefly."
    )

    print("LLM response:", response.content)
