import os
from dotenv import load_dotenv
from knowledge_base import KnowledgeBase

if __name__ == "__main__":
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable. Or use the .env file.")

    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://apis-internal.intel.com/generativeaiembedding/v2/"
    kb = KnowledgeBase(
        api_key=api_key,
        embed_base_url="https://apis-internal.intel.com/generativeaiembedding/v2/",
        llm_base_url="https://apis-internal.intel.com/generativeaiinference/v4",
    )

    kb.build_index(
        code_dirs=["code"],
        urls=["https://docs.pytorch.org/docs/stable/distributed.html"]
    )

    response = kb.query("Explain PyTorch Collective API all_reduce")
    print(response)