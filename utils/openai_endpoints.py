#!/usr/bin/env python3
"""
OpenAI Endpoints Configuration

This module provides centralized management of OpenAI-related API endpoints.
"""

# Base URLs
OPENAI_BASE_URL = "https://api.openai.com/v1"
EMBEDDING_BASE_URL = "https://apis-internal.intel.com/generativeaiembedding/v2"
INFERENCE_BASE_URL = "https://apis-internal.intel.com/generativeaiinference/v4"
AUTH_BASE_URL = "https://apis-internal.intel.com/v1/auth/token"

# Specific Endpoints
OPENAI_CHAT_COMPLETIONS = f"{OPENAI_BASE_URL}/chat/completions"
OPENAI_EMBEDDINGS = f"{OPENAI_BASE_URL}/embeddings"
