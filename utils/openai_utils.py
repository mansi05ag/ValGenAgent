#!/usr/bin/env python3
"""
OpenAI Utilities

This module provides utility functions for creating OpenAI clients with cross-platform proxy support.
"""

import os
import platform
import httpx
from openai import OpenAI


def create_openai_client(api_key: str, base_url: str = None, max_retries: int = 2, timeout: float = 60.0) -> OpenAI:
    """
    Create an OpenAI client with cross-platform proxy support.

    Args:
        api_key: OpenAI API key
        base_url: Custom base URL for OpenAI API
        max_retries: Maximum number of retries for requests
        timeout: Request timeout in seconds

    Returns:
        OpenAI client instance
    """
    # Configure proxy settings based on platform
    if platform.system().lower() == 'windows':
        # For Windows, use the Intel proxy
        proxy_url = "http://proxy-dmz.intel.com:912"
    else:
        # For Linux/Unix, no proxy
        proxy_url = None

    # Create httpx client with proxy configuration
    if proxy_url:
        http_client = httpx.Client(
            proxy=proxy_url,
            timeout=timeout,
            follow_redirects=True
        )
    else:
        http_client = httpx.Client(
            timeout=timeout,
            follow_redirects=True
        )

    # Set default base URL if not provided
    if base_url is None:
        base_url = "https://apis-internal.intel.com/generativeaiinference/v4"

    # Create OpenAI client with custom HTTP client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=max_retries
    )

    return client


def setup_cross_platform_environment():
    """Setup environment variables for cross-platform compatibility."""
    # Clear proxy settings to avoid conflicts
    for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        if proxy_var in os.environ:
            del os.environ[proxy_var]

    # Set no_proxy and NO_PROXY for Windows
    if platform.system().lower() == 'windows':
        os.environ['no_proxy'] = "http://proxy-dmz.intel.com:912"
        os.environ['NO_PROXY'] = "http://proxy-dmz.intel.com:912"
    else:
        # For Linux/Unix systems
        os.environ['no_proxy'] = ""
        os.environ['NO_PROXY'] = ""
