#!/usr/bin/env python3
"""
OpenAI API Key Management Utility
This module provides functionality to cache and manage OpenAI API keys
with automatic expiration and regeneration.
"""

import os
import time
import functools
import json
from datetime import datetime, timedelta
import subprocess
import logging
import platform
import requests

# Set proxy environment variables at module import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global cache to store the API key and its expiration time
# Using a dictionary instead of an LRU cache since we only store one key
_api_key_cache = {
    'key': None,
    'expiry_time': None
}

def generate_new_api_key():
    os.environ["no_proxy"] = "http://proxy-dmz.intel.com:912"
    os.environ["NO_PROXY"] = "http://proxy-dmz.intel.com:912"
    if platform.system().lower() == 'windows':
        # call generate_new_api_key function for windows
        return generate_new_api_key_windows()
    elif platform.system().lower() == 'linux':
        # call generate_new_api_key function for windows
        return generate_new_api_key_linux()
    else:
        #platform is not supported.
        logger.error(f"Failed to generate new API key: The platform '{platform.system().lower()}' isn't supported.")
        return None

def generate_new_api_key_windows():
    """
    Generate a new OpenAI API key using Intel's internal API for windows.

    Returns:
        str: A new OpenAI API key
    """
    logger.info("Generating new OpenAI API key via Intel API...")

    if not requests:
        logger.error("requests library is not available. Please install it with: pip install requests")
        return None

    try:
        # Prepare the request data
        url = "https://apis-internal.intel.com/v1/auth/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": "13807c4e-0025-4bc7-86ad-bcb0f8fa6a73",
            "client_secret": "Kzv8Q~ykJ5-dnjWU2d8HzQM0rEE5tdGvUX.88cvW"
        }

        proxies = {
            'no_proxy': "",
            'NO_PROXY': ""
        }

        # Make the POST request
        logger.debug("Making POST request to Intel API...")
        response = requests.post(url, headers=headers, data=data, proxies=proxies, timeout=30)

        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.debug(f"Response content: {response.text[:100]}...")
            return None

        # Parse the response to extract the access token
        try:
            response_json = response.json()
            new_key = response_json.get('access_token')

            if not new_key:
                logger.error("Access token not found in API response")
                logger.debug(f"API Response: {str(response_json)[:100]}...")  # Log first 100 chars only for security
                return None

            logger.info("Successfully generated new API key")
            return new_key

        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from API")
            logger.debug(f"Raw response: {response.text[:100]}...")  # Log first 100 chars only for security
            return None

    except requests.exceptions.Timeout:
        logger.error("Request timed out while connecting to Intel API")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Intel API - check your network connection")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to generate new API key: {str(e)}")
        return None

def generate_new_api_key_linux():
    """
    Generate a new OpenAI API key using Intel's internal API for linux.
    Returns:
        str: A new OpenAI API key
    """
    logger.info("Generating new OpenAI API key via Intel API...")

    try:
        # Prepare the curl command to get the access token
        curl_command = [
            "bash", "-c",
            'no_proxy="" && NO_PROXY="" && response=$(curl -X POST "https://apis-internal.intel.com/v1/auth/token" '
            '-d "grant_type=client_credentials&client_id=13807c4e-0025-4bc7-86ad-bcb0f8fa6a73&client_secret=Kzv8Q~ykJ5-dnjWU2d8HzQM0rEE5tdGvUX.88cvW" '
            '-H "Content-Type: application/x-www-form-urlencoded") && echo "$response"'
        ]

        # Execute the command and get the response
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        response_text = result.stdout.strip()

        # Parse the response to extract the access token
        try:
            response_json = json.loads(response_text)
            new_key = response_json.get('access_token')

            if not new_key:
                logger.error("Access token not found in API response")
                logger.debug(f"API Response: {response_text[:100]}...")  # Log first 100 chars only for security
                return None

            logger.info("Successfully generated new API key")
            return new_key

        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from API")
            logger.debug(f"Raw response: {response_text[:100]}...")  # Log first 100 chars only for security
            return None

    except subprocess.SubprocessError as e:
        logger.error(f"Failed to execute curl command: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to generate new API key: {str(e)}")
        return None

def is_key_valid(api_key):
    """
    Check if the API key is valid by making a test request to OpenAI.
    Args:
        api_key (str): The OpenAI API key to validate
    Returns:
        bool: True if the key is valid, False otherwise
    """
    if not api_key:
        return False

    # For now, just check if the key is not None or empty
    return bool(api_key)

def get_openai_api_key(force_new=True, key_lifetime_minutes=60):
    """
    Get a valid OpenAI API key, either from cache or by generating a new one.
    Args:
        force_new (bool): If True, ignores cached key and generates a new one
        key_lifetime_minutes (int): How long (in minutes) a key should be considered valid
    Returns:
        str: A valid OpenAI API key
    """
    global _api_key_cache

    current_time = time.time()

    # First check environment variable
    env_key = os.environ.get("OPENAI_API_KEY")

    # If force_new is True, always generate a new key
    if force_new:
        logger.info("Forcing generation of new API key")
        new_key = generate_new_api_key()
        if new_key:
            _api_key_cache['key'] = new_key
            _api_key_cache['expiry_time'] = current_time + (key_lifetime_minutes * 60)
            os.environ["OPENAI_API_KEY"] = new_key
            logger.info(f"New API key will expire at {datetime.fromtimestamp(_api_key_cache['expiry_time'])}")
            return new_key

    # Check if we have a cached key that's not expired
    cached_key = _api_key_cache.get('key')
    expiry_time = _api_key_cache.get('expiry_time', 0)

    if cached_key and expiry_time > current_time and is_key_valid(cached_key):
        logger.info("Using cached API key (valid)")
        # Ensure the environment variable is set
        os.environ["OPENAI_API_KEY"] = cached_key
        return cached_key

    # Check if environment variable key is valid
    if env_key and is_key_valid(env_key) and not force_new:
        logger.info("Using API key from environment variable")
        # Update cache with environment key
        _api_key_cache['key'] = env_key
        _api_key_cache['expiry_time'] = current_time + (key_lifetime_minutes * 60)
        return env_key

    # Generate a new key
    logger.info("No valid key found, generating new API key")
    new_key = generate_new_api_key()

    if new_key:
        new_expiry_time = current_time + (key_lifetime_minutes * 60)
        _api_key_cache['key'] = new_key
        _api_key_cache['expiry_time'] = new_expiry_time
        os.environ["OPENAI_API_KEY"] = new_key
        logger.info(f"New API key will expire at {datetime.fromtimestamp(new_expiry_time)}")
        return new_key
    else:
        logger.error("Failed to generate a valid API key")
        return None

def clear_api_key_cache():
    """Clear the API key cache, forcing a new key to be generated next time."""
    global _api_key_cache
    _api_key_cache = {'key': None, 'expiry_time': None}
    logger.info("API key cache cleared")

# Example of how to use the function as a decorator to ensure valid API key
# def with_valid_api_key(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         # Ensure we have a valid API key before calling the function
#         api_key = get_openai_api_key()
#         if not api_key:
#             raise ValueError("Failed to obtain a valid OpenAI API key")
#         return func(*args, **kwargs)