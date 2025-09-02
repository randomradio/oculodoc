"""SGLang OCRFlux VLM analyzer implementation."""

import asyncio
import base64
import json
from io import BytesIO
from typing import Dict, List, Any, Optional
import logging

import aiohttp
import os
from PIL import Image

from ..interfaces.vlm_analyzer import IVLMAnalyzer, VLMAnalysisResult
from ..errors import ConfigurationError, InferenceError


class SGLangOCRFluxAnalyzer(IVLMAnalyzer):
    """SGLang OCRFlux analyzer for Vision-Language Model analysis.

    This implementation communicates with a SGLang server running the OCRFlux model
    for document understanding and content extraction.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 30000,
        model_name: str = "chat-doc/ocrflux-3b",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the SGLang OCRFlux analyzer.

        Args:
            host: Host where SGLang server is running.
            port: Port where SGLang server is listening.
            model_name: Name of the model to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
        """
        self.host = host
        self.port = port
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the SGLang OCRFlux analyzer.

        Args:
            config: Configuration dictionary containing connection settings.

        Raises:
            ConfigurationError: If configuration is invalid or connection fails.
        """
        session_created_here = False

        try:
            # Update configuration from provided config
            self.host = config.get("host", self.host)
            self.port = config.get("port", self.port)
            self.model_name = config.get("model_name", self.model_name)
            self.timeout = config.get("timeout", self.timeout)
            self.max_retries = config.get("max_retries", self.max_retries)
            self.retry_delay = config.get("retry_delay", self.retry_delay)

            # Update base URL
            self.base_url = f"http://{self.host}:{self.port}"

            # Create HTTP session
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
                session_created_here = True

            # Test connection
            await self._test_connection()

            self.logger.info(f"SGLang OCRFlux analyzer initialized: {self.base_url}")

        except Exception as e:
            # Clean up session if we created it and initialization failed
            if session_created_here and self.session and not self.session.closed:
                await self.session.close()
                self.session = None

            raise ConfigurationError(
                f"Failed to initialize SGLang OCRFlux analyzer: {e}"
            ) from e

    async def _test_connection(self) -> None:
        """Test connection to SGLang server."""
        if self.session is None:
            raise ConfigurationError("HTTP session not initialized")

        try:
            # Try to get model info or health check
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status not in [
                    200,
                    404,
                ]:  # 404 is ok if health endpoint doesn't exist
                    response.raise_for_status()

        except aiohttp.ClientError as e:
            raise ConfigurationError(
                f"Cannot connect to SGLang server at {self.base_url}: {e}"
            ) from e

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64 encoded image string.
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to bytes buffer
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # Encode to base64
        return base64.b64encode(image_bytes).decode("utf-8")

    def _prepare_prompt(self, prompt: str) -> str:
        """Prepare prompt for OCRFlux model.

        Args:
            prompt: User prompt.

        Returns:
            Formatted prompt for the model.
        """
        # OCRFlux specific prompt formatting
        system_prompt = (
            "You are an expert OCR and document analysis assistant. "
            "Analyze the provided document image and extract the requested information accurately. "
            "Provide structured, well-formatted responses."
        )

        return f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    async def _make_request(
        self, image_data: str, prompt: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Make request to SGLang server.

        Args:
            image_data: Base64 encoded image data.
            prompt: Prompt for the model.
            **kwargs: Additional request parameters.

        Returns:
            Response from the server.

        Raises:
            InferenceError: If the request fails after retries.
        """
        if self.session is None:
            raise InferenceError("HTTP session not initialized")

        # Prepare request data
        request_data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prepare_prompt(prompt)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.1),
            "stream": False,
        }

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise InferenceError(
                            f"Server returned status {response.status}: {error_text}"
                        )

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise InferenceError(
                        f"Request failed after {self.max_retries} attempts: {e}",
                        model_type="sglang_ocrflux",
                        operation="inference",
                    ) from e
            except Exception as e:
                raise InferenceError(
                    f"Unexpected error during inference: {e}",
                    model_type="sglang_ocrflux",
                    operation="inference",
                ) from e

        # This should never be reached, but just in case
        raise InferenceError(
            f"All retry attempts failed. Last error: {last_error}",
            model_type="sglang_ocrflux",
            operation="inference",
        )

    def _parse_response(self, response: Dict[str, Any]) -> List[VLMAnalysisResult]:
        """Parse SGLang response into VLMAnalysisResult objects.

        Args:
            response: Response from SGLang server.

        Returns:
            List of VLMAnalysisResult objects.
        """
        results = []

        try:
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

                    # Try to parse structured content
                    try:
                        # Check if content is JSON
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, dict):
                            # Handle structured response
                            for key, value in parsed_content.items():
                                result = VLMAnalysisResult(
                                    content_type=self._infer_content_type(key, value),
                                    content=str(value),
                                    confidence=0.9,  # Default confidence
                                    metadata={
                                        "field": key,
                                        "model": "sglang_ocrflux",
                                        "structured": True,
                                    },
                                )
                                results.append(result)
                        else:
                            # Handle as single text result
                            result = VLMAnalysisResult(
                                content_type="text",
                                content=content,
                                confidence=0.8,
                                metadata={"model": "sglang_ocrflux"},
                            )
                            results.append(result)
                    except json.JSONDecodeError:
                        # Handle as plain text
                        result = VLMAnalysisResult(
                            content_type="text",
                            content=content,
                            confidence=0.8,
                            metadata={"model": "sglang_ocrflux"},
                        )
                        results.append(result)
            else:
                self.logger.warning(f"Unexpected response format: {response}")

        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            # Return error result
            result = VLMAnalysisResult(
                content_type="error",
                content=f"Error parsing response: {e}",
                confidence=0.0,
                metadata={"error": True, "model": "sglang_ocrflux"},
            )
            results.append(result)

        return results

    def _infer_content_type(self, key: str, value: Any) -> str:
        """Infer content type based on key and value.

        Args:
            key: Field key.
            value: Field value.

        Returns:
            Content type string.
        """
        key_lower = key.lower()

        if "table" in key_lower:
            return "table"
        elif "image" in key_lower or "figure" in key_lower:
            return "image"
        elif "equation" in key_lower or "math" in key_lower:
            return "formula"
        else:
            return "text"

    async def analyze(
        self, image_data: str, prompt: str, **kwargs: Any
    ) -> List[VLMAnalysisResult]:
        """Analyze document image with VLM.

        Args:
            image_data: Base64 encoded image data or image path.
            prompt: The prompt for the VLM analysis.
            **kwargs: Additional keyword arguments for analysis.

        Returns:
            List of VLMAnalysisResult objects.

        Raises:
            InferenceError: If analysis fails.
        """
        try:
            # Normalize input: support path, data URI, or raw base64
            encoded_image: str
            if image_data.startswith("data:image"):
                # data URI: data:image/<fmt>;base64,<payload>
                try:
                    encoded_image = image_data.split(",", 1)[1]
                except Exception:
                    raise InferenceError(
                        "Invalid data URI format",
                        model_type="sglang_ocrflux",
                        operation="analyze",
                    )
            elif (
                (
                    image_data.startswith("/")
                    or image_data.startswith("./")
                    or image_data.startswith("../")
                )
                and os.path.exists(image_data)
                and os.path.isfile(image_data)
            ):
                # treat as file path only if it exists
                image = Image.open(image_data)
                encoded_image = self._encode_image(image)
            else:
                # assume raw base64 (no data URI header)
                encoded_image = image_data

            # Make request to SGLang server
            response = await self._make_request(encoded_image, prompt, **kwargs)

            # Parse response
            results = self._parse_response(response)

            return results

        except Exception as e:
            raise InferenceError(
                f"VLM analysis failed: {e}",
                model_type="sglang_ocrflux",
                operation="analyze",
            ) from e

    async def batch_analyze(
        self, image_data_list: List[str], prompts: List[str], **kwargs: Any
    ) -> List[List[VLMAnalysisResult]]:
        """Batch analyze multiple images.

        Args:
            image_data_list: List of base64 encoded image data or image paths.
            prompts: List of prompts for the VLM analysis.
            **kwargs: Additional keyword arguments for analysis.

        Returns:
            List of lists of VLMAnalysisResult objects.

        Raises:
            InferenceError: If batch analysis fails.
        """
        if len(image_data_list) != len(prompts):
            raise InferenceError(
                f"Number of images ({len(image_data_list)}) must match number of prompts ({len(prompts)})",
                model_type="sglang_ocrflux",
                operation="batch_analyze",
            )

        # Process sequentially for now (can be optimized for parallel processing)
        results = []
        for image_data, prompt in zip(image_data_list, prompts):
            try:
                result = await self.analyze(image_data, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch analysis failed for image: {e}")
                # Add error result
                error_result = VLMAnalysisResult(
                    content_type="error",
                    content=f"Batch analysis failed: {e}",
                    confidence=0.0,
                    metadata={"error": True, "model": "sglang_ocrflux"},
                )
                results.append([error_result])

        return results

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

        self.logger.info("SGLang OCRFlux analyzer cleaned up")

    async def get_server_info(self) -> Dict[str, Any]:
        """Get information about the SGLang server."""
        if self.session is None:
            return {"status": "not_connected"}

        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "connected",
                        "url": self.base_url,
                        "models": data.get("data", []),
                    }
                else:
                    return {
                        "status": "error",
                        "url": self.base_url,
                        "error": f"HTTP {response.status}",
                    }
        except Exception as e:
            return {"status": "error", "url": self.base_url, "error": str(e)}
