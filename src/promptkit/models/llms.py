"""Define interface protocols for LLM models, with optional structured tool support."""
from typing import Protocol, Iterator, TypedDict, Optional, List, Dict, Any


class ToolSpec(TypedDict, total=False):
    """Schema for a tool definition compatible with function-calling models."""
    name: str
    description: str
    parameters: Dict[str, Any]  # typically a JSON schema–style object
    type: str
    url: str


class LLMOutput(TypedDict):
    """Structured output from any LLM model."""
    reasoning: str
    output: str


class LLMModel(Protocol):
    """Protocol definition for all LLM client implementations."""

    temperature: float
    model: str
    supports_tools: bool  # indicates if this model can handle tool usage

    def __init__(self, model: str, temperature: float, supports_tools: bool = False):
        """Initialize the model client."""
        ...

    def generate(
        self,
        prompt: str,
        tools: Optional[List[ToolSpec]] = None
    ) -> LLMOutput:
        """Generate a response from the model.

        Args:
            prompt (str): The prompt to send to the model.
            tools (Optional[List[ToolSpec]]): Optional list of structured tool definitions.

        Raises:
            NotImplementedError: If tools are provided but not supported.

        Returns:
            LLMOutput: A dict containing 'reasoning' and 'output' strings.
        """
        ...

    def stream_generate(
        self,
        prompt: str,
        tools: Optional[List[ToolSpec]] = None
    ) -> Iterator[str]:
        """Stream model tokens as they are generated.

        Args:
            prompt (str): The input prompt.
            tools (Optional[List[ToolSpec]]): Optional list of structured tool definitions.

        Raises:
            NotImplementedError: If tools are provided but not supported.

        Returns:
            Iterator[str]: Yields strings as model tokens are produced.
        """
        ...


class EmbeddingsModel(Protocol):
    """Protocol for embedding model clients."""

    def embed(self, text: str, model: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text (str): Input text to embed.
            model (str): The embedding model to use.

        Returns:
            list[float]: Vector of floats representing the embedding.
        """
        ...
