"""
Test cases for LLM processing functionality.
Tests LLM wrapper, response generation, and error handling.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from assignment_omni.llm.wrapper import build_llm, summarization_chain


class TestLLMProcessing:
    """Test cases for LLM processing."""
    
    def test_llm_initialization(self):
        """Test LLM initialization with different configurations."""
        with patch('assignment_omni.llm.wrapper.ChatOllama') as mock_ollama:
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            
            llm = build_llm()
            
            # Verify ChatOllama was called with correct parameters
            mock_ollama.assert_called_once()
            call_args = mock_ollama.call_args
            assert call_args[1]['model'] == 'llama3.2'
            assert call_args[1]['base_url'] == 'http://localhost:11434'
            assert call_args[1]['temperature'] == 0.2
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_response_generation(self, mock_ollama):
        """Test LLM response generation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a test response about RAG."
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        response = llm.invoke("What is RAG?")
        
        assert response.content == "This is a test response about RAG."
        mock_llm.invoke.assert_called_once_with("What is RAG?")
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_error_handling(self, mock_ollama):
        """Test LLM error handling."""
        # Mock LLM error
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        
        with pytest.raises(Exception, match="LLM service unavailable"):
            llm.invoke("Test prompt")
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_different_prompts(self, mock_ollama):
        """Test LLM with different types of prompts."""
        # Mock LLM responses
        mock_response = Mock()
        mock_response.content = "Weather response"
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        
        # Test weather prompt
        weather_response = llm.invoke("What's the weather like?")
        assert weather_response.content == "Weather response"
        
        # Test RAG prompt
        rag_response = llm.invoke("Summarize this document")
        assert rag_response.content == "Weather response"
        
        # Verify both calls were made
        assert mock_llm.invoke.call_count == 2
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_response_formatting(self, mock_ollama):
        """Test LLM response formatting and content extraction."""
        # Mock response with different formats
        mock_response = Mock()
        mock_response.content = "Formatted response"
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        response = llm.invoke("Test prompt")
        
        # Test content extraction
        content = getattr(response, 'content', str(response))
        assert content == "Formatted response"
    
    def test_summarization_chain_creation(self):
        """Test summarization chain creation."""
        with patch('assignment_omni.llm.wrapper.build_llm') as mock_build_llm:
            mock_llm = Mock()
            mock_build_llm.return_value = mock_llm
            
            chain = summarization_chain()
            
            # Verify chain was created
            assert chain is not None
            mock_build_llm.assert_called_once()
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_temperature_settings(self, mock_ollama):
        """Test LLM temperature configuration."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        
        # Verify temperature is set correctly
        call_args = mock_ollama.call_args
        assert call_args[1]['temperature'] == 0.2
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_model_configuration(self, mock_ollama):
        """Test LLM model configuration."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        
        # Verify model and base URL are set correctly
        call_args = mock_ollama.call_args
        assert call_args[1]['model'] == 'llama3.2'
        assert call_args[1]['base_url'] == 'http://localhost:11434'
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_langsmith_integration(self, mock_ollama):
        """Test LLM LangSmith integration."""
        with patch.dict(os.environ, {'LANGSMITH_API_KEY': 'test_key'}):
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            
            llm = build_llm()
            
            # Verify LangSmith environment variables are set
            assert os.environ.get('LANGCHAIN_TRACING_V2') == 'true'
            assert os.environ.get('LANGCHAIN_ENDPOINT') == 'https://api.smith.langchain.com'
            assert os.environ.get('LANGCHAIN_API_KEY') == 'test_key'


class TestLLMErrorScenarios:
    """Test cases for LLM error scenarios."""
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_connection_error(self, mock_ollama):
        """Test LLM connection error handling."""
        mock_ollama.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            build_llm()
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_timeout_error(self, mock_ollama):
        """Test LLM timeout error handling."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Request timeout")
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        
        with pytest.raises(Exception, match="Request timeout"):
            llm.invoke("Test prompt")
    
    @patch('assignment_omni.llm.wrapper.ChatOllama')
    def test_llm_invalid_response(self, mock_ollama):
        """Test LLM invalid response handling."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = None  # Invalid response
        mock_ollama.return_value = mock_llm
        
        llm = build_llm()
        response = llm.invoke("Test prompt")
        
        # Should handle None response gracefully
        assert response is None


if __name__ == "__main__":
    pytest.main([__file__])
