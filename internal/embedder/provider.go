package embedder

import (
	"fmt"
	"time"
)

// ProviderType represents the type of embedding provider
type ProviderType string

const (
	// ProviderOllama uses local Ollama for embeddings
	ProviderOllama ProviderType = "ollama"
	// ProviderOllamaRemote uses a remote Ollama instance
	ProviderOllamaRemote ProviderType = "ollama-remote"
	// ProviderOpenAI uses OpenAI's embedding API
	ProviderOpenAI ProviderType = "openai"
	// ProviderVoyage uses Voyage AI's embedding API
	ProviderVoyage ProviderType = "voyage"
	// ProviderVertexAI uses Google Cloud Vertex AI embedding API
	ProviderVertexAI ProviderType = "vertexai"
)

// IsValid returns true if the provider type is recognized
func (p ProviderType) IsValid() bool {
	switch p {
	case ProviderOllama, ProviderOllamaRemote, ProviderOpenAI, ProviderVoyage, ProviderVertexAI:
		return true
	default:
		return false
	}
}

// DisplayName returns a human-readable name for the provider
func (p ProviderType) DisplayName() string {
	switch p {
	case ProviderOllama:
		return "Ollama (local)"
	case ProviderOllamaRemote:
		return "Ollama (remote)"
	case ProviderOpenAI:
		return "OpenAI"
	case ProviderVoyage:
		return "Voyage AI"
	case ProviderVertexAI:
		return "Google Vertex AI"
	default:
		return "Unknown"
	}
}

// RequiresAPIKey returns true if the provider requires an API key
func (p ProviderType) RequiresAPIKey() bool {
	switch p {
	case ProviderOpenAI, ProviderVoyage:
		return true
	default:
		return false
	}
}

// DefaultDimensions returns the default embedding dimensions for this provider
func (p ProviderType) DefaultDimensions() int {
	switch p {
	case ProviderOllama, ProviderOllamaRemote:
		return 768 // Jina Code embeddings
	case ProviderOpenAI:
		return 1536 // text-embedding-3-small
	case ProviderVoyage:
		return 1024 // voyage-code-3
	case ProviderVertexAI:
		return 768 // text-embedding-004 default
	default:
		return 768
	}
}

// MaxContextTokens returns the maximum context window size in tokens for this provider.
// Returns a conservative limit with safety margin to prevent failures.
func (p ProviderType) MaxContextTokens() int {
	switch p {
	case ProviderOpenAI:
		return 8000 // text-embedding-3-small: 8191 minus safety margin
	case ProviderVoyage:
		return 15000 // voyage-code-3: 16000 minus safety margin
	case ProviderVertexAI:
		return 3000 // text-embedding-005: 3072 minus safety margin
	default: // ProviderOllama, ProviderOllamaRemote, unknown
		return 8000 // Jina v2: 8192 minus safety margin
	}
}

// AllProviders returns all available provider types
func AllProviders() []ProviderType {
	return []ProviderType{
		ProviderOllama,
		ProviderOllamaRemote,
		ProviderOpenAI,
		ProviderVoyage,
		ProviderVertexAI,
	}
}

// APIProviders returns providers that use external APIs (require keys)
func APIProviders() []ProviderType {
	return []ProviderType{
		ProviderOpenAI,
		ProviderVoyage,
	}
}

// ProviderConfig holds the configuration for creating an embedder
type ProviderConfig struct {
	Provider string
	Timeout  time.Duration // Timeout for embedding requests
	Ollama   OllamaProviderSettings
	OpenAI   OpenAIProviderSettings
	Voyage   VoyageProviderSettings
	VertexAI VertexAIProviderSettings
}

// OllamaProviderSettings holds Ollama-specific settings
type OllamaProviderSettings struct {
	URL   string
	Model string
}

// OpenAIProviderSettings holds OpenAI-specific settings
type OpenAIProviderSettings struct {
	APIKey string
	Model  string
}

// VoyageProviderSettings holds Voyage AI-specific settings
type VoyageProviderSettings struct {
	APIKey string
	Model  string
}

// VertexAIProviderSettings holds Google Vertex AI-specific settings
type VertexAIProviderSettings struct {
	ProjectID string
	Location  string
	Model     string
}

// NewFromConfig creates an Embedder based on the provider configuration
func NewFromConfig(cfg *ProviderConfig) (Embedder, error) {
	provider := ProviderType(cfg.Provider)
	if !provider.IsValid() {
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}

	switch provider {
	case ProviderOllama, ProviderOllamaRemote:
		return NewOllamaClient(OllamaConfig{
			BaseURL: cfg.Ollama.URL,
			Model:   cfg.Ollama.Model,
			Timeout: cfg.Timeout,
		}), nil

	case ProviderOpenAI:
		if cfg.OpenAI.APIKey == "" {
			return nil, &EmbeddingError{
				Code:       "API_KEY_REQUIRED",
				Message:    "OpenAI API key is required",
				Suggestion: "Run 'pm config provider' to configure your API key",
			}
		}
		return NewOpenAIClient(OpenAIConfig{
			APIKey:  cfg.OpenAI.APIKey,
			Model:   cfg.OpenAI.Model,
			Timeout: cfg.Timeout,
		}), nil

	case ProviderVoyage:
		if cfg.Voyage.APIKey == "" {
			return nil, &EmbeddingError{
				Code:       "API_KEY_REQUIRED",
				Message:    "Voyage API key is required",
				Suggestion: "Run 'pm config provider' to configure your API key",
			}
		}
		return NewVoyageClient(VoyageConfig{
			APIKey:  cfg.Voyage.APIKey,
			Model:   cfg.Voyage.Model,
			Timeout: cfg.Timeout,
		}), nil

	case ProviderVertexAI:
		if cfg.VertexAI.ProjectID == "" {
			return nil, &EmbeddingError{
				Code:       "PROJECT_ID_REQUIRED",
				Message:    "Vertex AI project ID is required",
				Suggestion: "Run 'pm config provider' to configure your project ID",
			}
		}
		return NewVertexAIClient(VertexAIConfig{
			ProjectID: cfg.VertexAI.ProjectID,
			Location:  cfg.VertexAI.Location,
			Model:     cfg.VertexAI.Model,
			Timeout:   cfg.Timeout,
		})

	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}
}
