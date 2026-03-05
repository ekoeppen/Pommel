package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// VertexAIConfig holds configuration for the Google Vertex AI embedding client.
type VertexAIConfig struct {
	ProjectID string
	Location  string
	Model     string
	BaseURL   string // For testing only
	Timeout   time.Duration
}

// DefaultVertexAIConfig returns the default configuration for Vertex AI.
func DefaultVertexAIConfig() VertexAIConfig {
	return VertexAIConfig{
		Location: "us-central1",
		Model:    "text-embedding-005",
		Timeout:  120 * time.Second, // Long timeout for large batches
	}
}

// VertexAIClient provides embedding generation via Google Vertex AI's API.
type VertexAIClient struct {
	projectID   string
	location    string
	model       string
	baseURL     string
	httpClient  *http.Client
	tokenSource oauth2.TokenSource
}

// vertexAIEmbedRequest represents the request to Vertex AI's prediction endpoint.
type vertexAIEmbedRequest struct {
	Instances []vertexAIInstance `json:"instances"`
}

type vertexAIInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitempty"`
}

// vertexAIEmbedResponse represents the response from Vertex AI's prediction endpoint.
type vertexAIEmbedResponse struct {
	Predictions []struct {
		Embeddings struct {
			Values []float32 `json:"values"`
		} `json:"embeddings"`
	} `json:"predictions"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

// NewVertexAIClient creates a new Vertex AI embedding client.
func NewVertexAIClient(cfg VertexAIConfig) (*VertexAIClient, error) {
	defaults := DefaultVertexAIConfig()

	if cfg.Location == "" {
		cfg.Location = defaults.Location
	}
	if cfg.Model == "" {
		cfg.Model = defaults.Model
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = defaults.Timeout
	}

	// Use background context for token source initialization
	ts, err := google.DefaultTokenSource(context.Background(), "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, &EmbeddingError{
			Code:       "AUTH_FAILED",
			Message:    "Failed to initialize Google Application Default Credentials",
			Suggestion: "Ensure you have authenticated via 'gcloud auth application-default login'",
			Cause:      err,
		}
	}

	return &VertexAIClient{
		projectID:   cfg.ProjectID,
		location:    cfg.Location,
		model:       cfg.Model,
		baseURL:     cfg.BaseURL,
		tokenSource: ts,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}, nil
}

// Health checks if the Vertex AI API is accessible with valid credentials.
func (c *VertexAIClient) Health(ctx context.Context) error {
	// Do a minimal embedding request to verify credentials
	_, err := c.EmbedSingle(ctx, "health check")
	return err
}

// EmbedSingle generates an embedding for a single text input.
func (c *VertexAIClient) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := c.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, &EmbeddingError{
			Code:    "EMBEDDING_EMPTY",
			Message: "Vertex AI returned no embeddings for the input",
		}
	}
	return embeddings[0], nil
}

// Embed generates embeddings for multiple texts in a single request.
// Vertex AI supports batching up to 250 instances per request.
func (c *VertexAIClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// Vertex AI limit is 250 instances per request
	const maxBatchSize = 250
	var allEmbeddings [][]float32

	for i := 0; i < len(texts); i += maxBatchSize {
		end := i + maxBatchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		embeddings, err := c.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

func (c *VertexAIClient) embedBatch(ctx context.Context, batch []string) ([][]float32, error) {
	instances := make([]vertexAIInstance, len(batch))
	for i, text := range batch {
		instances[i] = vertexAIInstance{
			Content:  text,
			TaskType: "RETRIEVAL_DOCUMENT", // Standard for code/docs
		}
	}

	reqBody := vertexAIEmbedRequest{
		Instances: instances,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := c.baseURL
	if url == "" {
		url = fmt.Sprintf("https://aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:predict",
			c.projectID, c.location, c.model)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Get OAuth2 token
	token, err := c.tokenSource.Token()
	if err != nil {
		return nil, &EmbeddingError{
			Code:       "TOKEN_ERROR",
			Message:    "Failed to get OAuth2 token",
			Suggestion: "Check your Google Cloud credentials",
			Cause:      err,
		}
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token.AccessToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, ErrProviderUnavailable.WithCause(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp, body)
	}

	var embedResp vertexAIEmbedResponse
	if err := json.Unmarshal(body, &embedResp); err != nil {
		return nil, &EmbeddingError{
			Code:    "INVALID_RESPONSE",
			Message: "received invalid response from Vertex AI",
			Cause:   err,
		}
	}

	if embedResp.Error != nil {
		return nil, &EmbeddingError{
			Code:    "API_ERROR",
			Message: embedResp.Error.Message,
		}
	}

	result := make([][]float32, len(embedResp.Predictions))
	for i, pred := range embedResp.Predictions {
		result[i] = pred.Embeddings.Values
	}

	return result, nil
}

func (c *VertexAIClient) handleErrorResponse(resp *http.Response, body []byte) error {
	var errResp vertexAIEmbedResponse
	json.Unmarshal(body, &errResp)

	msg := "unknown error"
	if errResp.Error != nil {
		msg = errResp.Error.Message
	}

	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return &EmbeddingError{
			Code:       "AUTH_FAILED",
			Message:    "Google Cloud authentication failed: " + msg,
			Suggestion: "Ensure your project ID is correct and you have Vertex AI User permissions",
			Retryable:  false,
		}

	case http.StatusTooManyRequests:
		return &EmbeddingError{
			Code:       "RATE_LIMITED",
			Message:    "Vertex AI rate limit exceeded: " + msg,
			Suggestion: "Waiting to retry automatically...",
			Retryable:  true,
		}

	case http.StatusBadRequest:
		return &EmbeddingError{
			Code:      "INVALID_REQUEST",
			Message:   "Invalid request to Vertex AI: " + msg,
			Retryable: false,
		}

	default:
		retryable := resp.StatusCode >= 500
		return &EmbeddingError{
			Code:      "REQUEST_FAILED",
			Message:   fmt.Sprintf("Vertex AI request failed with status %d: %s", resp.StatusCode, msg),
			Retryable: retryable,
		}
	}
}

// ModelName returns the configured model name.
func (c *VertexAIClient) ModelName() string {
	return c.model
}

// Dimensions returns the embedding dimension size.
func (c *VertexAIClient) Dimensions() int {
	return 768 // text-embedding-005
}

// Compile-time check that VertexAIClient implements Embedder
var _ Embedder = (*VertexAIClient)(nil)
