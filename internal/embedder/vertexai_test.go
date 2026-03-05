package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

type mockTokenSource struct {
	token *oauth2.Token
}

func (m *mockTokenSource) Token() (*oauth2.Token, error) {
	return m.token, nil
}

func TestVertexAIClient_EmbedSingle(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "Bearer fake-token", r.Header.Get("Authorization"))

		var req vertexAIEmbedRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Len(t, req.Instances, 1)
		assert.Equal(t, "test text", req.Instances[0].Content)

		resp := vertexAIEmbedResponse{
			Predictions: []struct {
				Embeddings struct {
					Values []float32 `json:"values"`
				} `json:"embeddings"`
			}{
				{
					Embeddings: struct {
						Values []float32 `json:"values"`
					}{
						Values: []float32{0.1, 0.2, 0.3},
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := &VertexAIClient{
		projectID: "test-project",
		location:  "us-central1",
		model:     "text-embedding-004",
		baseURL:   server.URL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		tokenSource: &mockTokenSource{
			token: &oauth2.Token{AccessToken: "fake-token"},
		},
	}

	embeddings, err := client.EmbedSingle(context.Background(), "test text")
	require.NoError(t, err)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, embeddings)
}

func TestVertexAIClient_Embed_Batching(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var req vertexAIEmbedRequest
		json.NewDecoder(r.Body).Decode(&req)

		predictions := make([]struct {
			Embeddings struct {
				Values []float32 `json:"values"`
			} `json:"embeddings"`
		}, len(req.Instances))

		for i := range req.Instances {
			predictions[i].Embeddings.Values = []float32{float32(i)}
		}

		resp := vertexAIEmbedResponse{
			Predictions: predictions,
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := &VertexAIClient{
		projectID: "test-project",
		location:  "us-central1",
		model:     "text-embedding-004",
		baseURL:   server.URL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		tokenSource: &mockTokenSource{
			token: &oauth2.Token{AccessToken: "fake-token"},
		},
	}

	// Test batching with 300 texts (should be 2 requests: 250 + 50)
	texts := make([]string, 300)
	for i := range texts {
		texts[i] = "text"
	}

	embeddings, err := client.Embed(context.Background(), texts)
	require.NoError(t, err)
	assert.Len(t, embeddings, 300)
	assert.Equal(t, 2, callCount)
}
