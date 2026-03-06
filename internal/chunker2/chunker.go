package chunker2

import (
	"context"

	"github.com/pommel-dev/pommel/internal/models"
)

// Chunker2 implements the models.Chunker interface using the new engine.
type Chunker2 struct {
	lang models.Language
}

// NewChunker2 creates a new Chunker2 instance.
func NewChunker2(lang models.Language) *Chunker2 {
	return &Chunker2{lang: lang}
}

// Chunk implements models.Chunker.
func (c *Chunker2) Chunk(ctx context.Context, file *models.SourceFile) (*models.ChunkResult, error) {
	// Skeleton implementation
	return &models.ChunkResult{
		File: file,
	}, nil
}

// Language implements models.Chunker.
func (c *Chunker2) Language() models.Language {
	return c.lang
}

// Registry manages the new chunker implementations.
type Registry struct {
	chunkers map[models.Language]models.Chunker
}

// NewRegistry creates a new Registry for chunker2.
func NewRegistry() *Registry {
	return &Registry{
		chunkers: make(map[models.Language]models.Chunker),
	}
}
