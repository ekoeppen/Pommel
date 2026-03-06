package chunker

import (
	"context"
	"fmt"
	"strings"

	"github.com/pommel-dev/pommel/internal/embedder"
	"github.com/pommel-dev/pommel/internal/models"
)

const (
	// DefaultMaxTokens is the target token count for sliding window chunks.
	DefaultMaxTokens = 512
	// DefaultOverlapTokens is the number of tokens to overlap between chunks.
	DefaultOverlapTokens = 128
	// MinLinesPerChunk ensures chunks have some minimum content.
	MinLinesPerChunk = 5
)

// ChunkerRegistry handles the chunking of source files using a sliding window strategy.
type ChunkerRegistry struct{}

// NewChunkerRegistry creates a new ChunkerRegistry.
func NewChunkerRegistry() (*ChunkerRegistry, error) {
	return &ChunkerRegistry{}, nil
}

// Chunk implements models.Chunker.
func (r *ChunkerRegistry) Chunk(ctx context.Context, file *models.SourceFile) (*models.ChunkResult, error) {
	if file == nil {
		return nil, fmt.Errorf("file is required")
	}

	result := &models.ChunkResult{
		File:   file,
		Chunks: make([]*models.Chunk, 0),
		Errors: make([]error, 0),
	}

	content := string(file.Content)
	lines := strings.Split(content, "\n")
	totalLines := len(lines)
	if totalLines == 0 {
		totalLines = 1
	}

	// 1. Always add a file-level chunk
	fileChunk := &models.Chunk{
		FilePath:     file.Path,
		StartLine:    1,
		EndLine:      totalLines,
		Level:        models.ChunkLevelFile,
		Language:     file.Language,
		Content:      content,
		Name:         file.Path,
		LastModified: file.LastModified,
	}
	fileChunk.SetHashes()
	result.Chunks = append(result.Chunks, fileChunk)

	// 2. For larger files, create overlapping sections
	tokens := embedder.EstimateTokens(content)
	if tokens > DefaultMaxTokens {
		sections := splitIntoSections(file, lines, fileChunk.ID)
		result.Chunks = append(result.Chunks, sections...)
	}

	return result, nil
}

func splitIntoSections(file *models.SourceFile, lines []string, parentID string) []*models.Chunk {
	var chunks []*models.Chunk
	currentStart := 0

	// Target characters per chunk based on tokens
	targetChars := embedder.MaxCharsForTokens(DefaultMaxTokens)
	overlapChars := embedder.MaxCharsForTokens(DefaultOverlapTokens)

	for currentStart < len(lines) {
		// Find end line for this chunk based on character/token limit
		currentChars := 0
		currentEnd := currentStart

		for i := currentStart; i < len(lines); i++ {
			lineLen := len(lines[i]) + 1 // +1 for newline
			if currentChars+lineLen > targetChars && i > currentStart+MinLinesPerChunk {
				break
			}
			currentChars += lineLen
			currentEnd = i + 1
		}

		// Extract chunk content
		chunkLines := lines[currentStart:currentEnd]
		chunkContent := strings.Join(chunkLines, "\n")

		// Create chunk
		section := &models.Chunk{
			FilePath:     file.Path,
			StartLine:    currentStart + 1,
			EndLine:      currentEnd,
			Level:        models.ChunkLevelSection,
			Language:     file.Language,
			Content:      chunkContent,
			Name:         fmt.Sprintf("%s (lines %d-%d)", file.Path, currentStart+1, currentEnd),
			ParentID:     &parentID,
			LastModified: file.LastModified,
		}
		section.SetHashes()
		chunks = append(chunks, section)

		if currentEnd >= len(lines) {
			break
		}

		// Calculate overlap for next chunk
		nextStart := currentEnd
		backChars := 0
		for i := currentEnd - 1; i > currentStart; i-- {
			backChars += len(lines[i]) + 1
			if backChars >= overlapChars {
				nextStart = i
				break
			}
		}

		// Ensure we always make progress
		if nextStart <= currentStart {
			nextStart = currentEnd
		}

		currentStart = nextStart
	}

	return chunks
}

// GetChunkerForExtension exists for backward compatibility but is no longer needed.
func (r *ChunkerRegistry) GetChunkerForExtension(ext string) (models.Chunker, bool) {
	return &LegacyChunkerWrapper{registry: r}, true
}

// LegacyChunkerWrapper wraps the registry to implement the models.Chunker interface.
type LegacyChunkerWrapper struct {
	registry *ChunkerRegistry
}

func (w *LegacyChunkerWrapper) Chunk(ctx context.Context, file *models.SourceFile) (*models.ChunkResult, error) {
	return w.registry.Chunk(ctx, file)
}

func (w *LegacyChunkerWrapper) Language() models.Language {
	return models.LangUnknown
}
