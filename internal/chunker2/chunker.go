package chunker2

import (
	"context"
	"fmt"
	"strings"

	"github.com/pommel-dev/pommel/internal/embedder"
	"github.com/pommel-dev/pommel/internal/models"
)

const (
	// DefaultMaxTokensPerChunk is the target token count for sliding window chunks.
	DefaultMaxTokensPerChunk = 512
	// DefaultOverlapTokens is the number of tokens to overlap between chunks.
	DefaultOverlapTokens = 128
	// MinLinesPerChunk ensures chunks have some minimum content.
	MinLinesPerChunk = 5
)

// Chunker provides a robust line-and-token-based chunking strategy.
// It generates a file-level chunk and multiple overlapping sections for all files.
type Chunker struct {
	lang models.Language
}

// NewChunker creates a new Chunker.
func NewChunker(lang models.Language) *Chunker {
	return &Chunker{lang: lang}
}

// Language implements models.Chunker.
func (c *Chunker) Language() models.Language {
	return c.lang
}

// Chunk implements models.Chunker.
func (c *Chunker) Chunk(ctx context.Context, file *models.SourceFile) (*models.ChunkResult, error) {
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
		Language:     string(c.lang),
		Content:      content,
		Name:         file.Path,
		LastModified: file.LastModified,
	}
	fileChunk.SetHashes()
	result.Chunks = append(result.Chunks, fileChunk)

	// 2. For larger files, create overlapping sections
	tokens := embedder.EstimateTokens(content)
	if tokens > DefaultMaxTokensPerChunk {
		sections := splitIntoSections(file, lines, fileChunk.ID)
		result.Chunks = append(result.Chunks, sections...)
	}

	return result, nil
}

func splitIntoSections(file *models.SourceFile, lines []string, parentID string) []*models.Chunk {
	var chunks []*models.Chunk
	currentStart := 0

	// Target characters per chunk based on tokens
	targetChars := embedder.MaxCharsForTokens(DefaultMaxTokensPerChunk)
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

// Registry manages the chunker instances.
type Registry struct{}

// NewRegistry creates a new Registry.
func NewRegistry() *Registry {
	return &Registry{}
}

// Get returns the chunker for the given language.
// In this simplified engine, it always returns the same generic strategy.
func (r *Registry) Get(lang models.Language) (models.Chunker, bool) {
	return NewChunker(lang), true
}
