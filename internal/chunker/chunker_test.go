package chunker

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/pommel-dev/pommel/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChunkerRegistry(t *testing.T) {
	registry, err := NewChunkerRegistry()
	require.NoError(t, err)
	require.NotNil(t, registry)

	// Create a long source file to trigger section splitting
	// (more than 512 tokens / ~1800 chars)
	var sb strings.Builder
	sb.WriteString("package test\n\n")
	for i := 0; i < 200; i++ {
		sb.WriteString("// This is a long comment to fill space and trigger splitting into multiple overlapping sections.\n")
	}
	source := sb.String()

	file := &models.SourceFile{
		Path:         "test.go",
		Content:      []byte(source),
		Language:     "go",
		LastModified: time.Now(),
	}

	result, err := registry.Chunk(context.Background(), file)
	require.NoError(t, err)
	require.NotNil(t, result)

	// We expect 1 file chunk and several overlapping section chunks
	assert.Greater(t, len(result.Chunks), 1, "Should have more than just the file chunk")
	
	fileChunkCount := 0
	sectionChunkCount := 0
	for _, c := range result.Chunks {
		if c.Level == models.ChunkLevelFile {
			fileChunkCount++
		} else if c.Level == models.ChunkLevelSection {
			sectionChunkCount++
		}
	}

	assert.Equal(t, 1, fileChunkCount, "Should have exactly 1 file-level chunk")
	assert.Greater(t, sectionChunkCount, 0, "Should have at least one section chunk")

	// Verify the first section starts at line 1
	assert.Equal(t, 1, result.Chunks[1].StartLine)
	
	// Verify that the ID of the file chunk is set as the ParentID for the sections
	for i := 1; i < len(result.Chunks); i++ {
		assert.Equal(t, result.Chunks[0].ID, *result.Chunks[i].ParentID)
	}
}
