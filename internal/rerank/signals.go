package rerank

import (
	"strings"
	"time"
)

// Signal functions return a score boost/penalty in range [-0.2, 0.2]

// PathMatchSignal boosts results where the file path contains query terms.
// This is high-signal since developers often organize code by functionality in paths.
func PathMatchSignal(filePath string, queryTerms []string) float64 {
	if filePath == "" || len(queryTerms) == 0 {
		return 0
	}

	pathLower := strings.ToLower(filePath)
	matches := 0

	for _, term := range queryTerms {
		if strings.Contains(pathLower, strings.ToLower(term)) {
			matches++
		}
	}

	if matches == 0 {
		return 0
	}

	// More matches = higher boost, capped at 0.2
	boost := float64(matches) * 0.1
	if boost > 0.2 {
		boost = 0.2
	}
	return boost
}

// ExactPhraseSignal boosts results containing the exact query phrase within the code content.
func ExactPhraseSignal(content string, query string) float64 {
	if query == "" {
		return 0
	}

	contentLower := strings.ToLower(content)
	queryLower := strings.ToLower(query)

	if strings.Contains(contentLower, queryLower) {
		return 0.15 // Significant boost for exact phrase match
	}
	return 0
}

// RecencyBoost slightly boosts recently modified files to prioritize active development.
func RecencyBoost(modTime time.Time, now time.Time) float64 {
	if modTime.IsZero() {
		return 0
	}

	// Check for future time (shouldn't happen but handle it)
	if modTime.After(now) {
		return 0
	}

	daysSince := now.Sub(modTime).Hours() / 24

	// Very recent (< 1 day): max boost
	if daysSince < 1 {
		return 0.1
	}

	// Recent (< 7 days): medium boost
	if daysSince < 7 {
		return 0.05
	}

	// Moderately recent (< 30 days): small boost
	if daysSince < 30 {
		return 0.02
	}

	// Old file: no boost
	return 0
}
