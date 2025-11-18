# API Documentation

## Base URL
`https://api.spectral-disruption-profiler.com/v1`

## Authentication
Use API key in header: `Authorization: Bearer <api_key>`

## Endpoints

### POST /score
Score gRNA sequences for disruption.

**Request Body**:
```json
{
  "sequences": ["ATCGATCG", "GCTAGCTA"],
  "k": 0.3,
  "bootstrap_samples": 1000
}
```

**Response**:
```json
{
  "results": [
    {
      "sequence": "ATCGATCG",
      "score": 0.85,
      "ci": [0.82, 0.88],
      "entropy": 0.12,
      "delta_f1": 0.05,
      "sidelobes": 3
    }
  ]
}
```

### GET /health
Health check.

**Response**: `{"status": "ok"}`

## Rate Limits
1000 requests per day for free tier, unlimited for premium.