# Candidate Evaluation Backend

Minimal Express backend for the e-hospital candidate evaluation project.

## Requirements

- Node.js 18+

## Setup

```bash
npm install
```

Create a `.env` file (ignored by git) for any environment variables you need.

## Usage

```bash
npm start
```

The server entry point is `server.js`.

## Docker

Build the image:

```bash
docker build -t candidate-eval-backend .
```

Run with your `.env` values (not copied into the image):

```bash
docker run --env-file .env -p 8080:8080 candidate-eval-backend
```

