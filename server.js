import express from 'express';
import dotenv from 'dotenv';
import OpenAI, { toFile } from 'openai';
import cors from 'cors';
import multer from 'multer';

dotenv.config({ override: true });

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_RESPONSES_MODEL = process.env.OPENAI_RESPONSES_MODEL || 'gpt-4.1';
const VECTOR_STORE_ID_1 = process.env.VECTOR_STORE_ID_1 || null;
const VECTOR_STORE_ID_2 = process.env.VECTOR_STORE_ID_2 || null;
const VECTOR_STORE_ID_3 = process.env.VECTOR_STORE_ID_3 || null;

if (!OPENAI_API_KEY) {
  console.error('Environment variable "OPENAI_API_KEY" is required.');
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const upload = multer({ storage: multer.memoryStorage() });

const PORT = parseInt(process.env.PORT ?? '', 10) || 8080;
const CORS_ALLOW_ORIGINS = (process.env.CORS_ALLOW_ORIGINS || '*')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

// Assistant instructions for the chatbot
const ASSISTANT_INSTRUCTIONS = `You are a hiring evaluation assistant. Use the uploaded job description and candidate résumé to answer questions, surface evidence, and highlight risks. You can also answer general questions about the job or résumé.
Follow these instructions carefully:
- Whenever the user references the job, role requirements, or the candidate, call the file_search tool first and wait for its response before replying. Retrieve context from both job-description and résumé files; never guess without searching.
- If the user requests a full fit assessment, structure the reply as: Summary, Strengths, Concerns/Risks, Recommendation (e.g., Strong Fit / Mixed Fit / Not Fit) with a one-line rationale.
- For other questions about the documents (e.g., "What skills does the candidate list?"), provide a concise answer grounded in the retrieved snippets. You may include brief bullet points when helpful.
- If the user asks something unrelated to the uploaded documents, you may answer directly without using file_search.
- Keep responses concise (under 120 words) and professional.
- If searches return nothing relevant, say: ➔ "I couldn't find enough information in the uploaded job description or résumé to answer that."
- If more than one resume uploaded for one job description, roles and responsibilities, then you have to find the best candidate and rank them according to the best fit to the last.
- If more than one resume uploaded and many job description uploaded as well, for many job description, roles and responsibilities, then you find the best candidate fit one candidate for each job description.`;

const FILE_COLLECTIONS = {
  primary: createCollection({
    key: 'primary',
    label: 'Job Description',
    vectorStoreName: 'primary-knowledge-base',
    envVar: 'VECTOR_STORE_ID_1',
    initialVectorStoreId: VECTOR_STORE_ID_1,
  }),
  secondary: createCollection({
    key: 'secondary',
    label: 'Candidate Resumes',
    vectorStoreName: 'secondary-knowledge-base',
    envVar: 'VECTOR_STORE_ID_2',
    initialVectorStoreId: VECTOR_STORE_ID_2,
  }),
  training: createCollection({
    key: 'training',
    label: 'Training Programs',
    vectorStoreName: 'training-knowledge-base',
    envVar: 'VECTOR_STORE_ID_3',
    initialVectorStoreId: VECTOR_STORE_ID_3,
  }),
};

const COLLECTION_LIST = Object.values(FILE_COLLECTIONS);

const app = express();
app.use(
  cors({
    origin: CORS_ALLOW_ORIGINS.includes('*') ? '*' : CORS_ALLOW_ORIGINS,
    methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  }),
);
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.get('/file-collections', (_req, res) => {
  res.json({
    collections: COLLECTION_LIST.map((collection) => ({
      key: collection.key,
      label: collection.label,
      vector_store_id: collection.vectorStoreId ?? null,
    })),
  });
});

app.get('/files/:collectionKey', async (req, res) => {
  const collection = getCollection(req.params.collectionKey);
  if (!collection) {
    return res.status(404).json({ error: `Unknown file collection "${req.params.collectionKey}".` });
  }

  if (!collection.vectorStoreId) {
    return res.json({ files: [] });
  }

  try {
    const files = await fetchVectorStoreFiles(collection.vectorStoreId);
    const payload = (await Promise.all(
      files
        .filter((file) => file.status === 'completed')
        .map((file) => buildFileResponse(collection.vectorStoreId, file)),
    )).filter(Boolean);
    payload.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    res.json({ files: payload });
  } catch (error) {
    console.error(`Error listing files for ${collection.key}:`, error);
    res.status(500).json({ error: 'Failed to list files.' });
  }
});

app.post('/files/:collectionKey', upload.single('file'), async (req, res) => {
  const collection = getCollection(req.params.collectionKey);
  if (!collection) {
    return res.status(404).json({ error: `Unknown file collection "${req.params.collectionKey}".` });
  }

  const incomingFile = req.file;
  if (!incomingFile) {
    return res.status(400).json({ error: 'A file must be provided.' });
  }

  try {
    const vectorStoreId = await ensureVectorStoreForCollection(collection);
    const openaiFile = await openai.files.create({
      file: await toFile(incomingFile.buffer, incomingFile.originalname || 'upload.dat'),
      purpose: 'assistants',
    });

    const vectorStoreFile = await openai.vectorStores.files.create(vectorStoreId, {
      file_id: openaiFile.id,
    });

    const payload = await buildFileResponse(vectorStoreId, vectorStoreFile);
    res.status(201).json({ file: payload });
  } catch (error) {
    console.error(`Error uploading file to ${collection.key}:`, error);
    res.status(500).json({ error: 'Failed to upload file.' });
  }
});

app.delete('/files/:collectionKey/:fileId', async (req, res) => {
  const collection = getCollection(req.params.collectionKey);
  if (!collection) {
    return res.status(404).json({ error: `Unknown file collection "${req.params.collectionKey}".` });
  }

  if (!collection.vectorStoreId) {
    return res.status(404).json({ error: 'No vector store configured for this collection yet.' });
  }

  const { fileId } = req.params;

  try {
    try {
      await openai.vectorStores.files.del(collection.vectorStoreId, fileId);
    } catch (err) {
      if (err.status !== 404 && err.statusCode !== 404) {
        throw err;
      }
    }

    try {
      await openai.files.del(fileId);
    } catch (err) {
      if (err.status !== 404 && err.statusCode !== 404) {
        console.warn(`[Files] Unable to delete OpenAI file ${fileId}:`, err.message);
      }
    }

    res.json({ ok: true });
  } catch (error) {
    console.error(`Error deleting file ${fileId} from ${collection.key}:`, error);
    res.status(500).json({ error: 'Failed to delete file.' });
  }
});

app.post('/chatbot', async (req, res) => {
  const message = req.body?.message;
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Property "message" is required.' });
  }

  const vectorStoreIds = getActiveVectorStoreIds();

  try {
    const response = await openai.responses.create({
      model: OPENAI_RESPONSES_MODEL,
      input: message,
      instructions: ASSISTANT_INSTRUCTIONS,
      ...(vectorStoreIds.length
        ? {
            tools: [
              {
                type: 'file_search',
                vector_store_ids: vectorStoreIds,
              },
            ],
          }
        : {}),
    });

    const text = extractResponseText(response);
    res.json({
      response: text,
      vector_store_ids: vectorStoreIds,
    });
  } catch (error) {
    console.error('Error handling /chatbot request:', error);
    const message = error?.message || 'Failed to process request';
    res.status(500).json({ error: message });
  }
});

app.post('/training/from-file', async (req, res) => {
  const { jobDescription, candidateInfo } = req.body;

  if (!jobDescription || !candidateInfo) {
    return res.status(400).json({ error: 'Both jobDescription and candidateInfo are required.' });
  }

  const trainingCollection = FILE_COLLECTIONS.training;
  if (!trainingCollection.vectorStoreId) {
    return res.status(400).json({ 
      error: 'No training file uploaded. Please upload a training programs file first.' 
    });
  }

  const vectorStoreIds = getTrainingVectorStoreIds();

  try {
    const prompt = `You are a training recommendation expert.

TASK: Find suitable training programs for a candidate based on job requirements and available training options.

JOB POSITION TO ANALYZE: "${jobDescription}"
CANDIDATE TO ANALYZE: "${candidateInfo}"

Instructions:
1. Search the uploaded job descriptions to find the job matching "${jobDescription}"
2. Search the uploaded resumes to find the candidate matching "${candidateInfo}"
3. Search the uploaded training programs file to find all available training options
4. Analyze the skill gaps between what the job requires and what the candidate currently has
5. Select the most suitable training programs from the training file that would help close these gaps

Return ONLY a valid JSON object with this exact structure:
{
  "recommendations": [
    {
      "id": "unique-id-from-file",
      "name": "Training Program Name (exactly as it appears in the training file)",
      "description": "Brief description of what the training covers",
      "duration": "Duration if specified in the file, otherwise 'TBD'",
      "reason": "Specific reason why this training addresses the candidate's skill gap for this particular job"
    }
  ]
}

Important:
- Only recommend programs that exist in the uploaded training file
- Select 3-5 most relevant programs
- If no suitable programs found in the file, return empty recommendations array
- Base reasons on actual skill gaps between job requirements and candidate profile`;

    const response = await openai.responses.create({
      model: OPENAI_RESPONSES_MODEL,
      input: prompt,
      instructions: 'You are a training recommendation assistant. You must search all uploaded files (job descriptions, resumes, and training programs) to provide accurate recommendations. Always respond with valid JSON only.',
      tools: [
        {
          type: 'file_search',
          vector_store_ids: vectorStoreIds,
        },
      ],
    });

    const text = extractResponseText(response);
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
      const result = JSON.parse(jsonMatch[0]);
      res.json(result);
    } else {
      res.json({ recommendations: [] });
    }
  } catch (error) {
    console.error('Error generating recommendations from file:', error);
    res.status(500).json({ error: 'Failed to generate recommendations.' });
  }
});

app.post('/training/ai-generate', async (req, res) => {
  const { jobDescription, candidateInfo } = req.body;

  if (!jobDescription || !candidateInfo) {
    return res.status(400).json({ error: 'Both jobDescription and candidateInfo are required.' });
  }

  const vectorStoreIds = getActiveVectorStoreIds();

  try {
    const prompt = `You are a training recommendation expert.

TASK: Generate personalized training recommendations for a candidate based on job requirements.

JOB POSITION TO ANALYZE: "${jobDescription}"
CANDIDATE TO ANALYZE: "${candidateInfo}"

Instructions:
1. Search the uploaded job descriptions to find the job matching "${jobDescription}"
2. Search the uploaded resumes to find the candidate matching "${candidateInfo}"
3. Carefully analyze the skill gaps between:
   - What the job requires (skills, experience, qualifications)
   - What the candidate currently has (skills, experience, education)
4. Generate practical, industry-standard training recommendations that would help the candidate succeed in this role

Return ONLY a valid JSON object with this exact structure:
{
  "recommendations": [
    {
      "id": "ai-unique-id",
      "name": "Training Program Name",
      "description": "What the candidate will learn and why it's valuable for this role",
      "duration": "Realistic duration (e.g., '2 weeks', '1 month', '40 hours')",
      "reason": "Specific skill gap this training addresses - be specific about what the job needs vs what the candidate has"
    }
  ]
}

Important:
- Generate 3-5 highly relevant training recommendations
- Be specific and practical - recommend real-world training topics
- Focus on the actual skill gaps identified from the documents
- Each recommendation should directly address a gap between job requirements and candidate abilities`;

    const response = await openai.responses.create({
      model: OPENAI_RESPONSES_MODEL,
      input: prompt,
      instructions: 'You are a training recommendation assistant. You must search uploaded job descriptions and resumes to understand requirements and candidate profile. Generate practical, actionable training recommendations. Always respond with valid JSON only.',
      ...(vectorStoreIds.length
        ? {
            tools: [
              {
                type: 'file_search',
                vector_store_ids: vectorStoreIds,
              },
            ],
          }
        : {}),
    });

    const text = extractResponseText(response);
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
      const result = JSON.parse(jsonMatch[0]);
      res.json(result);
    } else {
      res.json({ recommendations: [] });
    }
  } catch (error) {
    console.error('Error generating AI recommendations:', error);
    res.status(500).json({ error: 'Failed to generate recommendations.' });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});

function createCollection({ key, label, vectorStoreName, envVar, initialVectorStoreId }) {
  return {
    key,
    label,
    vectorStoreName,
    vectorStoreId: initialVectorStoreId || null,
    envVar,
  };
}

function getCollection(key) {
  return key ? FILE_COLLECTIONS[key] ?? null : null;
}

async function ensureVectorStoreForCollection(collection) {
  if (collection.vectorStoreId) {
    return collection.vectorStoreId;
  }

  const store = await openai.vectorStores.create({
    name: collection.vectorStoreName,
  });
  collection.vectorStoreId = store.id;
  console.log(
    `[FileUpload] Created vector store ${store.id} for collection "${collection.key}". Set ${collection.envVar} to reuse.`,
  );
  return collection.vectorStoreId;
}

async function fetchVectorStoreFiles(vectorStoreId) {
  const files = [];
  for await (const file of openai.vectorStores.files.list(vectorStoreId)) {
    files.push(file);
  }
  return files;
}

async function buildFileResponse(vectorStoreId, vectorStoreFile) {
  if (!vectorStoreFile) {
    return null;
  }

  let openaiFile = null;
  try {
    openaiFile = await openai.files.retrieve(vectorStoreFile.id);
  } catch (error) {
    if (error?.status === 404 || error?.statusCode === 404) {
      console.warn(
        `[Files] Removing orphaned entry ${vectorStoreFile.id} from vector store ${vectorStoreId}`,
      );
      try {
        await openai.vectorStores.files.del(vectorStoreId, vectorStoreFile.id);
      } catch (cleanupError) {
        console.warn(
          `[Files] Failed to delete orphaned entry ${vectorStoreFile.id}:`,
          cleanupError.message,
        );
      }
      return null;
    }
    console.warn(`[Files] Unable to load metadata for ${vectorStoreFile.id}:`, error.message);
  }

  return {
    id: vectorStoreFile.id,
    status: vectorStoreFile.status,
    usage_bytes: vectorStoreFile.usage_bytes,
    created_at: vectorStoreFile.created_at,
    vector_store_id: vectorStoreFile.vector_store_id,
    last_error: vectorStoreFile.last_error ?? null,
    filename: openaiFile?.filename ?? null,
    bytes: openaiFile?.bytes ?? null,
    mime_type: openaiFile?.mime_type ?? null,
    purpose: openaiFile?.purpose ?? null,
  };
}

function getActiveVectorStoreIds() {
  const ids = [];
  for (const collection of COLLECTION_LIST) {
    if (collection.vectorStoreId && !ids.includes(collection.vectorStoreId)) {
      ids.push(collection.vectorStoreId);
    }
  }
  if (ids.length < 2 && VECTOR_STORE_ID_1 && !ids.includes(VECTOR_STORE_ID_1)) {
    ids.push(VECTOR_STORE_ID_1);
  }
  if (ids.length < 2 && VECTOR_STORE_ID_2 && !ids.includes(VECTOR_STORE_ID_2)) {
    ids.push(VECTOR_STORE_ID_2);
  }
  return ids.slice(0, 2);
}

/**
 * Get vector store IDs for training recommendations.
 * Prioritizes training collection + secondary (resumes).
 * Limited to 2 due to OpenAI API constraint.
 */
function getTrainingVectorStoreIds() {
  const ids = [];
  // Prioritize training and secondary (resumes) for training recommendations
  const priority = ['training', 'secondary', 'primary'];
  for (const key of priority) {
    const collection = FILE_COLLECTIONS[key];
    if (collection?.vectorStoreId && !ids.includes(collection.vectorStoreId)) {
      ids.push(collection.vectorStoreId);
    }
  }
  return ids.slice(0, 2); // OpenAI allows max 2 vector stores
}

function extractResponseText(response) {
  if (!response) {
    throw new Error('Empty response from OpenAI');
  }

  if (typeof response.output_text === 'string' && response.output_text.trim()) {
    return response.output_text.trim();
  }

  const outputs = Array.isArray(response.output) ? response.output : [];
  for (const item of outputs) {
    if (item?.type === 'output_text') {
      if (typeof item.text === 'string' && item.text.trim()) {
        return item.text.trim();
      }
      if (item.text && typeof item.text.value === 'string' && item.text.value.trim()) {
        return item.text.value.trim();
      }
    }

    if (item?.type === 'message') {
      const contentItems = Array.isArray(item.content) ? item.content : [];
      for (const content of contentItems) {
        if (content?.type === 'output_text') {
          const value = content.text?.value ?? content.text;
          if (typeof value === 'string' && value.trim()) {
            return value.trim();
          }
        }
      }
    }
  }

  throw new Error('No text output returned from OpenAI response');
}
