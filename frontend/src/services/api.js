// frontend/src/services/api.js

// Centralized API client for the frontend.

// Resolve API base URL from Vite env. If empty/undefined, default to same-origin
// so requests like '/jobs' hit the current host (useful behind Nginx in prod).
const rawBase = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL = rawBase && rawBase.trim() !== '' ? rawBase : '';

if (API_BASE_URL === '') {
  // Same-origin mode (expected in containerized prod behind Nginx)
  console.info('Using same-origin API base for requests.');
}

/**
 * Build a full URL for an API endpoint.
 * When API_BASE_URL === '', this returns relative paths (e.g., '/jobs').
 */
const buildUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

/**
 * Common fetch wrapper with robust error handling:
 * - Builds full URL using API_BASE_URL
 * - On non-OK responses, attempts to parse JSON first; if not JSON, uses raw text
 * - Attaches status and url to the thrown Error for upstream handling
 */
const apiFetch = async (endpoint, options = {}) => {
  const url = buildUrl(endpoint);

  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      // Read body once and try to parse it as JSON; fallback to text/statusText
      let message = `HTTP ${response.status}`;
      try {
        const text = await response.text();
        if (text) {
          try {
            const data = JSON.parse(text);
            message = data.detail || data.error || data.message || message;
          } catch {
            // Not JSON; use raw text (trim to avoid giant HTML pages)
            message = text || response.statusText || message;
          }
        } else if (response.statusText) {
          message = response.statusText;
        }
      } catch {
        // Ignore body read errors; keep default message
      }

      const error = new Error(message);
      error.status = response.status;
      error.url = url;
      throw error;
    }

    if (response.status === 204) {
      return null;
    }

    // Prefer JSON; fallback to text
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return response.json();
    }
    return response.text();

  } catch (error) {
    console.error(`API call to ${endpoint} failed:`, error);
    throw error;
  }
};

/**
 * Fetch the list of all jobs for the authenticated user.
 * @param {string} token - Google ID token (JWT)
 */
export const getJobs = (token) => {
  return apiFetch('/jobs', {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });
};

/**
 * Fetch real-time status of a single job.
 * @param {string} jobId
 * @param {string} token
 */
export const getJobStatus = (jobId, token) => {
  return apiFetch(`/jobs/${jobId}/status`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });
};

// --- NEW FUNCTION ---
/**
 * Fetch all details for a single job by its ID.
 * Used to populate the JobDetailPage.
 * @param {string} jobId
 * @param {string} token
 */
export const getJobById = (jobId, token) => {
  // This currently uses the same endpoint as getJobStatus, which returns
  // the full job object. If a dedicated endpoint is created later,
  // only this URL needs to be changed.
  return apiFetch(`/jobs/${jobId}/status`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });
};
// --- END NEW FUNCTION ---

/**
 * Create a new editing job by uploading files and a prompt.
 * @param {string} prompt
 * @param {File[]} files
 * @param {string} token
 */
export const createJob = async (prompt, files, token) => {
  const formData = new FormData();
  formData.append('prompt', prompt);

  if (files && files.length > 0) {
    files.forEach(file => {
      formData.append('files', file, file.name);
    });
  } else {
    return Promise.reject(new Error('At least one file must be provided to start a job.'));
  }

  // Do not set Content-Type manually for FormData; the browser will set it with boundary.
  return apiFetch('/jobs', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
    body: formData,
  });
};

/**
 * Construct the URL for downloading a completed job's output.
 * For authenticated downloads, call fetch with Authorization header manually (see JobItem.jsx).
 * @param {string} jobId
 */
export const getDownloadUrl = (jobId) => {
  return buildUrl(`/jobs/${jobId}/download`);
};