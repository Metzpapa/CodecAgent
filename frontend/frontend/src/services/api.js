// frontend/src/services/api.js

// This file centralizes all API communication with the FastAPI backend.

// Get the base URL for the API from environment variables.
// This makes it easy to switch between development (localhost) and production.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

if (!API_BASE_URL) {
    console.error("VITE_API_BASE_URL is not set. Please check your .env file.");
}

/**
 * A helper function to handle common fetch logic, including error handling.
 * @param {string} endpoint - The API endpoint to call (e.g., '/jobs').
 * @param {object} options - The options object for the fetch call (method, headers, body).
 * @returns {Promise<any>} - A promise that resolves with the JSON response.
 * @throws {Error} - Throws an error if the network response is not ok.
 */
const apiFetch = async (endpoint, options = {}) => {
    const url = `${API_BASE_URL}${endpoint}`;

    try {
        const response = await fetch(url, options);

        // If the response is not in the 200-299 range, it's an error.
        if (!response.ok) {
            // Try to parse the error message from the backend's JSON response.
            const errorData = await response.json().catch(() => ({ detail: "An unknown error occurred." }));
            const errorMessage = errorData.detail || `HTTP error! status: ${response.status}`;
            // Create a custom error object to provide more context.
            const error = new Error(errorMessage);
            error.status = response.status;
            throw error;
        }

        // If the response is successful but has no content (e.g., 204 No Content), return null.
        if (response.status === 204) {
            return null;
        }

        // Otherwise, parse and return the JSON body.
        return response.json();

    } catch (error) {
        console.error(`API call to ${endpoint} failed:`, error);
        // Re-throw the error so the calling component can handle it (e.g., show a notification).
        throw error;
    }
};

/**
 * Fetches the list of all jobs for the currently authenticated user.
 * @param {string} token - The user's Google ID token (JWT).
 * @returns {Promise<Array>} - A promise that resolves to an array of job objects.
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
 * Fetches the real-time status of a single job from the Celery backend.
 * @param {string} jobId - The ID of the job to check.
 * @param {string} token - The user's Google ID token (JWT).
 * @returns {Promise<object>} - A promise that resolves to the job status object.
 */
export const getJobStatus = (jobId, token) => {
    return apiFetch(`/jobs/${jobId}/status`, {
        method: 'GET',
        headers: {
            'Authorization': `Bearer ${token}`,
        },
    });
};

/**
 * Creates a new editing job by uploading files and a prompt.
 * @param {string} prompt - The user's natural language prompt.
 * @param {File[]} files - An array of File objects to upload.
 * @param {string} token - The user's Google ID token (JWT).
 * @returns {Promise<object>} - A promise that resolves to the newly created job's info.
 */
export const createJob = async (prompt, files, token) => {
    const formData = new FormData();
    formData.append('prompt', prompt);

    if (files && files.length > 0) {
        files.forEach(file => {
            // The third argument is the filename, which is important for the backend.
            formData.append('files', file, file.name);
        });
    } else {
        // The backend endpoint requires at least one file.
        // We can handle this here to provide a better error message.
        return Promise.reject(new Error("At least one file must be provided to start a job."));
    }

    // Note: When using fetch with FormData, we DO NOT manually set the
    // 'Content-Type' header. The browser does this automatically and includes
    // the necessary 'boundary' parameter for multipart data.
    return apiFetch('/jobs', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
        },
        body: formData,
    });
};

/**
 * Constructs the URL for downloading a completed job's output file.
 * This doesn't make an API call, it just builds the correct URL string.
 * @param {string} jobId - The ID of the job to download.
 * @returns {string} - The full URL for the download endpoint.
 */
export const getDownloadUrl = (jobId) => {
    // Note: We don't include the token here because file downloads via an <a> tag
    // or window.location cannot easily have custom headers. A more secure production
    // system might use short-lived, single-use download tokens passed as a query parameter.
    // For our prototype, relying on the browser's session/cookie (if we had one) or
    // simply having a temporarily open endpoint is acceptable. The backend *will* still
    // need to handle auth for this endpoint, but we can't send the Bearer token this way.
    //
    // A better approach for a stateless app:
    // 1. Frontend calls a new endpoint GET /jobs/{job_id}/download-token?token=...
    // 2. Backend verifies the user token, generates a short-lived (e.g., 30s) one-time-use token.
    // 3. Backend returns the URL: /jobs/{job_id}/download?download_token=...
    // 4. Frontend opens this URL. The backend verifies the download_token.
    // For this prototype, we will assume the user is authenticated for the download request.
    // The simplest way is to open the URL and let the browser handle it, but it won't send the auth header.
    // Let's stick to the simple URL and address auth on download later if needed.
    return `${API_BASE_URL}/jobs/${jobId}/download`;
};