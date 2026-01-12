/**
 * Shared chat functionality for consistent message rendering across templates.
 */

// Source label configuration - consistent across all templates
const SOURCE_LABELS = {
    dataset: 'Dataset Answer',
    uploaded: 'Uploaded File Answer',
    llm: 'Generated Answer',
    uncertain: 'Low Confidence'
};

// Source label CSS classes - consistent colors
const SOURCE_CLASSES = {
    dataset: 'source-tag dataset',
    uploaded: 'source-tag uploaded',
    llm: 'source-tag llm',
    uncertain: 'source-tag uncertain'
};

// Dataset source display names
const DATASET_SOURCE_NAMES = {
    main_dataset: 'Main Curated Dataset',
    uploaded_files: 'Uploaded Files',
    llm_fallback: 'LLM Fallback',
    none: 'No Source'
};

/**
 * Render a chat message with consistent formatting.
 * @param {HTMLElement} container - Container element to append message to
 * @param {string} text - Message text
 * @param {string} type - 'user' or 'bot'
 * @param {string} source - Source label (dataset, uploaded, llm, uncertain)
 * @param {Object} metadata - Additional metadata (dataset_source, similarity_score, is_grounded)
 */
function renderMessage(container, text, type = 'bot', source = '', metadata = {}) {
    const wrapper = document.createElement('div');
    wrapper.className = type === 'user' ? 'user-msg' : 'bot-msg';

    const header = document.createElement('div');
    header.className = 'msg-header';

    const name = document.createElement('strong');
    name.textContent = type === 'user' ? 'You' : 'Tutor';
    header.appendChild(name);

    if (type === 'bot') {
        // Add source tag
        if (source) {
            const sourceTag = document.createElement('span');
            sourceTag.className = SOURCE_CLASSES[source] || 'source-tag';
            sourceTag.textContent = SOURCE_LABELS[source] || source;
            header.appendChild(sourceTag);
        }

        // Add dataset source indicator if available
        if (metadata.dataset_source && metadata.dataset_source !== 'none') {
            const datasetIndicator = document.createElement('span');
            datasetIndicator.className = 'dataset-indicator';
            datasetIndicator.textContent = `From: ${DATASET_SOURCE_NAMES[metadata.dataset_source] || metadata.dataset_source}`;
            header.appendChild(datasetIndicator);
        }

        // Add similarity score if available
        if (metadata.similarity_score !== undefined && metadata.similarity_score > 0) {
            const scoreIndicator = document.createElement('span');
            scoreIndicator.className = 'similarity-score';
            scoreIndicator.textContent = `Similarity: ${metadata.similarity_score.toFixed(2)}`;
            header.appendChild(scoreIndicator);
        }
    }

    const body = document.createElement('div');
    body.className = 'msg-body';
    
    if (type === 'bot') {
        // Render markdown for bot messages
        if (typeof marked !== 'undefined') {
            try {
                const rawHtml = marked.parse(text || '');
                if (typeof DOMPurify !== 'undefined') {
                    body.innerHTML = DOMPurify.sanitize(rawHtml);
                } else {
                    body.innerHTML = rawHtml;
                }
            } catch (err) {
                body.textContent = text || '';
            }
        } else {
            body.textContent = text || '';
        }
    } else {
        body.textContent = text || '';
    }

    wrapper.appendChild(header);
    wrapper.appendChild(body);

    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}

/**
 * Handle file upload with validation and feedback.
 * @param {File} file - File object to upload
 * @returns {Promise<Object>} Upload result
 */
async function uploadFile(file) {
    // Validate file type
    const supportedTypes = ['.txt', '.csv', '.md', '.markdown'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!supportedTypes.includes(fileExt)) {
        return {
            success: false,
            error: `Unsupported file type. Supported types: ${supportedTypes.join(', ')}`
        };
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        return {
            success: false,
            error: 'File size exceeds 16MB limit'
        };
    }

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            return {
                success: true,
                message: data.message || 'File uploaded successfully',
                filename: data.filename
            };
        } else {
            return {
                success: false,
                error: data.error || 'Upload failed'
            };
        }
    } catch (err) {
        return {
            success: false,
            error: 'Network error: ' + err.message
        };
    }
}

/**
 * Send a chat message and render the response.
 * @param {string} message - User message
 * @param {HTMLElement} chatBox - Chat container element
 * @param {HTMLInputElement} input - Input element
 * @param {boolean} useUploaded - Whether to prioritize uploaded files
 */
async function sendMessage(message, chatBox, input, useUploaded = false) {
    if (!message.trim()) return;

    // Render user message
    renderMessage(chatBox, message, 'user');
    input.value = '';

    try {
        const formData = new URLSearchParams({
            'message': message,
            'use_uploaded': useUploaded.toString()
        });

        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        const data = await response.json();

        // Render bot response with metadata
        renderMessage(chatBox, data.response || '', 'bot', data.source || 'uncertain', {
            dataset_source: data.dataset_source,
            similarity_score: data.similarity_score || 0,
            is_grounded: data.is_grounded || false
        });
    } catch (err) {
        renderMessage(chatBox, 'Unable to reach the tutor right now. Please try again shortly.', 'bot', 'uncertain');
    }
}


