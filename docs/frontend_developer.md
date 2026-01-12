## Frontend Developer Documentation

The frontend is a **Flask‑rendered web UI** that exposes the AI tutor as an interactive chat experience with authentication and a clean, modern layout.

- **Templates**: `templates/base.html`, `templates/index.html`, `templates/login.html`, `templates/register.html`, `templates/chat.html`
- **Static assets**: `static/style.css`, `static/chat.js`
- **Backend interface**: Flask routes in `app.py`

### UI Architecture

- **Layout template (`base.html`)**
  - Provides the **global HTML skeleton**: `<head>`, meta tags, CSS/JS includes.
  - Defines reusable blocks for:
    - Navigation (login/register links, current user display).
    - Main content area (extended by individual pages).
  - Ensures visual and UX consistency across:
    - Landing page (`index.html`)
    - Authentication screens (`login.html`, `register.html`)
    - Chat workspace (`chat.html`)

- **Landing page (`index.html`)**
  - Presents the **project narrative** and key value proposition (AI tutor for Python/data science).
  - Contains calls‑to‑action like “Try the tutor” that route to the login or chat pages depending on session state.

- **Authentication views**
  - `login.html`:
    - Form fields: email, password.
    - Error messaging area (e.g., “Invalid credentials”).
  - `register.html`:
    - Form fields: name, email, password.
    - Error messaging for missing fields or duplicate email.

- **Chat interface (`chat.html`)**
  - Chat transcript area that shows alternating **user** and **assistant** messages.
  - Input box for the user’s question.
  - Optional toggle to indicate whether **uploaded files** should be used (`use_uploaded` flag).
  - Visual cues for message source (e.g., dataset vs LLM vs uncertain) can be added via CSS classes.

### Styling (`static/style.css`)

The CSS stylesheet defines a **minimal, responsive layout** oriented around reading and typing code‑related explanations.

- **Key goals**:
  - Maintain **high contrast** and readability for long text responses.
  - Provide distinct visual styling for:
    - System prompts / metadata (source labels, status indicators).
    - User vs assistant messages.
  - Ensure the layout degrades gracefully on smaller screens.

- **Typical components**:
  - Top navigation bar with branding and user/session state.
  - Card‑like container around the chat region.
  - Scrollable chat log with padding and subtle separators.
  - Prominent input area with clear call‑to‑action button.

Frontend developers can extend this stylesheet to:

- Add **syntax‑highlighted message blocks** for code.
- Visually differentiate messages based on `source` (dataset, llm, uploaded, uncertain).
- Integrate dark mode or theming without altering backend code.

### Client‑Side Logic (`static/chat.js`)

The JavaScript layer provides the dynamic behavior of the chat UI.

- **Core responsibilities**:
  - Capture user input from the chat form.
  - Send asynchronous **POST** requests to the Flask `/ask` endpoint.
  - Update the chat transcript DOM with:
    - User message.
    - Assistant response.
    - Optional metadata such as similarity score or dataset source.

- **Network protocol**:
  - Endpoint: `/ask`
  - Method: `POST`
  - Form fields:
    - `message`: user’s plain‑text question.
    - `use_uploaded` (optional): `"true"` or `"false"` to prefer uploaded documents.
  - Response payload (JSON, from `app.py`):
    - `response`: formatted answer text.
    - `source`: `"dataset" | "uploaded" | "llm" | "uncertain"`.
    - `dataset_source`: name of the dataset (`"main_dataset"` or `"uploaded_files"`).
    - `similarity_score`: float similarity measure used for retrieval.
    - `is_grounded`: boolean flag indicating if the answer is grounded in retrieved context.

- **Error handling**:
  - If the backend returns an error or the network fails, the UI should:
    - Display a user‑friendly message (e.g., “The tutor is temporarily unavailable.”).
    - Preserve the user’s typed question for retry.

### Flask–Frontend Integration

The Flask app in `app.py` acts as the controller between templates and the RAG/chatbot backend.

- **Routes and their templates**:
  - `/` → `index.html`, with context variables:
    - `username` (if logged in).
    - `logged_in` boolean, used to toggle buttons/links.
  - `/login` → `login.html`
  - `/register` → `register.html`
  - `/chat` → `chat.html` (requires `session['user']`).

- **Session‑aware rendering**:
  - Frontend pages read **Flask session variables** to:
    - Show or hide navigation elements.
    - Personalize the chat interface with the user’s name.

### Frontend Considerations for Research

From a research point of view, the frontend layer provides:

- A **controlled interaction environment** for evaluating user experience with RAG‑based tutors.
- The ability to log and analyze:
  - Question types.
  - Perceived latency (via UI loading indicators).
  - User behavior around ambiguous or “uncertain” responses.

Extensions that can be discussed in a paper include:

- A/B testing different **explanatory visualizations** for model confidence.
- Integrating inline explanations of **why** a response was deemed grounded (e.g., highlighting supporting chunks).
- Collecting **implicit feedback** via user corrections or follow‑up questions.




