/**
 * PaySense AI — Frontend Application
 * Handles file upload, chat, and document comparison.
 */

const API_BASE = "";

// State
let sessionId = null;
const documents = {};

// DOM Elements
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const uploadZone = document.getElementById("upload-zone");
const uploadProgress = document.getElementById("upload-progress");
const progressFill = document.getElementById("progress-fill");
const progressText = document.getElementById("progress-text");
const documentList = document.getElementById("document-list");
const docCount = document.getElementById("doc-count");
const compareSection = document.getElementById("compare-section");
const compareDocA = document.getElementById("compare-doc-a");
const compareDocB = document.getElementById("compare-doc-b");
const compareBtn = document.getElementById("compare-btn");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

// ============ Session ============

async function initSession() {
    try {
        const resp = await fetch(`${API_BASE}/api/session`, {method: "POST"});
        const data = await resp.json();
        sessionId = data.session_id;
        console.log("Session created:", sessionId);
    } catch (err) {
        console.error("Failed to create session:", err);
        addMessage("assistant", "⚠️ Could not connect to the server. Please make sure the backend is running.");
    }
}

// ============ File Upload ============

browseBtn.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("click", (e) => {
    if (e.target !== browseBtn) fileInput.click();
});

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
    }
});

async function uploadFile(file) {
    if (!sessionId) {
        addMessage("assistant", "⚠️ Session not ready. Please wait...");
        return;
    }

    // Show progress
    uploadProgress.classList.remove("hidden");
    progressFill.style.width = "20%";
    progressText.textContent = "Uploading document...";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);

    try {
        progressFill.style.width = "50%";
        progressText.textContent = "Extracting text & fields...";

        const resp = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            let detail = `Upload failed (HTTP ${resp.status})`;
            try {
                const err = await resp.json();
                detail = err.detail || detail;
            } catch {
            }
            throw new Error(detail);
        }

        progressFill.style.width = "100%";
        progressText.textContent = "Done!";

        const data = await resp.json();
        documents[data.doc_id] = data.fields;
        renderDocuments();
        addMessage("assistant", `✅ ${data.message}\n\nI found: **${data.fields.document_type}** for **${data.fields.employee_name || "Unknown"}** (${data.fields.period || "Unknown period"})`);

        setTimeout(() => {
            uploadProgress.classList.add("hidden");
            progressFill.style.width = "0%";
        }, 1500);
    } catch (err) {
        progressText.textContent = `❌ ${err.message}`;
        progressFill.style.width = "0%";
        addMessage("assistant", `❌ Upload failed: ${err.message}`);
        setTimeout(() => uploadProgress.classList.add("hidden"), 3000);
    }

    fileInput.value = "";
}

// ============ Document List ============

function renderDocuments() {
    const ids = Object.keys(documents);
    docCount.textContent = `${ids.length} uploaded`;
    documentList.innerHTML = "";

    ids.forEach((docId) => {
        const fields = documents[docId];
        const card = document.createElement("div");
        card.className = "doc-card";
        card.innerHTML = `
            <div class="doc-card-header">
                <div class="doc-card-title">
                    <strong>${fields.employee_name || docId}</strong>
                    ${fields.period ? `<span class="doc-card-period">${fields.period}</span>` : ""}
                </div>
                <span class="doc-type-badge">${fields.document_type || "doc"}</span>
            </div>
            <div class="doc-card-stats">
                ${fields.basic_pay != null ? `<div class="doc-stat"><span class="doc-stat-label">Basic</span><span class="doc-stat-value">₹${Number(fields.basic_pay).toLocaleString("en-IN")}</span></div>` : ""}
                ${fields.gross_salary != null ? `<div class="doc-stat"><span class="doc-stat-label">Gross</span><span class="doc-stat-value">₹${Number(fields.gross_salary).toLocaleString("en-IN")}</span></div>` : ""}
                ${fields.net_pay != null ? `<div class="doc-stat doc-stat-highlight"><span class="doc-stat-label">Net Pay</span><span class="doc-stat-value">₹${Number(fields.net_pay).toLocaleString("en-IN")}</span></div>` : ""}
            </div>
        `;
        documentList.appendChild(card);
    });

    // Show compare section if 2+ documents
    if (ids.length >= 2) {
        compareSection.classList.remove("hidden");
        compareDocA.innerHTML = ids.map((id) => `<option value="${id}">${documents[id].employee_name || id} (${documents[id].period || ""})</option>`).join("");
        compareDocB.innerHTML = ids.map((id) => `<option value="${id}">${documents[id].employee_name || id} (${documents[id].period || ""})</option>`).join("");
        if (ids.length >= 2) compareDocB.selectedIndex = 1;
    } else {
        compareSection.classList.add("hidden");
    }
}

// ============ Chat ============

chatInput.addEventListener("input", () => {
    sendBtn.disabled = !chatInput.value.trim();
    // Auto-resize textarea
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim()) sendMessage();
    }
});

sendBtn.addEventListener("click", sendMessage);

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !sessionId) return;

    addMessage("user", message);
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    // Show typing indicator
    const typingId = addTypingIndicator();

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({session_id: sessionId, message}),
        });

        removeTypingIndicator(typingId);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Chat failed");
        }

        const data = await resp.json();
        addMessage("assistant", data.response);
    } catch (err) {
        removeTypingIndicator(typingId);
        addMessage("assistant", `⚠️ Error: ${err.message}`);
    }
}

function addMessage(role, content) {
    const div = document.createElement("div");
    div.className = `message ${role}-message`;
    const avatar = role === "user" ? "👤" : "🤖";
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content"><p>${formatMessage(content)}</p></div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(text) {
    // Basic markdown-like formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}

function addTypingIndicator() {
    const id = "typing-" + Date.now();
    const div = document.createElement("div");
    div.id = id;
    div.className = "message assistant-message";
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content"><p class="loading-spinner">Thinking</p></div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ============ Compare ============

compareBtn.addEventListener("click", async () => {
    const docA = compareDocA.value;
    const docB = compareDocB.value;

    if (docA === docB) {
        addMessage("assistant", "⚠️ Please select two different documents to compare.");
        return;
    }

    addMessage("user", `Compare ${documents[docA]?.period || docA} vs ${documents[docB]?.period || docB}`);
    const typingId = addTypingIndicator();

    try {
        const resp = await fetch(`${API_BASE}/api/compare`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                session_id: sessionId,
                doc_id_a: docA,
                doc_id_b: docB,
            }),
        });

        removeTypingIndicator(typingId);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Comparison failed");
        }

        const data = await resp.json();
        addMessage("assistant", data.explanation);
    } catch (err) {
        removeTypingIndicator(typingId);
        addMessage("assistant", `⚠️ Comparison error: ${err.message}`);
    }
});

// ============ Initialize ============

initSession();