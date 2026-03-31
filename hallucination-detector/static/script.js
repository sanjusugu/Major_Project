const askBtn = document.getElementById("ask-btn");
const ingestBtn = document.getElementById("ingest-btn");
const questionInput = document.getElementById("question-input");
const loadingState = document.getElementById("loading");
const resultsSection = document.getElementById("results");

// DOM Elements for Results
const resAnswer = document.getElementById("res-answer");
const resSimScore = document.getElementById("res-sim-score");
const resSimLabel = document.getElementById("res-sim-label");
const resJudgeVerdict = document.getElementById("res-judge-verdict");
const resJudgeExp = document.getElementById("res-judge-exp");
const resHybridScore = document.getElementById("res-hybrid-score");
const resHybridBar = document.getElementById("res-hybrid-bar");
const resHybridGating = document.getElementById("res-hybrid-gating");

function showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.style.color = type === "error" ? "#ef4444" : "#22c55e";
    toast.classList.remove("hidden");
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 3000);
}

ingestBtn.addEventListener("click", async () => {
    try {
        ingestBtn.disabled = true;
        ingestBtn.innerHTML = '<span class="icon">⌛</span> Syncing...';
        const res = await fetch("/ingest", { method: "POST" });
        if (res.ok) {
            showToast("Documents synced successfully.");
        } else {
            showToast("Failed to sync documents.", "error");
        }
    } catch (err) {
        showToast("Network error.", "error");
    } finally {
        ingestBtn.disabled = false;
        ingestBtn.innerHTML = '<span class="icon">📄</span> Sync Documents';
    }
});

askBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) return;

    // Reset UI
    resultsSection.classList.add("hidden");
    loadingState.classList.remove("hidden");
    resHybridBar.style.width = '0%';

    askBtn.disabled = true;

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, top_k: 4 })
        });

        const data = await response.json();

        if (response.ok) {
            // Populate Data
            resAnswer.textContent = data.answer;

            // Similarity
            resSimScore.textContent = data.similarity.max_score.toFixed(2);
            resSimLabel.textContent = data.similarity.label;

            // Judge
            if (data.judge) {
                resJudgeVerdict.textContent = data.judge.verdict.replace("_", " ");
                resJudgeExp.textContent = data.judge.explanation;

                // Color Verdict
                resJudgeVerdict.className = 'verdict-value ' +
                    (data.judge.verdict === "SUPPORTED" ? 'success' :
                        data.judge.verdict === "PARTIAL" ? 'warning' : 'danger');
            } else {
                resJudgeVerdict.textContent = "SKIPPED";
                resJudgeExp.textContent = "Similarity too low, bypassing judge.";
            }

            // Hybrid
            const finalScore = data.hybrid.final_score * 100;
            resHybridScore.textContent = data.hybrid.final_score.toFixed(2);
            resHybridGating.textContent = `Gating: ${data.hybrid.gating_label.replace(/_/g, " ")}`;

            // Render UI
            loadingState.classList.add("hidden");
            resultsSection.classList.remove("hidden");

            // Animate progress bar based on final score
            setTimeout(() => {
                resHybridBar.style.width = `${Math.min(finalScore, 100)}%`;

                // Set color of progress bar based on hallucination threshold
                if (data.hybrid.final_score > 0.8) {
                    resHybridBar.style.background = "var(--success)";
                    resSimLabel.className = 'badge bg-success';
                } else if (data.hybrid.final_score > 0.4) {
                    resHybridBar.style.background = "var(--warning)";
                    resSimLabel.className = 'badge bg-warning';
                } else {
                    resHybridBar.style.background = "var(--danger)";
                    resSimLabel.className = 'badge bg-danger';
                }
            }, 100);

        } else {
            loadingState.classList.add("hidden");
            showToast(data.detail || "Server error", "error");
        }

    } catch (err) {
        loadingState.classList.add("hidden");
        showToast("Failed to connect to server.", "error");
    } finally {
        askBtn.disabled = false;
    }
});

questionInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        askBtn.click();
    }
});
