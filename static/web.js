const fileForm = document.getElementById("fileForm");
const urlForm = document.getElementById("urlForm");
const fileInput = document.getElementById("fileInput");
const audioUrl = document.getElementById("audioUrl");
const fileMeta = document.getElementById("fileMeta");
const dropzone = document.getElementById("dropzone");
const submitFileBtn = document.getElementById("submitFileBtn");
const submitUrlBtn = document.getElementById("submitUrlBtn");
const languageSelect = document.getElementById("language");
const detailSelect = document.getElementById("detail");
const taskHint = document.getElementById("taskHint");
const dashboard = document.getElementById("dashboard");
const statusChip = document.getElementById("statusChip");
const progressText = document.getElementById("progressText");
const progressPercent = document.getElementById("progressPercent");
const progressFill = document.getElementById("progressFill");
const taskIdBadge = document.getElementById("taskIdBadge");
const taskSourceBadge = document.getElementById("taskSourceBadge");
const taskTitle = document.getElementById("taskTitle");
const taskLanguage = document.getElementById("taskLanguage");
const taskDetail = document.getElementById("taskDetail");
const taskSource = document.getElementById("taskSource");
const taskStage = document.getElementById("taskStage");
const warningBox = document.getElementById("warningBox");
const errorBox = document.getElementById("errorBox");
const resultSummary = document.getElementById("resultSummary");
const resultActions = document.getElementById("resultActions");
const summaryPreview = document.getElementById("summaryPreview");
const transcriptPreview = document.getElementById("transcriptPreview");
const logsContainer = document.getElementById("logsContainer");
const copySummaryBtn = document.getElementById("copySummaryBtn");
const copySummaryTabBtn = document.getElementById("copySummaryTabBtn");
const copyTranscriptBtn = document.getElementById("copyTranscriptBtn");
const resetBtn = document.getElementById("resetBtn");
const modeTabs = Array.from(document.querySelectorAll(".mode-tab"));
const contentTabs = Array.from(document.querySelectorAll(".tab-btn"));
const flowSteps = Array.from(document.querySelectorAll("[data-flow-step]"));

const contentPanels = {
    summary: document.getElementById("summaryTab"),
    transcript: document.getElementById("transcriptTab"),
    log: document.getElementById("logTab"),
};

const panels = {
    file: document.getElementById("filePanel"),
    url: document.getElementById("urlPanel"),
};

const storageKey = "conspectum:last-task-id";

const state = {
    currentTaskId: null,
    pollTimer: null,
    renderedMessagesCount: 0,
    lastData: null,
    liveTimer: null,
    launchToken: 0,
};

function setHint(text) {
    taskHint.textContent = text;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.innerText = text ?? "";
    return div.innerHTML;
}

function prettyLanguage(code, fallback = "Ожидание") {
    if (code === "en") return "Английский";
    if (code === "ru") return "Русский";
    return code || fallback;
}

function prettyDetail(code) {
    if (code === "brief") return "Кратко";
    if (code === "detailed") return "Подробно";
    return "Сбалансированно";
}

function truncate(text, max = 58) {
    if (!text) return "Ожидание";
    if (text.length <= max) return text;
    const edge = Math.max(12, Math.floor((max - 3) / 2));
    return `${text.slice(0, edge)}...${text.slice(-edge)}`;
}

function formatBytes(bytes) {
    if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) return "—";
    const units = ["B", "KB", "MB", "GB"];
    let value = Number(bytes);
    let idx = 0;
    while (value >= 1024 && idx < units.length - 1) {
        value /= 1024;
        idx += 1;
    }
    return `${value.toFixed(value >= 100 || idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function formatTime(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function formatDuration(seconds, running = false) {
    if (seconds === null || seconds === undefined) return running ? "В работе" : "—";
    const total = Math.max(0, Number(seconds));
    const hrs = Math.floor(total / 3600);
    const mins = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
}

function persistTask(taskId) {
    localStorage.setItem(storageKey, taskId);
}

function clearPersistedTask() {
    localStorage.removeItem(storageKey);
}

function clearPollTimer() {
    if (state.pollTimer) {
        clearTimeout(state.pollTimer);
        state.pollTimer = null;
    }
}

function clearLiveTimer() {
    if (state.liveTimer) {
        clearInterval(state.liveTimer);
        state.liveTimer = null;
    }
}

function setMode(mode) {
    modeTabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.mode === mode);
    });
    Object.entries(panels).forEach(([key, panel]) => {
        panel.classList.toggle("active", key === mode);
    });
}

function setContentTab(mode) {
    contentTabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.tab === mode);
    });
    Object.entries(contentPanels).forEach(([key, panel]) => {
        panel.classList.toggle("active", key === mode);
    });
}

function setStatus(status) {
    statusChip.className = `status-chip ${status}`;
    if (status === "done") {
        statusChip.textContent = "Готово";
        return;
    }
    if (status === "error") {
        statusChip.textContent = "Ошибка";
        return;
    }
    if (status === "running") {
        statusChip.textContent = "В работе";
        return;
    }
    statusChip.textContent = "Готово к запуску";
}

function getFlowGroup(stage, status) {
    if (status === "done") return "pdf";
    if (!stage || stage === "Ожидание") return null;

    if (
        [
            "В очереди",
            "Началась обработка",
            "Подготовка URL",
            "Скачивание источника",
        ].includes(stage)
    ) {
        return "upload";
    }

    if (["Распознавание аудио", "Транскрипт готов"].includes(stage)) {
        return "transcript";
    }

    if (
        [
            "Определение языка",
            "Сборка конспекта",
            "Сборка разделов",
            "Постобработка",
        ].includes(stage)
    ) {
        return "summary";
    }

    if (
        [
            "Сборка PDF",
            "Повторная сборка PDF",
            "Проблема с PDF",
            "Только TEX",
            "Готово",
            "Ошибка",
        ].includes(stage)
    ) {
        return "pdf";
    }

    return "upload";
}

function updateFlowSteps(stage, status) {
    const order = ["upload", "transcript", "summary", "pdf"];
    const activeKey = getFlowGroup(stage, status);
    const activeIndex = activeKey ? order.indexOf(activeKey) : -1;

    flowSteps.forEach((step) => {
        const stepIndex = order.indexOf(step.dataset.flowStep);
        step.classList.remove("pending", "current", "complete", "error");

        if (status === "idle" || activeIndex === -1) {
            step.classList.add("pending");
            return;
        }

        if (status === "done") {
            step.classList.add("complete");
            return;
        }

        if (stepIndex < activeIndex) {
            step.classList.add("complete");
            return;
        }

        if (stepIndex === activeIndex) {
            step.classList.add(status === "error" ? "error" : "current");
            return;
        }

        step.classList.add("pending");
    });
}

function inferDisplayedProgress(progress, status, stage) {
    const safe = Math.max(0, Math.min(100, Number(progress || 0)));
    if (status === "idle") return 0;
    if (status === "done") return 100;
    if (status === "error") return Math.max(safe, 5);

    const stageRanges = {
        "В очереди": [2, 4],
        "Началась обработка": [4, 8],
        "Подготовка URL": [8, 12],
        "Скачивание источника": [12, 18],
        "Распознавание аудио": [18, 38],
        "Транскрипт готов": [38, 42],
        "Определение языка": [42, 48],
        "Сборка конспекта": [48, 60],
        "Сборка разделов": [60, 80],
        "Постобработка": [80, 90],
        "Сборка PDF": [90, 96],
        "Повторная сборка PDF": [94, 98],
        "Проблема с PDF": [92, 96],
        "Только TEX": [97, 99],
    };

    const [minStage, maxStage] = stageRanges[stage] || [2, 99];
    return Math.min(Math.max(safe, minStage), maxStage);
}

function updateProgress(progress, status = "idle", stage = "Ожидание") {
    const displayed = inferDisplayedProgress(progress, status, stage);
    progressFill.style.width = `${displayed}%`;
    progressPercent.textContent = `${displayed}%`;
    progressFill.classList.toggle("running", status === "running");
    updateFlowSteps(stage, status);

    if (status === "done") {
        progressText.textContent = "Обработка завершена";
        return;
    }

    if (status === "error") {
        progressText.textContent = "Обработка остановлена";
        return;
    }

    if (status === "idle") {
        progressText.textContent = "Готово к запуску";
        return;
    }

    if (stage === "Началась обработка") {
        progressText.textContent = "Началась обработка, делаем качественный конспект, ожидайте";
        return;
    }

    if (stage && stage !== "В очереди") {
        progressText.textContent = stage;
        return;
    }

    progressText.textContent = displayed > 0 ? "Запускаю обработку" : "Подготовка задачи";
}

function setBusy(_mode, busy) {
    submitFileBtn.disabled = busy;
    submitUrlBtn.disabled = busy;
    modeTabs.forEach((tab) => {
        tab.disabled = busy;
    });
}

function unlockInputs() {
    submitFileBtn.disabled = false;
    submitUrlBtn.disabled = false;
    modeTabs.forEach((tab) => {
        tab.disabled = false;
    });
}

function showWarning(text) {
    warningBox.style.display = text ? "block" : "none";
    warningBox.textContent = text || "";
}

function showError(text) {
    errorBox.style.display = text ? "block" : "none";
    errorBox.textContent = text || "";
}

function updateFileMeta() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
        fileMeta.innerHTML =
            "<strong>Файл ещё не выбран.</strong><br>После выбора здесь появятся имя, размер и тип источника.";
        return;
    }

    fileMeta.innerHTML = [
        `<strong>${escapeHtml(file.name)}</strong>`,
        `Размер: ${escapeHtml(formatBytes(file.size))}`,
        `Тип: ${escapeHtml(file.type || "неизвестно")}`,
    ].join("<br>");
}

function resetLogs() {
    state.renderedMessagesCount = 0;
    logsContainer.innerHTML =
        '<div class="empty">Сообщения обработки появятся здесь сразу после старта задачи.</div>';
}

function renderMessages(messages) {
    if (!messages || messages.length === 0) {
        resetLogs();
        return;
    }

    const stick =
        logsContainer.scrollHeight - logsContainer.scrollTop - logsContainer.clientHeight < 24;
    let list = logsContainer.querySelector(".log-list");

    if (!list || messages.length < state.renderedMessagesCount) {
        logsContainer.innerHTML = '<ol class="log-list"></ol>';
        list = logsContainer.querySelector(".log-list");
        state.renderedMessagesCount = 0;
    }

    messages.slice(state.renderedMessagesCount).forEach((message) => {
        const item = document.createElement("li");
        item.textContent = message;
        list.appendChild(item);
    });

    state.renderedMessagesCount = messages.length;
    if (stick || state.renderedMessagesCount <= 2) {
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
}

function renderResults(data) {
    const links = [];
    if (data.bundle_url) {
        links.push(`<a class="download" href="${data.bundle_url}">Скачать всё</a>`);
    }
    if (data.transcript_url) {
        links.push(
            `<a class="download" href="${data.transcript_url}" download="transcript.txt">Транскрипт TXT</a>`
        );
    }
    if (data.tex_url) {
        links.push(`<a class="download" href="${data.tex_url}" download="result.tex">Файл TEX</a>`);
    }
    if (data.pdf_url) {
        links.push(`<a class="download primary" href="${data.pdf_url}" download="result.pdf">Файл PDF</a>`);
    }

    resultActions.innerHTML = links.length
        ? links.join("")
        : '<span class="chip">Файлы появятся после завершения обработки</span>';
}

function inferStage(data) {
    if (!data) return "Ожидание";
    if (data.stage) return data.stage;
    if (data.status === "done") return "Готово";
    if (data.status === "error") return "Ошибка";

    const messages = (data.messages || []).map((message) => String(message).toLowerCase());
    const latest = messages.slice(-6).join(" ");

    if (latest.includes("fetching audio")) return "Скачивание источника";
    if (latest.includes("starting transcription")) return "Распознавание аудио";
    if (latest.includes("transcription complete")) return "Транскрипт готов";
    if (latest.includes("detected language")) return "Определение языка";
    if (latest.includes("topic of the lecture") || latest.includes("abstract of the lecture")) {
        return "Сборка конспекта";
    }
    if (latest.includes("starting postprocessing")) return "Постобработка";
    if (latest.includes("retrying pdf generation")) return "Повторная сборка PDF";
    if (latest.includes("pdf generation skipped")) return "Только TEX";
    if (latest.includes("failed to convert latex to pdf")) return "Проблема с PDF";
    if (latest.includes("progress:")) return "Сборка разделов";
    return data.source_mode === "url" ? "Подготовка URL" : "В очереди";
}

function refreshLiveDuration() {
    return;
}

function ensureLiveDurationTimer() {
    clearLiveTimer();
    if (state.lastData && state.lastData.status === "running" && state.lastData.created_at) {
        refreshLiveDuration();
        state.liveTimer = setInterval(refreshLiveDuration, 1000);
    }
}

function renderMetadata(data) {
    const stageLabel = inferStage(data);
    taskIdBadge.textContent = data.task_id ? `ID ${data.task_id.slice(0, 8)}` : "Новая задача";
    taskSourceBadge.textContent = data.source_mode === "url" ? "Источник: URL" : "Источник: файл";
    taskTitle.textContent = data.title || "Ожидание";
    taskLanguage.textContent = prettyLanguage(data.language, "Автоопределение");
    taskDetail.textContent = prettyDetail(data.detail);
    taskSource.textContent = truncate(data.source_name, 54);
    taskSource.title = data.source_name || "";
    taskStage.textContent = stageLabel;

    const lines = [
        `<strong>${escapeHtml(data.title || "Метаданные ещё подготавливаются.")}</strong>`,
        `Язык: ${escapeHtml(prettyLanguage(data.language, "Автоопределение"))}`,
        `Глубина: ${escapeHtml(prettyDetail(data.detail))}`,
    ];

    if (data.abstract_words || data.transcript_words) {
        lines.push(
            `Слов: ${escapeHtml(String(data.abstract_words || 0))} в конспекте, ${escapeHtml(
                String(data.transcript_words || 0)
            )} в транскрипте`
        );
    }

    resultSummary.innerHTML = lines.join("<br>");
    summaryPreview.textContent =
        data.abstract || "Финальный конспект появится здесь после завершения обработки.";
    transcriptPreview.textContent =
        data.transcript_preview || "Транскрипт появится здесь после распознавания аудио.";
    copySummaryBtn.disabled = !data.abstract;
    copySummaryTabBtn.disabled = !data.abstract;
    copyTranscriptBtn.disabled = !data.transcript_preview;
    ensureLiveDurationTimer();
}

function renderIdleDashboard() {
    dashboard.classList.add("visible");
    state.lastData = null;
    state.currentTaskId = null;
    state.renderedMessagesCount = 0;
    setStatus("idle");
    updateProgress(0, "idle", "Ожидание");
    showWarning("");
    showError("");
    taskIdBadge.textContent = "Новая задача";
    taskSourceBadge.textContent = "Ожидает источник";
    taskTitle.textContent = "Будет определён автоматически";
    taskLanguage.textContent = languageSelect.value ? prettyLanguage(languageSelect.value) : "Автоопределение";
    taskDetail.textContent = prettyDetail(detailSelect.value);
    taskSource.textContent = "Файл или URL";
    taskSource.title = "";
    taskStage.textContent = "Ожидание запуска";
    resultSummary.innerHTML =
        "<strong>После обработки появятся:</strong><br>transcript.txt, result.tex, result.pdf и архив со всеми файлами.";
    resultActions.innerHTML = [
        '<span class="chip">transcript.txt</span>',
        '<span class="chip">result.tex</span>',
        '<span class="chip">result.pdf</span>',
    ].join("");
    summaryPreview.textContent = "Здесь появится финальный конспект после завершения обработки.";
    transcriptPreview.textContent = "Здесь появится транскрипт аудио после распознавания.";
    copySummaryBtn.disabled = true;
    copySummaryTabBtn.disabled = true;
    copyTranscriptBtn.disabled = true;
    resetLogs();
    clearLiveTimer();
}

function prepareDashboard(sourceLabel) {
    dashboard.classList.add("visible");
    state.lastData = null;
    state.renderedMessagesCount = 0;
    setStatus("running");
    updateProgress(0, "running", "Началась обработка");
    showWarning("");
    showError("");
    taskIdBadge.textContent = "Задача создаётся";
    taskSourceBadge.textContent = sourceLabel ? truncate(sourceLabel, 38) : "Подготовка источника";
    taskTitle.textContent = "Ожидание";
    taskLanguage.textContent = languageSelect.value ? prettyLanguage(languageSelect.value) : "Автоопределение";
    taskDetail.textContent = prettyDetail(detailSelect.value);
    taskSource.textContent = sourceLabel || "Подготовка";
    taskSource.title = sourceLabel || "";
    taskStage.textContent = "Началась обработка";
    resultSummary.textContent = "После старта здесь появятся метаданные и ссылки на скачивание.";
    resultActions.innerHTML = '<span class="chip">Файлы появятся после обработки</span>';
    summaryPreview.textContent = "Финальный конспект появится здесь после завершения обработки.";
    transcriptPreview.textContent = "Транскрипт появится здесь после распознавания аудио.";
    copySummaryBtn.disabled = true;
    copySummaryTabBtn.disabled = true;
    copyTranscriptBtn.disabled = true;
    resetLogs();
    clearLiveTimer();
}

async function readError(response) {
    const type = response.headers.get("content-type") || "";
    if (type.includes("application/json")) {
        const payload = await response.json().catch(() => null);
        if (payload && typeof payload.detail === "string") return payload.detail;
    }
    return await response.text().catch(() => "Ошибка запроса.");
}

async function copyText(value, message) {
    if (!value) {
        setHint("Пока нечего копировать.");
        return;
    }

    try {
        await navigator.clipboard.writeText(value);
        setHint(message);
    } catch (_error) {
        setHint("Буфер обмена недоступен в этом контексте браузера.");
    }
}

function handleTaskError(error, taskId) {
    if (taskId !== state.currentTaskId) return;
    clearPollTimer();
    const isMissingTask = (error.message || "").toLowerCase().includes("task not found");
    setStatus("error");
    updateProgress(
        state.lastData?.progress ?? 0,
        "error",
        state.lastData ? inferStage(state.lastData) : "Ошибка"
    );
    showError(
        isMissingTask
            ? "Задача больше недоступна. Обычно это значит, что сервер был перезапущен или задача устарела."
            : error.message || "Не удалось загрузить статус задачи."
    );

    if (isMissingTask) {
        clearPersistedTask();
        setHint("Текущая задача больше недоступна.");
    } else {
        setHint("Не удалось обновить статус. Попробуй запустить обработку ещё раз.");
    }
}

async function pollTask(taskId) {
    const response = await fetch(`/status/${encodeURIComponent(taskId)}`);
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    if (taskId !== state.currentTaskId) return;

    const currentStage = inferStage(data);
    state.lastData = data;
    renderMessages(data.messages || []);
    renderMetadata(data);
    renderResults(data);
    showWarning(data.warning || "");
    showError(data.status === "error" ? data.error || "Неизвестная ошибка" : "");

    if (data.status === "done") {
        clearPersistedTask();
        setStatus("done");
        updateProgress(data.progress ?? 100, "done", currentStage);
        setHint("Готово. Файлы и предпросмотр уже доступны.");
        return;
    }

    if (data.status === "error") {
        clearPersistedTask();
        setStatus("error");
        updateProgress(data.progress ?? 0, "error", currentStage);
        setHint("Задача завершилась с ошибкой. Проверь log и попробуй ещё раз.");
        return;
    }

    setStatus("running");
    updateProgress(data.progress ?? 0, "running", currentStage);
    state.pollTimer = setTimeout(() => {
        pollTask(taskId).catch((error) => handleTaskError(error, taskId));
    }, 1500);
}

function startPolling(taskId) {
    clearPollTimer();
    clearLiveTimer();
    state.currentTaskId = taskId;
    persistTask(taskId);
    state.renderedMessagesCount = 0;
    pollTask(taskId).catch((error) => handleTaskError(error, taskId));
}

async function startTask(endpoint, formData, sourceLabel, mode) {
    const launchToken = ++state.launchToken;
    clearPollTimer();
    clearLiveTimer();
    state.currentTaskId = null;
    clearPersistedTask();
    prepareDashboard(sourceLabel);
    setBusy(mode, true);
    setHint("Задача создаётся. Статус появится ниже через несколько секунд.");

    try {
        const response = await fetch(endpoint, { method: "POST", body: formData });
        if (!response.ok) throw new Error(await readError(response));
        if (launchToken !== state.launchToken) return;
        const payload = await response.json();
        if (launchToken !== state.launchToken) return;
        logsContainer.innerHTML =
            '<div class="empty">Задача создана. Ожидаем начало обработки и первые сообщения pipeline.</div>';
        startPolling(payload.task_id);
        setHint("Обработка запущена. Интерфейс будет обновляться автоматически.");
    } catch (error) {
        if (launchToken !== state.launchToken) return;
        setStatus("error");
        updateProgress(0, "error", "Ошибка запуска");
        showError(error.message || "Не удалось запустить задачу.");
        setHint("Запуск не удался. Проверь сообщение и попробуй ещё раз.");
    } finally {
        setBusy(mode, false);
    }
}

function resetWorkspace() {
    state.launchToken += 1;
    clearPollTimer();
    clearLiveTimer();
    state.currentTaskId = null;
    state.lastData = null;
    state.renderedMessagesCount = 0;
    clearPersistedTask();
    unlockInputs();
    fileInput.value = "";
    audioUrl.value = "";
    setMode("file");
    setContentTab("summary");
    updateFileMeta();
    renderIdleDashboard();
    setHint("Рабочая область очищена. Выбери новый файл или ссылку и запусти обработку.");
}

modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

contentTabs.forEach((tab) => {
    tab.addEventListener("click", () => setContentTab(tab.dataset.tab));
});

languageSelect.addEventListener("change", () => {
    if (!state.currentTaskId && !state.lastData) renderIdleDashboard();
});

detailSelect.addEventListener("change", () => {
    if (!state.currentTaskId && !state.lastData) renderIdleDashboard();
});

fileInput.addEventListener("change", updateFileMeta);

["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
    });
});

["dragleave", "dragend", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
    });
});

dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});

dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button, label")) return;
    fileInput.click();
});

dropzone.addEventListener("drop", (event) => {
    const dropped = event.dataTransfer.files;
    if (!dropped || dropped.length === 0) return;
    const transfer = new DataTransfer();
    transfer.items.add(dropped[0]);
    fileInput.files = transfer.files;
    updateFileMeta();
});

fileForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
        setHint("Сначала выбери аудиофайл.");
        updateFileMeta();
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    if (languageSelect.value) formData.append("language", languageSelect.value);
    if (detailSelect.value) formData.append("detail", detailSelect.value);
    await startTask("/upload", formData, file.name, "file");
});

urlForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = audioUrl.value.trim();
    if (!url) {
        setHint("Сначала вставь прямой URL на аудиофайл.");
        audioUrl.focus();
        return;
    }

    const formData = new FormData();
    formData.append("audio_url", url);
    if (languageSelect.value) formData.append("language", languageSelect.value);
    if (detailSelect.value) formData.append("detail", detailSelect.value);
    await startTask("/upload-url", formData, url, "url");
});

copySummaryBtn.addEventListener("click", async () => {
    await copyText(state.lastData?.abstract || "", "Конспект скопирован.");
});

copySummaryTabBtn.addEventListener("click", async () => {
    await copyText(state.lastData?.abstract || "", "Конспект скопирован.");
});

copyTranscriptBtn.addEventListener("click", async () => {
    await copyText(state.lastData?.transcript_preview || "", "Транскрипт скопирован.");
});

resetBtn.addEventListener("click", resetWorkspace);

updateFileMeta();
unlockInputs();
setMode("file");
setContentTab("summary");
renderIdleDashboard();

const initialTaskId = localStorage.getItem(storageKey);
if (initialTaskId && initialTaskId.trim()) {
    prepareDashboard("Возобновление обработки...");
    setHint("Восстанавливаю текущую обработку после обновления страницы...");
    startPolling(initialTaskId.trim());
}
