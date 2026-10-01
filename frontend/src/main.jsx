import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Box,
  CheckCircle2,
  ChevronDown,
  Circle,
  Clock3,
  Database,
  FileText,
  Layers3,
  LoaderCircle,
  Play,
  RotateCcw,
  Search,
  Sparkles,
  TerminalSquare,
  XCircle,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const STAGES = [
  { id: "loading", label: "Loading", subtitle: "PDF → JSONL", icon: FileText },
  { id: "chunking", label: "Chunking", subtitle: "JSONL → Chunks", icon: Layers3 },
  { id: "embedding", label: "Embedding", subtitle: "Chunks → Vectors", icon: Sparkles },
  { id: "retrieval", label: "Retrieval", subtitle: "Query → Top-K", icon: Search },
];

function ms(value) {
  if (value === undefined || value === null) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${Number(value).toFixed(2)} ms`;
}

function App() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [status, setStatus] = useState(null);
  const [stageState, setStageState] = useState({});
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);

  const [strategy, setStrategy] = useState("structure_block_aware");
  const [targetTokens, setTargetTokens] = useState(600);
  const [overlap, setOverlap] = useState(100);
  const [fullRebuild, setFullRebuild] = useState(false);

  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [retrieving, setRetrieving] = useState(false);
  const [results, setResults] = useState([]);
  const [retrievalTime, setRetrievalTime] = useState(null);
  const [error, setError] = useState("");

  const completedCount = useMemo(
    () => STAGES.filter((s) => stageState[s.id]?.status === "success").length,
    [stageState]
  );
  const selectedFileInfo = useMemo(
    () => files.find((file) => file.name === selectedFile),
    [files, selectedFile]
  );

  async function request(url, options) {
    const response = await fetch(`${API}${url}`, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Có lỗi xảy ra.");
    return data;
  }

  async function refresh() {
    try {
      const [fileData, statusData] = await Promise.all([
        request("/api/files"),
        request("/api/pipeline/status"),
      ]);
      setFiles(fileData.files || []);
      setStatus(statusData);
      if (!selectedFile && fileData.files?.length) {
        setSelectedFile(fileData.files[0].name);
      }
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  function pushLog(stage, message, kind = "info", durationMs = null) {
    setLogs((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        time: new Date().toLocaleTimeString("vi-VN"),
        stage,
        message,
        kind,
        durationMs,
      },
    ]);
  }

  function markRunning(stage) {
    setStageState((prev) => ({
      ...prev,
      [stage]: { status: "running", duration_ms: null },
    }));
  }

  function markSuccess(stage, data) {
    setStageState((prev) => ({
      ...prev,
      [stage]: {
        status: "success",
        duration_ms: data.duration_ms,
        details: data.details,
      },
    }));
    pushLog(stage, data.message, "success", data.duration_ms);
  }

  function markError(stage, message) {
    setStageState((prev) => ({
      ...prev,
      [stage]: { status: "error" },
    }));
    pushLog(stage, message, "error");
  }

  async function runPipeline() {
    if (!selectedFile && !fullRebuild) {
      setError("Không có PDF trong data/raw.");
      return;
    }

    if (
      fullRebuild &&
      !window.confirm(
        "Full rebuild sẽ xóa và xây dựng lại toàn bộ Vector Database từ tất cả tài liệu trong corpus, không chỉ tài liệu đang chọn."
      )
    ) {
      return;
    }

    setRunning(true);
    setError("");
    setResults([]);
    setRetrievalTime(null);
    setStageState((prev) => ({
      ...prev,
      loading: undefined,
      chunking: undefined,
      embedding: undefined,
      retrieval: undefined,
    }));
    setLogs([]);

    try {
      if (fullRebuild) {
        markRunning("embedding");
        pushLog("embedding", "Bắt đầu rebuild toàn bộ canonical corpus...");
        const data = await request("/api/corpus/rebuild", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirm: true }),
        });
        markSuccess("embedding", data);
        setFullRebuild(false);
      } else {
        markRunning("loading");
        markRunning("chunking");
        markRunning("embedding");
        pushLog("corpus", `Kiểm tra tài liệu ${selectedFile}...`);
        const data = await request("/api/corpus/index-document", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_name: selectedFile,
            strategy,
            target_tokens: Number(targetTokens),
            token_overlap: Number(overlap),
          }),
        });
        const timings = data.timings || {};
        setStageState((prev) => ({
          ...prev,
          loading: {
            status: "success",
            duration_ms: Number(timings.loading || 0) + Number(timings.captioning || 0),
          },
          chunking: { status: "success", duration_ms: timings.chunking || 0 },
          embedding: { status: "success", duration_ms: timings.embedding_indexing || 0 },
        }));
        pushLog(
          "corpus",
          `${data.message}. Chunks ${data.chunks_before} → ${data.chunks_after}; vectors ${data.vectors_before} → ${data.vectors_after}.`,
          "success",
          data.duration_ms
        );
      }
      await refresh();
    } catch (e) {
      markError(fullRebuild ? "embedding" : "loading", e.message);
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  async function runRetrieval() {
    if (!query.trim()) {
      setError("Vui lòng nhập query.");
      return;
    }

    setRetrieving(true);
    setError("");
    setResults([]);
    markRunning("retrieval");
    pushLog("retrieval", `Bắt đầu Retrieval Top-${topK}...`);

    try {
      const data = await request("/api/retrieval/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          top_k: Number(topK),
        }),
      });

      setResults(data.results || []);
      setRetrievalTime(data.total_duration_ms);
      setStageState((prev) => ({
        ...prev,
        retrieval: {
          status: "success",
          duration_ms: data.total_duration_ms,
        },
      }));

      for (const log of data.logs || []) {
        pushLog(
          log.stage,
          log.message,
          log.status === "success" ? "success" : "info",
          log.duration_ms
        );
      }
    } catch (e) {
      markError("retrieval", e.message);
      setError(e.message);
    } finally {
      setRetrieving(false);
    }
  }

  const [chatProvider, setChatProvider] = useState("gemini");
  const [chatLoading, setChatLoading] = useState(false);
  const [ragAnswer, setRagAnswer] = useState(null);

  async function runRagChat() {
    if (!query.trim()) return;
    setChatLoading(true);
    setRagAnswer(null);
    setError("");

    try {
      const res = await fetch(`${API}/api/rag/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: topK,
          top_n: 5,
          threshold: 0.35,
          provider: chatProvider
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "RAG Chat thất bại.");
      }

      const data = await res.json();
      setRagAnswer(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setChatLoading(false);
    }
  }

  function resetUI() {
    setStageState({});
    setLogs([]);
    setResults([]);
    setRagAnswer(null);
    setRetrievalTime(null);
    setError("");
    setFullRebuild(false);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Box size={21} /></div>
          <div>
            <strong>RAG Lab</strong>
            <span>Pipeline Console</span>
          </div>
        </div>

        <nav className="nav">
          <a className="active"><Activity size={17} /> Pipeline</a>
          <a><Database size={17} /> Vector Store</a>
          <a><TerminalSquare size={17} /> Logs</a>
        </nav>

        <div className="sidebar-bottom">
          <div className="device-pill">
            <span className="dot" />
            {status?.device?.toUpperCase() || "—"}
          </div>
          <small>{status?.embedding_model || "BAAI/bge-m3"}</small>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="overline">VIETNAMESE PDF RAG</p>
            <h1>Pipeline Debug Console</h1>
            <p className="subtle">
              Quan sát trực tiếp Loading → Chunking → Embedding → Retrieval.
            </p>
          </div>
          <button className="ghost-btn" onClick={resetUI}>
            <RotateCcw size={16} /> Reset UI
          </button>
        </header>

        <section className="stage-grid">
          {STAGES.map((stage, index) => {
            const Icon = stage.icon;
            const state = stageState[stage.id];
            const statusName = state?.status || "idle";

            return (
              <React.Fragment key={stage.id}>
                <article className={`stage-card ${statusName}`}>
                  <div className="stage-top">
                    <div className="stage-icon"><Icon size={20} /></div>
                    <div className={`status-dot ${statusName}`}>
                      {statusName === "running" ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : statusName === "success" ? (
                        <CheckCircle2 size={17} />
                      ) : statusName === "error" ? (
                        <XCircle size={17} />
                      ) : (
                        <Circle size={14} />
                      )}
                    </div>
                  </div>
                  <strong>{stage.label}</strong>
                  <span>{stage.subtitle}</span>
                  <div className="stage-time">
                    <Clock3 size={13} /> {ms(state?.duration_ms)}
                  </div>
                </article>
                {index < STAGES.length - 1 && <div className="connector">→</div>}
              </React.Fragment>
            );
          })}
        </section>

        <div className="content-grid">
          <section className="panel setup-panel">
            <div className="panel-heading">
              <div>
                <p className="overline">01 — OFFLINE PIPELINE</p>
                <h2>Incremental Document Corpus</h2>
              </div>
              <div className="progress-ring">{completedCount}/4</div>
            </div>

            <label>Tài liệu PDF</label>
            <div className="select-wrap">
              <select
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.value)}
                disabled={running}
              >
                {files.length === 0 && <option>Không có PDF trong data/raw</option>}
                {files.map((file) => (
                  <option key={file.name} value={file.name}>
                    [{String(file.corpus_status || "new").toUpperCase()}] {file.name} ({file.size_mb} MB)
                  </option>
                ))}
              </select>
              <ChevronDown size={16} />
            </div>

            {selectedFileInfo && (
              <div className={`document-status ${selectedFileInfo.corpus_status || "new"}`}>
                Document status: {String(selectedFileInfo.corpus_status || "new").toUpperCase()}
              </div>
            )}

            <div className="form-row">
              <div>
                <label>Chunk strategy</label>
                <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                  <option value="structure_block_aware">Structure &amp; Block-Aware</option>
                  <option value="token">Token</option>
                  <option value="size">Size</option>
                  <option value="semantic">Semantic</option>
                </select>
              </div>
              <div>
                <label>Target tokens</label>
                <input
                  type="number"
                  min="100"
                  max="4000"
                  value={targetTokens}
                  onChange={(e) => setTargetTokens(e.target.value)}
                />
              </div>
              <div>
                <label>Overlap</label>
                <input
                  type="number"
                  min="0"
                  max="1000"
                  value={overlap}
                  onChange={(e) => setOverlap(e.target.value)}
                />
              </div>
            </div>

            <div className="rebuild-controls">
              <label className="rebuild-option">
                <input
                  type="checkbox"
                  checked={fullRebuild}
                  onChange={(e) => setFullRebuild(e.target.checked)}
                  disabled={running}
                />
                <span>Full rebuild vector database</span>
              </label>
              {fullRebuild && (
                <div className="rebuild-warning" role="alert">
                  Full rebuild sẽ xóa và xây dựng lại toàn bộ Vector Database từ tất cả tài liệu trong corpus, không chỉ tài liệu đang chọn.
                </div>
              )}
            </div>

            <button className="primary-btn" onClick={runPipeline} disabled={running || !selectedFile}>
              {running ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
              {running
                ? "Đang xử lý corpus..."
                : fullRebuild
                  ? "Rebuild Entire Corpus"
                  : "Index Selected Document"}
            </button>

            <div className="stats-row">
              <div><span>Documents</span><strong>{status?.pdf_documents ?? 0}</strong></div>
              <div><span>Chunks</span><strong>{status?.chunks ?? 0}</strong></div>
              <div><span>Vectors</span><strong>{status?.vectors ?? 0}</strong></div>
            </div>
          </section>

          <section className="panel log-panel">
            <div className="panel-heading">
              <div>
                <p className="overline">LIVE TRACE</p>
                <h2>Pipeline Logs</h2>
              </div>
              <TerminalSquare size={19} />
            </div>

            <div className="terminal">
              {logs.length === 0 ? (
                <div className="terminal-empty">
                  Chưa có log. Nhấn <strong>Index Selected Document</strong>.
                </div>
              ) : (
                logs.map((log) => (
                  <div className={`log-line ${log.kind}`} key={log.id}>
                    <span className="log-time">{log.time}</span>
                    <span className="log-stage">[{String(log.stage).toUpperCase()}]</span>
                    <span className="log-message">{log.message}</span>
                    {log.durationMs !== null && (
                      <span className="log-duration">{ms(log.durationMs)}</span>
                    )}
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        <section className="panel retrieval-panel">
          <div className="panel-heading">
            <div>
              <p className="overline">02 — ONLINE RETRIEVAL</p>
              <h2>Query Vector Database</h2>
            </div>
            {retrievalTime !== null && (
              <span className="latency-badge"><Clock3 size={14} /> {ms(retrievalTime)}</span>
            )}
          </div>

          <div className="query-row">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ví dụ: RAG hiện đại gồm những giai đoạn nào?"
              rows={3}
            />
            <div className="topk-box">
              <label>Top K</label>
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value))))}
              />
            </div>
            <div className="topk-box">
              <label>Provider</label>
              <select value={chatProvider} onChange={(e) => setChatProvider(e.target.value)}>
                <option value="gemini">Gemini API</option>
                <option value="openai">OpenAI API</option>
                <option value="ollama">Ollama Local</option>
                <option value="mock">Mock Test</option>
              </select>
            </div>
            <button className="search-btn" onClick={runRetrieval} disabled={retrieving || chatLoading}>
              {retrieving ? <LoaderCircle className="spin" size={18} /> : <Search size={18} />}
              {retrieving ? "Searching" : "Retrieve"}
            </button>
            <button className="primary-btn" style={{ height: "74px" }} onClick={runRagChat} disabled={retrieving || chatLoading}>
              {chatLoading ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
              {chatLoading ? "Generating..." : "RAG Answer"}
            </button>
          </div>

          {error && <div className="error-box">{error}</div>}

          {ragAnswer && (
            <div className="rag-answer-card">
              <div className="rag-answer-header">
                <Sparkles size={18} color="#2563eb" />
                <strong>RAG LLM Generated Answer</strong>
                <span className="rag-time">{ms(ragAnswer.total_duration_ms)}</span>
              </div>
              <div className="rag-answer-body">
                <p style={{ whiteSpace: "pre-wrap", fontSize: "14px", lineHeight: "1.7", color: "#1e293b" }}>
                  {ragAnswer.answer}
                </p>
              </div>

              {ragAnswer.citations && ragAnswer.citations.length > 0 && (
                <div className="citations-box">
                  <strong>Trích dẫn Nguồn (Citations):</strong>
                  <div className="citation-pills">
                    {ragAnswer.citations.map((c, i) => (
                      <span key={i} className="citation-pill">
                        📌 [{c.source_id}] {c.file} — Trang {c.page} ({c.section})
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {ragAnswer.image_urls && ragAnswer.image_urls.length > 0 && (
                <div className="citations-box">
                  <strong>Hình ảnh / Sơ đồ minh họa:</strong>
                  <div className="result-images">
                    {ragAnswer.image_urls.map((imgUrl, i) => (
                      <a key={i} href={`${API}${imgUrl}`} target="_blank" rel="noopener noreferrer" title="Click để mở ảnh kích thước lớn">
                        <img src={`${API}${imgUrl}`} alt={`Attached Visual ${i + 1}`} className="result-img" />
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="results-header">
            <span>TOP-K RESULTS</span>
            <span>{results.length} chunks</span>
          </div>

          <div className="result-list">
            {results.length === 0 ? (
              <div className="empty-results">
                Chạy pipeline trước, sau đó nhập query để xem Top-K.
              </div>
            ) : (
              results.map((item) => (
                <article className="result-card" key={item.chunk_id || `${item.rank}-${item.page}`}>
                  <div className="rank-col">
                    <div className="rank-number">{item.rank}</div>
                    <div className="score">{Number(item.score).toFixed(4)}</div>
                  </div>
                  <div className="result-body">
                    <div className="meta">
                      <span><FileText size={13} /> {item.source || "unknown"}</span>
                      {item.page !== null && item.page !== undefined && <span>Page {item.page}</span>}
                      <span>distance {Number(item.distance).toFixed(4)}</span>
                    </div>
                    <p style={{ whiteSpace: "pre-wrap" }}>{item.content}</p>

                    {item.image_urls && item.image_urls.length > 0 && (
                      <div className="result-images">
                        {item.image_urls.map((imgUrl, i) => (
                          <a key={i} href={`${API}${imgUrl}`} target="_blank" rel="noopener noreferrer" title="Click để mở ảnh kích thước lớn">
                            <img src={`${API}${imgUrl}`} alt={`Visual ${i + 1}`} className="result-img" />
                          </a>
                        ))}
                      </div>
                    )}

                    <details>
                      <summary>Metadata</summary>
                      <pre>{JSON.stringify(item.metadata, null, 2)}</pre>
                    </details>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
