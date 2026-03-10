import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files || []));
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("text", text);

      files.forEach((file) => {
        formData.append("files", file);
      });

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        error: "Failed to connect to backend API."
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Deception Detection Demo</h1>
        <p style={styles.subtitle}>
          Upload text and files to estimate deception rate.
        </p>

        <div style={styles.card}>
          <label style={styles.label}>Paste text</label>
          <textarea
            style={styles.textarea}
            placeholder="Enter review, statement, message, or claim..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <label style={styles.label}>Upload files</label>
          <input
            type="file"
            multiple
            onChange={handleFileChange}
            style={styles.input}
          />

          {files.length > 0 && (
            <div style={styles.fileList}>
              {files.map((file, index) => (
                <div key={index} style={styles.fileItem}>
                  {file.name} ({Math.round(file.size / 1024)} KB)
                </div>
              ))}
            </div>
          )}

          <button
            style={styles.button}
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {result && (
          <div style={styles.resultCard}>
            <h2 style={styles.resultTitle}>Result</h2>

            {result.error ? (
              <p style={styles.error}>{result.error}</p>
            ) : (
              <>
                <div style={styles.metricBox}>
                  <div style={styles.metricLabel}>Deception Rate</div>
                  <div style={styles.metricValue}>
                    {result.deception_rate}%
                  </div>
                </div>

                <div style={styles.grid}>
                  <div style={styles.smallCard}>
                    <div style={styles.smallLabel}>Classification</div>
                    <div style={styles.smallValue}>{result.label}</div>
                  </div>

                  <div style={styles.smallCard}>
                    <div style={styles.smallLabel}>Trustworthiness</div>
                    <div style={styles.smallValue}>
                      {result.trustworthiness_score}%
                    </div>
                  </div>

                  <div style={styles.smallCard}>
                    <div style={styles.smallLabel}>Confidence</div>
                    <div style={styles.smallValue}>
                      {result.confidence}%
                    </div>
                  </div>

                  <div style={styles.smallCard}>
                    <div style={styles.smallLabel}>Mode</div>
                    <div style={styles.smallValue}>
                      {result.model_mode}
                    </div>
                  </div>
                </div>

                {result.uploaded_files && result.uploaded_files.length > 0 && (
                  <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Uploaded Files</h3>
                    {result.uploaded_files.map((file, idx) => (
                      <div key={idx} style={styles.fileItem}>
                        {file.filename} - {file.size_bytes} bytes
                      </div>
                    ))}
                  </div>
                )}

                {result.flags && result.flags.length > 0 && (
                  <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Flags</h3>
                    <ul style={styles.list}>
                      {result.flags.map((flag, idx) => (
                        <li key={idx}>{flag}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#f4f7fb",
    padding: "40px 20px",
    fontFamily: "Arial, sans-serif",
  },
  container: {
    maxWidth: "900px",
    margin: "0 auto",
  },
  title: {
    fontSize: "36px",
    fontWeight: "700",
    marginBottom: "10px",
    color: "#111827",
  },
  subtitle: {
    fontSize: "16px",
    color: "#4b5563",
    marginBottom: "30px",
  },
  card: {
    background: "#ffffff",
    borderRadius: "16px",
    padding: "24px",
    boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
    marginBottom: "24px",
  },
  label: {
    display: "block",
    marginBottom: "8px",
    marginTop: "12px",
    fontWeight: "600",
    color: "#1f2937",
  },
  textarea: {
    width: "100%",
    minHeight: "180px",
    padding: "14px",
    borderRadius: "10px",
    border: "1px solid #d1d5db",
    fontSize: "15px",
    resize: "vertical",
    marginBottom: "10px",
  },
  input: {
    marginTop: "8px",
    marginBottom: "12px",
  },
  fileList: {
    marginTop: "10px",
    marginBottom: "16px",
  },
  fileItem: {
    background: "#f9fafb",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    padding: "10px",
    marginBottom: "8px",
    color: "#374151",
  },
  button: {
    background: "#2563eb",
    color: "#ffffff",
    border: "none",
    borderRadius: "10px",
    padding: "14px 20px",
    fontSize: "16px",
    fontWeight: "600",
    cursor: "pointer",
    marginTop: "12px",
  },
  resultCard: {
    background: "#ffffff",
    borderRadius: "16px",
    padding: "24px",
    boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
  },
  resultTitle: {
    fontSize: "24px",
    fontWeight: "700",
    marginBottom: "20px",
    color: "#111827",
  },
  metricBox: {
    background: "#eef4ff",
    border: "1px solid #bfdbfe",
    borderRadius: "14px",
    padding: "20px",
    marginBottom: "20px",
    textAlign: "center",
  },
  metricLabel: {
    fontSize: "14px",
    color: "#4b5563",
    marginBottom: "8px",
  },
  metricValue: {
    fontSize: "48px",
    fontWeight: "800",
    color: "#1d4ed8",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: "12px",
    marginBottom: "20px",
  },
  smallCard: {
    background: "#f9fafb",
    border: "1px solid #e5e7eb",
    borderRadius: "12px",
    padding: "16px",
  },
  smallLabel: {
    fontSize: "13px",
    color: "#6b7280",
    marginBottom: "6px",
  },
  smallValue: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#111827",
  },
  section: {
    marginTop: "20px",
  },
  sectionTitle: {
    fontSize: "18px",
    fontWeight: "700",
    marginBottom: "10px",
    color: "#111827",
  },
  list: {
    paddingLeft: "20px",
    color: "#374151",
  },
  error: {
    color: "#b91c1c",
    fontWeight: "600",
  },
};