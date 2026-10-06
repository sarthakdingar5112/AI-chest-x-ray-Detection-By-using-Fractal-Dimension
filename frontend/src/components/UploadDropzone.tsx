import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function UploadDropzone({ onFile, disabled }: Props) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onFile(files[0]);
    },
    [onFile]
  );

  return (
    <motion.div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className="card"
      animate={{
        borderColor: drag ? "var(--accent)" : "var(--border)",
        boxShadow: drag ? "0 0 0 4px var(--accent-glow)" : "var(--shadow)",
      }}
      whileHover={disabled ? {} : { y: -2 }}
      style={{
        cursor: disabled ? "default" : "pointer",
        padding: "56px 32px",
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 14,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.dcm,.dicom,image/*"
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <motion.div
        animate={{ y: drag ? -4 : 0, scale: drag ? 1.06 : 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
      >
        <UploadIcon />
      </motion.div>
      <div>
        <div style={{ fontSize: 18, fontWeight: 600 }}>
          Drop a chest X-ray here
        </div>
        <div className="muted" style={{ fontSize: 14, marginTop: 4 }}>
          or click to browse · PNG, JPG or DICOM · up to 25&nbsp;MB
        </div>
      </div>
    </motion.div>
  );
}

function UploadIcon() {
  return (
    <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
      <circle cx="26" cy="26" r="25" stroke="url(#u)" strokeWidth="1.4" opacity="0.5" />
      <path d="M26 34 V18 M26 18 L20 24 M26 18 L32 24" stroke="url(#u)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M17 34 H35" stroke="url(#u)" strokeWidth="2.2" strokeLinecap="round" />
      <defs>
        <linearGradient id="u" x1="0" y1="0" x2="52" y2="52">
          <stop stopColor="#35e0d4" />
          <stop offset="1" stopColor="#4d8dff" />
        </linearGradient>
      </defs>
    </svg>
  );
}
