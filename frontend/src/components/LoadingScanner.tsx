import { motion } from "framer-motion";

/** A clinical "scanning line" sweep over the uploaded X-ray while the backend works. */
export default function LoadingScanner({ src }: { src: string }) {
  return (
    <div
      className="card"
      style={{ position: "relative", overflow: "hidden", padding: 0, aspectRatio: "1 / 1" }}
    >
      <img
        src={src}
        alt="scanning"
        style={{ width: "100%", height: "100%", objectFit: "cover", filter: "grayscale(1) contrast(1.05) brightness(0.8)" }}
      />
      {/* sweeping scan line */}
      <motion.div
        initial={{ top: "0%" }}
        animate={{ top: ["0%", "100%", "0%"] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          height: 3,
          background: "linear-gradient(90deg, transparent, var(--accent), transparent)",
          boxShadow: "0 0 24px 6px var(--accent-glow)",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(180deg, rgba(7,11,18,0.1), rgba(7,11,18,0.55))",
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "center",
          paddingBottom: 22,
        }}
      >
        <div className="badge mono" style={{ letterSpacing: "0.08em" }}>
          <motion.span
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.4, repeat: Infinity }}
          >
            ● ANALYZING TEXTURE
          </motion.span>
        </div>
      </div>
    </div>
  );
}
