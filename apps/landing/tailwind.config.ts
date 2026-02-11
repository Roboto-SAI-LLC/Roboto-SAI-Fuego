import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#05060f",
        midnight: "#0b0f2a",
        navy: "#12163a",
        electric: "#38bdf8",
        neon: "#a855f7",
        pulse: "#22d3ee",
        steel: "#94a3b8",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["Sora", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(56, 189, 248, 0.25), 0 20px 60px rgba(15, 23, 42, 0.6)",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-16px)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "float-slow": "float 10s ease-in-out infinite",
        "fade-up": "fade-up 0.8s ease-out both",
      },
    },
  },
  plugins: [],
} satisfies Config;
