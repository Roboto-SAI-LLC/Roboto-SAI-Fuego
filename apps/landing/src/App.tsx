const features = [
  {
    title: "Inline Completions",
    description:
      "Predictive suggestions that stay in-flow and adapt to your style in TypeScript and Python.",
  },
  {
    title: "Smart Refactoring",
    description:
      "One-click improvements that keep intent intact and highlight the exact diff before applying.",
  },
  {
    title: "Doc Generation",
    description:
      "Generate clean, accurate docs and tests with context-aware prompts baked in.",
  },
];

const steps = [
  {
    title: "Install Extension",
    description: "Add Roboto SAI to VS Code in under a minute.",
  },
  {
    title: "Download Model",
    description: "Pull the optimized local model bundle for your machine.",
  },
  {
    title: "Code Locally",
    description: "Run completions and refactors with zero code leakage.",
  },
];

export default function App() {
  const formEndpoint =
    (import.meta as ImportMeta & {
      env?: { VITE_FORM_ENDPOINT?: string };
    }).env?.VITE_FORM_ENDPOINT ?? "";

  return (
    <div className="relative min-h-screen overflow-hidden bg-ink text-slate-100">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" />
      <div className="pointer-events-none absolute -left-24 top-[-160px] h-[420px] w-[420px] rounded-full orb orb-blue" />
      <div className="pointer-events-none absolute right-[-120px] top-[120px] h-[520px] w-[520px] rounded-full orb orb-violet" />
      <div className="pointer-events-none absolute bottom-[-200px] left-[10%] h-[520px] w-[520px] rounded-full orb orb-cyan" />

      <div className="relative z-10">
        <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-electric via-pulse to-neon text-sm font-semibold text-slate-900 shadow-glow">
              RS
            </div>
            <div>
              <p className="font-display text-lg">Roboto SAI</p>
              <p className="text-xs text-slate-400">Local AI pair programmer</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <a
              href="#features"
              className="hidden text-slate-300 transition hover:text-white md:inline"
            >
              Features
            </a>
            <a
              href="#waitlist"
              className="rounded-full border border-white/10 px-4 py-2 text-slate-200 transition hover:border-white/40 hover:text-white"
            >
              Join waitlist
            </a>
          </div>
        </header>

        <main className="mx-auto w-full max-w-6xl px-6 pb-24">
          <section className="grid items-center gap-10 py-10 md:grid-cols-[1.1fr_0.9fr] md:py-20">
            <div className="space-y-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1 text-xs uppercase tracking-[0.3em] text-slate-300">
                Private beta - 50 seats
              </span>
              <div className="space-y-4">
                <h1 className="font-display text-4xl leading-tight text-white md:text-6xl">
                  Roboto SAI is your
                  <span className="text-gradient"> local AI pair programmer</span>
                </h1>
                <p className="text-lg text-slate-300 md:text-xl">
                  Zero code leakage. Sub-300ms completions. Built for VS Code teams
                  who want speed without compromise.
                </p>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <a
                  href="#waitlist"
                  className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-electric via-pulse to-neon px-6 py-3 text-sm font-semibold text-slate-900 shadow-glow transition hover:scale-[1.02]"
                >
                  Request access
                </a>
                <a
                  href="#security"
                  className="inline-flex items-center justify-center rounded-full border border-white/10 px-6 py-3 text-sm font-semibold text-slate-100 transition hover:border-white/40"
                >
                  Security first
                </a>
              </div>
              <div className="flex flex-wrap gap-6 text-sm text-slate-400">
                <div>
                  <p className="text-white">TypeScript + Python</p>
                  <p>Inline completions and refactors</p>
                </div>
                <div>
                  <p className="text-white">Sub-300ms</p>
                  <p>Local model optimized for speed</p>
                </div>
              </div>
            </div>

            <div className="space-y-5">
              <div className="surface rounded-3xl p-6">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                  Live preview
                </p>
                <p className="mt-3 text-2xl font-display text-white">
                  Inline completions that feel instant.
                </p>
                <div className="mt-6 grid gap-4">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs text-slate-400">Latency</p>
                    <p className="text-3xl font-display text-white">280ms</p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs text-slate-400">Model</p>
                    <p className="text-lg text-white">Local, encrypted, on-device</p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center">
                  <p className="text-white">VS Code</p>
                  <p className="text-slate-400">Optimized extension</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center">
                  <p className="text-white">Private beta</p>
                  <p className="text-slate-400">50 builders</p>
                </div>
              </div>
            </div>
          </section>

          <section id="features" className="py-16">
            <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                  Features
                </p>
                <h2 className="font-display text-3xl text-white md:text-4xl">
                  Built for high-signal coding sessions
                </h2>
              </div>
              <p className="max-w-md text-sm text-slate-400">
                Designed for engineers who ship fast and keep their intellectual
                property locked down.
              </p>
            </div>
            <div className="grid gap-6 md:grid-cols-3">
              {features.map((feature) => (
                <div
                  key={feature.title}
                  className="surface rounded-3xl p-6 transition hover:-translate-y-1"
                >
                  <h3 className="font-display text-xl text-white">
                    {feature.title}
                  </h3>
                  <p className="mt-3 text-sm text-slate-300">
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>
          </section>

          <section id="how" className="py-16">
            <div className="grid gap-10 md:grid-cols-[0.9fr_1.1fr]">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                  How it works
                </p>
                <h2 className="mt-4 font-display text-3xl text-white md:text-4xl">
                  Get set up in minutes
                </h2>
                <p className="mt-4 text-sm text-slate-400">
                  Roboto SAI installs like a normal extension and keeps every
                  completion on your machine.
                </p>
              </div>
              <div className="grid gap-5">
                {steps.map((step, index) => (
                  <div
                    key={step.title}
                    className="flex items-start gap-4 rounded-2xl border border-white/10 bg-white/5 p-5"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-r from-electric to-neon text-sm font-semibold text-slate-900">
                      0{index + 1}
                    </div>
                    <div>
                      <h3 className="text-lg font-display text-white">
                        {step.title}
                      </h3>
                      <p className="mt-1 text-sm text-slate-300">
                        {step.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section id="security" className="py-16">
            <div className="surface rounded-3xl p-8 md:p-10">
              <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                <div>
                  <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs uppercase tracking-[0.3em] text-slate-300">
                    Security
                  </span>
                  <h2 className="mt-4 font-display text-3xl text-white">
                    Your code never leaves your machine
                  </h2>
                  <p className="mt-3 text-sm text-slate-300">
                    Roboto SAI runs locally with encrypted model files and no
                    outbound code transfer.
                  </p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 px-6 py-4 text-center text-sm text-slate-200">
                  Zero code leakage
                </div>
              </div>
            </div>
          </section>

          <section id="waitlist" className="py-16">
            <div className="grid gap-10 rounded-3xl border border-white/10 bg-white/5 p-8 md:grid-cols-[1.1fr_0.9fr] md:p-10">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400">
                  Waitlist
                </p>
                <h2 className="mt-4 font-display text-3xl text-white">
                  Request access to the private beta
                </h2>
                <p className="mt-3 text-sm text-slate-300">
                  We are onboarding 50 VS Code developers focused on TypeScript
                  and Python. Tell us where to send your invite.
                </p>
              </div>
              <div>
                <form
                  action={formEndpoint}
                  method="POST"
                  className="grid gap-4"
                >
                  <label className="text-sm text-slate-300" htmlFor="email">
                    Email address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required
                    autoComplete="email"
                    placeholder="you@company.com"
                    className="h-12 rounded-xl border border-white/10 bg-ink/60 px-4 text-sm text-white placeholder:text-slate-500 focus:border-electric focus:outline-none"
                  />
                  <button
                    type="submit"
                    className="inline-flex h-12 items-center justify-center rounded-xl bg-gradient-to-r from-electric via-pulse to-neon text-sm font-semibold text-slate-900 shadow-glow transition hover:scale-[1.01]"
                  >
                    Join the waitlist
                  </button>
                </form>
                <p className="mt-4 text-xs text-slate-400">
                  We will only use your email to reach out about the beta.
                </p>
              </div>
            </div>
          </section>
        </main>

        <footer className="border-t border-white/10 py-8">
          <div className="mx-auto flex w-full max-w-6xl flex-col items-start justify-between gap-4 px-6 text-sm text-slate-400 md:flex-row md:items-center">
            <span>Copyright Roboto SAI LLC 2026</span>
            <a
              href="https://github.com/Roboto-SAI"
              className="text-slate-300 transition hover:text-white"
            >
              GitHub
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}
