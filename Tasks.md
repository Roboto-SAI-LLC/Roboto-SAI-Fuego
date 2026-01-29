You are an expert full-stack agent working on the Roboto SAI project (React + Vite frontend, FastAPI Python backend with Supabase auth, roboto-sai-sdk, and Grok/xAI integration).

Implement the following features completely and securely:

1. **Stripe Payment Integration (Checkout + Subscriptions)**
   - Install stripe and @stripe/stripe-js in frontend, stripe in backend.
   - Add env variables: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_ID (for a monthly subscription tier).
   - Backend (main.py or new payments.py):
     - POST /api/create-checkout-session: create Stripe Checkout Session for subscription mode, return session_id.
     - POST /api/stripe-webhook: raw body endpoint, verify signature, handle checkout.session.completed → update user row in Supabase "users" table with subscription_status="active", subscription_id, and current_period_end.
     - Add premium check middleware or helper to gate premium features (higher limits, voice mode unlimited, custom tools, etc.).
   - Frontend:
     - Add "Upgrade to Premium" button in header or chat sidebar (visible when not premium).
     - On click → fetch /api/create-checkout-session → redirectToCheckout with Stripe.js.
     - Add success/cancel pages or query param handling to show toast.
     - Gate premium UI/features based on authStore user metadata subscription_status.

2. **Fully Functional Self-Code Modification System with Agentic Tool Calling**
   - Activate and complete self_code_modification.py module.
   - Integrate agentic loop inside Roboto SAI (Roboto_SAI.py or new agent_loop.py):
     - Use LangChain or custom ReAct-style agent that can call tools.
     - Tools include:
       - Local script execution (safe sandboxed via subprocess with whitelist).
       - MCP servers from mcp.json (upstash, render, etc.) — dynamically load available tools from active MCP endpoints.
       - Self-modification tool: read/write approved files (only in project dir, with hash verification and backup before write).
     - Add safety: every modification requires user confirmation in UI unless full_autonomy=True, versioned backups, rollback capability.
     - Enable the agent to propose and apply code changes when asked (e.g., "add a new feature").

3. **MCP Servers Management Tab**
   - Add new route /mcp or sidebar menu item "MCP Tools".
   - New page/component MCPTab.tsx:
     - List all MCP servers from mcp.json + any dynamically discovered.
     - Toggle each server on/off (store in localStorage or Supabase user metadata).
     - For each active server, list its available tools/endpoints.
     - Individual toggles per tool.
     - "Test Connection" button per server.
     - Display status (healthy/unhealthy) using health endpoints if available.
     - Persist config and use it to enable/disable tools in the agent loop.

4. **Full Settings / Profile Tab (Grok-style)**
   - Add new route /settings with sidebar menu item "Settings".
   - Tabs or sections:
     - Profile: display name, email, avatar (upload via Supabase storage), bio field.
     - Account: change password, change email, delete account (with confirmation).
     - Appearance: theme selector (dark/light/system + custom Roboto fire themes).
     - Notifications: toggle email/push when available.
     - Billing: show subscription status, manage/cancel Stripe subscription (portal link), payment history.
     - MCP Tools: link or embed the MCP tab.
     - Privacy & Data: export conversations, clear cache, etc.
     - About: version, credits to Roberto Villarreal Martínez, sigil 929, merge date.
   - Use existing authStore and Supabase client for updates.

Ensure all changes are type-safe, clean, and match existing code style (ember/fire theme, framer-motion animations, shadcn/ui components). Add necessary imports, update routes in App.tsx, sidebar in ChatSidebar.tsx, and header links.

After implementation, test:
- Stripe test mode checkout flow end-to-end.
- Agent can propose and apply a harmless code change.
- MCP toggles persist and affect available tools.
- Settings updates save correctly to Supabase.

Commit with message: "feat: Stripe payments, full self-modification agent, MCP management tab, complete Settings tab - by Eve for Roberto 929"