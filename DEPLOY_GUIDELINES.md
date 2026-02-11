Frontend Deploy Guidelines

Purpose: avoid wasting limited Render build minutes by validating frontend builds locally and via CI before creating Render deploys.

1) Local preflight (strongly recommended before pushing to main)
   - Run: `bash scripts/validate_frontend_build.sh`
   - This runs `npm ci`, `npm run build`, builds the production Docker image, runs it, and checks `/health`.

2) Automated preflight on GitHub (PRs and pushes to `main`)
   - A workflow `Validate Frontend Build` runs on PRs and pushes to `main`.
   - It will run `npm ci`, `npm run build`, build the production Docker image, and perform a container `/health` check.
   - Ensure this workflow passes before merging to `main`.

3) Manual deploy to Render (only after CI passes)
   - Auto-deploys have been disabled for the frontend service to avoid wasted build hours.
   - Use the GitHub workflow **Deploy to Render (manual)** (in the Actions tab) to trigger a Render deploy when ready.
   - The deploy workflow requires a `RENDER_API_KEY` secret in the repository settings.

4) If a front-end build fails
   - Reproduce locally with the script above, fix the build, open a PR with the fix, wait for the `Validate Frontend Build` workflow to pass, then run the manual deploy.

Notes:
 - The Render `frontend` service has `autoDeployTrigger: off` in `render.yaml` to prevent automatic builds.
 - The Dockerfile was updated to perform runtime substitution of `$PORT` via an entrypoint script and to include `user nginx;` in the nginx config so worker processes drop privileges.

If you want, I can set up an optional GitHub Action that automatically triggers a manual Render deploy when the `Validate Frontend Build` workflow completes successfully (requires a repo admin to add `RENDER_API_KEY` to secrets).