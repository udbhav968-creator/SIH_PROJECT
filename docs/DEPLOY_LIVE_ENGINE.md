# Making the public website analyse photographs live

The Vercel site cannot run the models: they are about 440 MB, and a Vercel function is capped at 250 MB. So
the engine runs on a **Hugging Face Space** (free CPU hardware), and the website sends photograph analysis
there. Everything else, including every measured result, is still served by Vercel, so the pages keep
working even while the engine is asleep.

```
browser ──pages, measured reports──▶ Vercel (road-shield-ai-engine.vercel.app)
   └─────photograph analysis────────▶ Hugging Face Space (the engine, public-demo mode)
```

## One-time setup (about 20 minutes, most of it waiting for the build)

**1. Merge the pull request.** Use base `master` ← compare `audit-2026-10-03`. The Space builds from `master`.

**2. Create a free Hugging Face account** at https://huggingface.co/join.

**3. Create the Space.** Go to https://huggingface.co/new-space and set:

| Setting | Value |
|---|---|
| Space name | `road-shield-engine` |
| SDK | **Docker** → **Blank** |
| Hardware | **CPU basic** (free) |
| Visibility | **Public** |

Then press **Create Space**.

**4. Add the two files.** No git is needed; do it in the browser:

- **Files** → **Add file** → **Create a new file**. Name it `Dockerfile`, paste the whole of
  `deploy/huggingface/Dockerfile` from this repository, and press **Commit**.
- Open the Space's `README.md` → **Edit**. Replace everything with `deploy/huggingface/README.md`, then
  **Commit**.

**5. Wait for the build.** Watch the **Logs** tab; it takes 10–15 minutes the first time. When the status
says **Running**, open this, with your username in place of `<you>`:

```
https://<you>-road-shield-engine.hf.space/api/v1/ready
```

It should show `"ready": true`. The Space address on its own (`https://<you>-road-shield-engine.hf.space`)
is also the full website, running live.

**6. Point the website at it.** In `web/config.js`, set:

```js
window.ROAD_SHIELD_ENGINE_URL = "https://<you>-road-shield-engine.hf.space";
```

Commit, push, and merge as usual. Vercel redeploys by itself.

**7. Check it.** Open https://road-shield-ai-engine.vercel.app/inspect. The status pill should say
**live engine · online**. Upload a road photograph and press **Analyse**.

## How it behaves

- **Sleeping:** a free Space goes to sleep after a period with no visitors (as far as I know, about 48 hours).
  The first visitor afterwards sees "live engine is waking up". Their photograph is analysed automatically
  once the engine is up, usually 1–2 minutes later. Meanwhile, **Try an example photograph** shows recorded
  results.
- **Public-demo safeguards** (`ROAD_SHIELD_PUBLIC=1`):
  - Uploads are analysed and not stored.
  - Endpoints that write data (fleet reports, work orders, training, settings) are locked.
  - No request can make the server open a file outside `datasets/`.
  - Each visitor gets 20 analyses per minute and 12 MB per request.
- **Privacy:** photographs travel from the visitor's browser to Hugging Face's servers to be analysed. Say
  so if you demo with photos of people.
- **New code:** the Space builds from GitHub. After merging new commits, use the Space's **Settings** →
  **Factory rebuild**.
- **Speed:** a photograph takes about 5–20 s on the free 2-CPU hardware. A laptop is faster.

## If something goes wrong

| What you see | What to do |
|---|---|
| The build fails in **Logs** | Copy the last 30 lines of the log and send them to me. |
| `/api/v1/ready` says a model is missing | The model file is not on `master`. Check that the PR is merged. |
| The pill stays on **engine waking** for more than 5 minutes | Open the Space page. If it shows an error, use **Factory rebuild**. |
| You want to go back to recorded results only | Set `ROAD_SHIELD_ENGINE_URL` back to `""`. |
