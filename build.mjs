// SceneCut compute-lane build entry.
// Reads the job spec from INCOMING_HOOK_BODY (URL-encoded form data from a build hook,
// e.g. `job=probe`), dispatches to a runner, and always leaves dist/ publishable.
import { writeFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { execSync } from "node:child_process";
import os from "node:os";

const OUT = "/tmp/scenecut-out";
mkdirSync(OUT, { recursive: true });
mkdirSync("dist", { recursive: true });

// --- job spec -------------------------------------------------------------
let spec = { job: "probe" };
const raw = process.env.INCOMING_HOOK_BODY;
if (raw) {
  const params = new URLSearchParams(raw.includes("=") ? raw : `job=${raw}`);
  spec = Object.fromEntries(params.entries());
}

// Self-identification: proves WHERE the build ran (the whole point of Path B).
const banner = {
  ran_at: new Date().toISOString(),
  hostname: os.hostname(),
  platform: `${os.platform()}/${os.arch()}`,
  cpus: os.cpus().length,
  cpu_model: os.cpus()[0]?.model ?? "?",
  mem_gb: Math.round(os.totalmem() / 1e9),
  job: spec,
};
writeFileSync(`${OUT}/banner.json`, JSON.stringify(banner, null, 2));
console.log(`[scenecut-lane] banner: ${JSON.stringify(banner)}`);

// --- dispatch ---------------------------------------------------------------
const runners = {
  probe: () => {
    // Minimal end-to-end proof: compute + timestamps + a python3 sanity check.
    let py = "n/a";
    try { py = execSync("python3 --version").toString().trim(); } catch {}
    return {
      job: "probe",
      ok: true,
      python: py,
      node: process.version,
      evidence: "cloud build ran this code; see banner.json timestamps",
    };
  },
};

const runner = runners[spec.job] ?? runners.probe;
let result;
try {
  result = await runner();
} catch (e) {
  result = { job: spec.job, ok: false, error: String(e) };
}

// Result JSONL (append semantics across retries are handled by ts keys).
writeFileSync(
  `${OUT}/result-${Date.now()}.json`,
  JSON.stringify({ ...result, finished_at: new Date().toISOString() }, null, 2),
);

// dist/ must exist and be publishable.
writeFileSync(
  "dist/index.html",
  `<!doctype html><meta charset="utf-8"><title>scenecut compute lane</title>
<pre>${JSON.stringify(banner, null, 2)}</pre>`,
);
console.log(`[scenecut-lane] DONE job=${spec.job} ok=${result.ok !== false}`);
