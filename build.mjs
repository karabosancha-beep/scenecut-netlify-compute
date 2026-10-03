// SceneCut compute-lane build entry (Netlify Path B — cloud builds).
// Reads the job spec from INCOMING_HOOK_BODY (URL-encoded form data from a build hook,
// e.g. `job=pyprobe` or `job=sbd&n=200`), dispatches to a runner, captures a full log,
// and leaves dist/out/ publishable — the static pickup path for the orchestrator:
//   https://<branch>--<site>.netlify.app/out/{banner.json,result.json,log.txt,...}
import { writeFileSync, mkdirSync, cpSync } from "node:fs";
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
const t0 = Date.now();
function runBash(script, args = []) {
  const cmd = `bash runners/${script} ${args.join(" ")}`;
  try {
    const stdout = execSync(cmd, {
      encoding: "utf8",
      timeout: 9 * 60 * 1000, // leave headroom under the build cap
      maxBuffer: 32 * 1024 * 1024,
      env: { ...process.env, OUT_DIR: OUT },
    });
    return { ok: true, stdout };
  } catch (e) {
    return { ok: false, stdout: String(e.stdout ?? "") + String(e.stderr ?? "") + `\nEXIT ${e.status}` };
  }
}

let result;
switch (spec.job) {
  case "probe": {
    let py = "n/a";
    try { py = execSync("python3 --version").toString().trim(); } catch {}
    result = { job: "probe", ok: true, python: py, node: process.version };
    break;
  }
  case "pyprobe": {
    const r = runBash("pyprobe.sh");
    result = { job: "pyprobe", ok: r.ok, log_tail: r.stdout.split("\n").slice(-14).join("\n") };
    writeFileSync(`${OUT}/log.txt`, r.stdout);
    break;
  }
  default:
    result = { job: spec.job, ok: false, error: `unknown job '${spec.job}'` };
}

result.wall_s = Math.round((Date.now() - t0) / 1000);
result.finished_at = new Date().toISOString();
writeFileSync(`${OUT}/result.json`, JSON.stringify(result, null, 2));

// --- static pickup path ------------------------------------------------------
// dist/out/* is served by the deploy URL — a zero-dependency path that works
// even when the Blobs plugin cannot resolve @netlify/blobs in a cloud-build runtime.
cpSync(OUT, "dist/out", { recursive: true });
writeFileSync(
  "dist/index.html",
  `<!doctype html><meta charset="utf-8"><title>scenecut compute lane</title>
<pre>${JSON.stringify(banner, null, 2)}</pre>
<p><a href="out/result.json">result.json</a> · <a href="out/log.txt">log.txt</a></p>`,
);
console.log(`[scenecut-lane] DONE job=${spec.job} ok=${result.ok !== false} wall=${result.wall_s}s`);
