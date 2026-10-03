// store-data — onPostBuild egress for the SceneCut compute lane.
// Pattern credit: netlify-build-as-compute-kit templates/onpostbuild (Path A, E2E-proven).
// Variant: egresses EVERY file under /tmp/scenecut-out/ (banner + results + logs),
// not a single JSONL. NETLIFY_BLOBS_CONTEXT is available here (onPostBuild phase)
// and getStore() resolves it automatically — no PAT in-runtime.
//
// Failure contract: any problem fails the plugin loudly via utils.build.failPlugin.

import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { getStore } from "@netlify/blobs";

async function listFiles(dir) {
  const out = [];
  for (const name of await readdir(dir)) {
    const p = join(dir, name);
    const s = await stat(p);
    if (s.isFile()) out.push({ name, path: p, size: s.size });
  }
  return out;
}

export default {
  onPostBuild: async ({ inputs, utils }) => {
    const srcDir = inputs.path || "/tmp/scenecut-out";
    const storeName = inputs.store || "scenecut-bench";

    try {
      const files = await listFiles(srcDir);
      if (files.length === 0) {
        return utils.build.failPlugin(
          `[store-data] no files in ${srcDir} — did the build command run before this plugin?`,
        );
      }

      const runTs = new Date().toISOString();
      const runPrefix = `runs/raw-${runTs.replace(/[:.]/g, "-")}`;
      const store = getStore(storeName);

      const uploaded = [];
      for (const f of files) {
        const bytes = await readFile(f.path);
        const key = `${runPrefix}/${f.name}`;
        await store.set(key, bytes, { contentType: "application/octet-stream" });
        uploaded.push({ key, size: f.size });
        console.log(`[store-data] stored ${key} (${f.size} B)`);
      }

      await store.setJSON("runs/latest", {
        run_ts: runTs,
        files: uploaded,
        total_bytes: uploaded.reduce((a, b) => a + b.size, 0),
      });

      console.log(
        `[store-data] EGRESS_OK n=${uploaded.length} bytes=${uploaded.reduce((a, b) => a + b.size, 0)} store=${storeName}`,
      );
      console.log(`[store-data] NETLIFY_BLOBS_CONTEXT resolved from plugin env (onPostBuild phase)`);
    } catch (err) {
      return utils.build.failPlugin(
        `[store-data] egress failed: ${err && err.message ? err.message : err}`,
        { error: err },
      );
    }
  },
};
