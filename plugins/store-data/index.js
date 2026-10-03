// store-data — onPostBuild egress for the SceneCut compute lane.
// Pattern credit: netlify-build-as-compute-kit templates/onpostbuild.
//
// Robustness twist for PATH B (cloud builds): @netlify/blobs is imported
// DYNAMICALLY inside onPostBuild with a graceful fallback — if the module
// cannot resolve in a cloud-build local-plugin context, we log the miss and
// still exit clean (the dist/out/ static-file copy in build.mjs is the
// fallback pickup path — results land on the deploy URL).
//
// Writes: every file under /tmp/scenecut-out/ → store "scenecut-bench"
//   runs/raw-<ts>/<filename>  — exact bytes
//   runs/latest               — pointer JSON {run_ts, files[], total_bytes}

import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";

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

    let getStore;
    try {
      ({ getStore } = await import("@netlify/blobs"));
    } catch (err) {
      console.log(
        `[store-data] @netlify/blobs not resolvable in this runtime (${String(err).slice(0, 120)}) — ` +
          `falling back to dist/out/ static pickup (build.mjs copies results there)`,
      );
      return;
    }

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
    } catch (err) {
      return utils.build.failPlugin(
        `[store-data] egress failed: ${err && err.message ? err.message : err}`,
        { error: err },
      );
    }
  },
};
