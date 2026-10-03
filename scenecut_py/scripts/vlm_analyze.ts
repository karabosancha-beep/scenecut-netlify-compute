/**
 * VLM Analysis Bridge — called by Python CLI or Next.js API routes.
 *
 * Usage: bun vlm_analyze.ts <video_path> <prompt_type> <output_path>
 *
 * prompt_type: "scene" (14-point cinematography analysis)
 *              "seam" (transition/cut analysis)
 *              "falsesplit" (false-split detection only)
 *
 * If VLM_PROMPT env var is set, it overrides the built-in prompt.
 */
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'fs';

const VIDEO_PATH = process.argv[2];
const PROMPT_TYPE = process.argv[3] || 'scene';
const OUTPUT_PATH = process.argv[4];

if (!VIDEO_PATH || !OUTPUT_PATH) {
  console.error('Usage: bun vlm_analyze.ts <video_path> <prompt_type> <output_path>');
  process.exit(1);
}

const SCENE_PROMPT = `You are an expert film scholar and cinematographer. Analyze this video clip.

Provide ALL of the following with 2-3 sentences each:
1. FILM: What film? Director? DP? Year? Confidence?
2. SHOT SIZE: Precise classification
3. ANGLE: Camera angle + psychological effect
4. MOVEMENT: Static or moving? Direction, speed, purpose?
5. LENS: Focal length estimate + depth of field
6. COMPOSITION: Framing technique + specific elements
7. LIGHTING: Key light direction/quality + style
8. COLOR: Palette + warm/cool + saturation + story function
9. MISE-EN-SCENE: Set/props/costume/blocking + meaning
10. CONTENT: What is happening? Setting, characters, action
11. MOOD: Emotional atmosphere
12. WHY BEAUTIFUL: What makes this shot-worthy
13. INFLUENCES: Painting/photo/film references
14. STUDENT LESSON: Technique to study + replicate`;

const SEAM_PROMPT = `You are an expert film editor analyzing a transition between two shots. The cut/transition point is at the approximate midpoint of this clip.

Analyze this transition:
1. TRANSITION TYPE: Hard cut, match cut, jump cut, dissolve, fade, wipe, or other? Be specific.
2. WHAT CHANGES: Describe what changes across the transition.
3. MATCH QUALITY: If match cut, what elements match?
4. EMOTIONAL EFFECT: Narrative/emotional purpose?
5. TECHNICAL QUALITY: Clean cut? Artifacts?
6. RATING: Effectiveness (1-10) with brief explanation.`;

// D17 calibration: the S4 prompts over-judged FALSE on low-contrast
// similar-content shots (snow/fur: 8/14 vs GT ~2/17) — six FALSE causes
// enumerated, zero GENUINE exemplars, binary-only output. This variant:
// operational "shot" definition, GENUINE classes that look continuous,
// evidence checklist BEFORE verdict, 3-way output (ambiguity abstains).
const FALSESPLIT_PROMPT = `You are a professional film EDITOR auditing an automatic scene-cut detector. The clip contains a candidate cut near the midpoint. A "shot" means one continuous camera recording: recording starts, the camera may pan/tilt/zoom/re-focus, recording ends. EVERY edit joining two recordings is a GENUINE cut — even when the content looks similar.

GENUINE (real cut): at the midpoint the clip switches to a different camera recording. All of these still count as GENUINE: same subject, different framing (wide to close-up); same subject from a different angle or camera position; a cutaway to another part of the same scene or habitat; match cuts between visually similar shots; hard cuts, dissolves, fades, wipes; two shots with the same colors, lighting, and subject matter.
FALSE (detector error) ONLY if the clip is ONE continuous recording fooled by an in-shot change: continuous pan/tilt/zoom/tracking (the picture slides or scales coherently across MANY frames — no single frame where the whole picture changes); focus pull or motion blur; lighting/exposure drifting gradually (sun, cloud, snow glare) with no one-frame step; a large subject moving across an otherwise stable frame.

Work through the evidence FIRST:
1. FRAMING: does shot size/angle change at the midpoint?
2. SUBJECT: does the subject's screen position/scale jump discontinuously?
3. BACKGROUND: does the background or geographic context change?
4. MOTION: is there coherent multi-frame camera motion that could explain the change?
5. LIGHTING: one-frame step, or gradual drift?
Then answer:
6. VERDICT: GENUINE or FALSE or UNCERTAIN
7. CONFIDENCE: 0-100%

Rules: a cut between two shots of the SAME subject or scene is still GENUINE. Content similarity is NOT evidence of error; only continuous camera motion is. If the checklist evidence is genuinely balanced, answer UNCERTAIN — do not default to FALSE because the pictures look alike.`;

const SHOTCHANGE_PROMPT = `You are a video analysis expert. Watch this clip carefully from start to finish.

Your ONLY job: determine whether this clip contains ANY shot change — a cut, transition, dissolve, or any visible switch to a DIFFERENT camera recording — anywhere in the clip. Camera movement WITHIN one continuous recording (pan, zoom, tilt) is NOT a shot change. But a re-frame or angle change on the SAME subject IS a shot change (a different recording), even when the content looks similar.

Answer:
1. SHOT CHANGE: YES or NO
2. CONFIDENCE: 0-100%
3. IF YES: Where approximately (beginning / middle / end), and what changes?

Be concise.`;

function getPrompt(): string {
  // Env var override (for custom prompts from API routes / autotune)
  if (process.env.VLM_PROMPT) return process.env.VLM_PROMPT;
  switch (PROMPT_TYPE) {
    case 'seam': return SEAM_PROMPT;
    case 'falsesplit': return FALSESPLIT_PROMPT;
    case 'shotchange': return SHOTCHANGE_PROMPT;
    default: return SCENE_PROMPT;
  }
}

async function main() {
  const zai = await ZAI.create();
  const videoBuffer = fs.readFileSync(VIDEO_PATH);
  const base64Video = videoBuffer.toString('base64');
  const dataUrl = `data:video/mp4;base64,${base64Video}`;

  const response = await zai.chat.completions.createVision({
    model: 'glm-5v-turbo',
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: getPrompt() },
        { type: 'video_url', video_url: { url: dataUrl } }
      ]
    }],
    thinking: { type: 'disabled' }
  });

  const content = response.choices[0]?.message?.content || '';
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify({
    ok: true,
    model: response.model || 'unknown',
    content,
    content_length: content.length,
    usage: response.usage,
  }, null, 2));
  console.log(JSON.stringify({
    ok: true,
    model: response.model || 'unknown',
    content_length: content.length,
  }));
}

main().catch(e => {
  console.log(JSON.stringify({ ok: false, error: e.message }));
  process.exit(1);
});
