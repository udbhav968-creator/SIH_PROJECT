/* ============================================================================
   Where the live inference engine runs.

   The public site (Vercel) cannot hold the models, so photograph analysis is
   sent to the engine running on a Hugging Face Space (deploy/huggingface/).
   Put that Space's address here, e.g.
       window.ROAD_SHIELD_ENGINE_URL = "https://your-name-road-shield-engine.hf.space";
   Leave it empty to show recorded results only. When this page is served BY the
   engine itself (laptop or the Space), it is ignored: same-origin wins.
   ========================================================================= */
window.ROAD_SHIELD_ENGINE_URL = "";
