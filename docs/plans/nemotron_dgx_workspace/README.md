# Architectural Guide: Nemotron-3 Nano 30B Fine-Tuning
This workspace contains configuration settings to fine-tune the NVIDIA Nemotron-3 Nano 30B (A3B) hybrid model inside a 128 GB memory boundary.
By targeting only dense backbone modules (`q_proj`, `v_proj`, etc.) and bypassing the 128 mixture-of-experts layers, we drop the training footprint significantly to avoid OOM crashes.