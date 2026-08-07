#!/bin/bash

# Download Wan2.1-T2V-1.3B base model
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ckpts/Self-Forcing/Wan2.1-T2V-1.3B

# Download Self-Forcing DMD checkpoint
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir ckpts/Self-Forcing
mv ckpts/Self-Forcing/checkpoints/self_forcing_dmd.pt ckpts/Self-Forcing/self_forcing_dmd.pt
rmdir ckpts/Self-Forcing/checkpoints 2>/dev/null
