# Evaluation of the flow matching network

The generator is a conditional U-Net predicting the flow matching velocity on
4x12x16 latents of the frozen SD VAE: three levels at 128, 256 and 384
channels, self-attention at 3x4, time and the six scores injected by FiLM in
every residual block, 32.8M parameters. Its architecture and training
objective are verified. It has not yet been trained on the full data, so
nothing below says anything about image quality or about Moira.

## What the tests establish

| check | result | what it shows |
| --- | --- | --- |
| output shape | [8, 4, 12, 16] in and out | skips and resolutions line up |
| zero-initialised head | output exactly 0 at construction | training starts from zero velocity |
| time embedding, t = 0.30 vs 0.35 | distance 11.2, norm 11.3 | nearby times are distinguishable |
| null embedding gradient | 1024 = 4 dropped rows x 256 | condition dropout reaches the right rows only |
| FiLM at init / perturbed | identical / different outputs | conditioning starts inert and can act |
| first loss, overfit test | 1.681, predicted 1 + 0.821^2 = 1.675 | noise, target and latent scale agree |
| overfit, 16 fixed latents | 0.0064 at step 200, 1.6e-6 at 700 | enough capacity, gradients flow everywhere |
| full script, real latents, CPU | 1.72 to 1.59 in 14 steps | the training loop runs end to end |
| round trip, embeddings to images | max difference 0.0 | the submission is exactly reproducible |

The agreement between the first overfit loss and its prediction is the most
informative line. A zero prediction against the target x1 - x0 costs the
latent variance plus the unit noise variance, so a match confirms Gaussian
noise, the right target and latents at the scale the VAE produced them.

The frozen VAE is not the bottleneck. On held-out cars it reconstructs at MSE
0.0047 in [0, 1], about 23 dB PSNR. Colour and body silhouette survive, which
are the two attributes Moira responds to; badges and wheel spokes soften.

## What the tests cannot establish

The overfit test is blind to a wrong but consistent objective. Before the fix,
the interpolation ran from data to noise while the target pointed from noise
to data, and the network overfit that to 1e-6 just as well. Only sampling from
a trained model can confirm the direction is right.

Nothing yet shows generalisation, sample quality, or that the model uses the
scores on real data. The shortened run above has a conditioning control, and
it reports no difference between Moira = 0 and Moira = 9, as it should for a
model trained for 14 steps.

## Risks for the full run

**Conditioning may be ignored.** The FiLM projections start at zero and only
grow if the scores help the loss. `select_images.py` decodes the same noise at
Moira = 0 and 9 and reports both predicted scores; a gap near zero means the
conditioning failed.

**Moira = 9 is an extrapolation.** About 100 of 6000 cars score above 7 with
her. Guidance pushes towards that sparse region and may produce artefacts
rather than convincing sports cars.

**The objective is a weak judge.** The regressor reaches R^2 0.218 and
Spearman 0.510 for Moira, and only 12 held-out cars score 7 or more, so its
accuracy where it matters is barely measured. Best-of-N selection limits how
far we exploit it compared with gradient ascent, and the ensemble minimum
guards against a single network's blind spots.

**Memorisation.** 33M parameters on 5400 latents. Each candidate's distance to
its nearest real car is compared with how far real cars sit from each other,
and near-copies are dropped before selection.

## What the full run has to report

1. Train and EMA validation loss against step (`output/latent_fm/fm_log.csv`).
2. Predicted Moira at condition 0 against condition 9 on the same noise.
3. Predicted Moira for the dataset, all candidates, and the final 10.
4. Nearest-neighbour distance of the final 10 against the copy threshold.
5. The final images next to Moira's real top decile.

Items 2 to 4 are written to `output/latent_fm/selection_stats.json` by `select_images.py`.
