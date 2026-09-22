"""Trains the conditional flow matching U-Net on the cached latents.

Keeps the EMA weights with the lowest held-out loss and stops once that has
not improved for PATIENCE steps. Resumes from output/fm_ckpt.pth if a run
was cut off. The best weights end up in output/fm_ema.pth.
"""
import os
import copy
import time
from torch.utils.data import TensorDataset, DataLoader

from constants import *
from model_fm import UNet

STEPS = 40000
BATCH = 128
LR = 1e-4
EMA_DECAY = 0.999
P_DROP = 0.1
LOG_EVERY = 200
SAVE_EVERY = 2000
PATIENCE = 2000   # steps without val improvement before stopping

LATENTS = os.path.join(OUTDIR, 'latents.pth')
CKPT = os.path.join(OUTDIR, 'fm_ckpt.pth')
EMA_BEST = os.path.join(OUTDIR, 'fm_ema_best.pth')
EMA_OUT = os.path.join(OUTDIR, 'fm_ema.pth')
LOG = os.path.join(OUTDIR, 'fm_log.csv')


def fm_loss(model, x1, c, drop, g=None):
    # t=0 is noise, t=1 is data, and the target points from noise to data
    x0 = torch.randn(x1.shape, device=x1.device, generator=g)
    t = torch.rand(x1.shape[0], device=x1.device, generator=g)

    t_ = t[:, None, None, None]
    xt = (1 - t_) * x0 + t_ * x1
    target = x1 - x0

    pred = model(xt, t, c, drop)
    return nn.functional.mse_loss(pred, target)


@torch.no_grad()
def val_loss(model, x1, c):
    # Fixed noise and times, so values are comparable across steps
    g = torch.Generator(device=x1.device).manual_seed(SEED)
    drop = torch.zeros(len(x1), dtype=torch.bool, device=x1.device)
    return fm_loss(model, x1, c, drop, g).item()


@torch.no_grad()
def ema_update(ema, model, decay=EMA_DECAY):
    # ema = decay * ema + (1 - decay) * live
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)


def main():
    if os.path.exists(EMA_OUT):
        print('generator: trained weights found, skipping')
        return

    torch.manual_seed(SEED)

    latent_dict = torch.load(LATENTS, map_location=device, weights_only=False)
    x_train = latent_dict['latents_train'].to(device)
    c_train = (latent_dict['scores_train'] / MAX_SCORE).to(device)
    x_test = latent_dict['latents_test'].to(device)
    c_test = (latent_dict['scores_test'] / MAX_SCORE).to(device)

    # Data already on the GPU, so no workers
    loader = DataLoader(TensorDataset(x_train, c_train), batch_size=BATCH, shuffle=True, drop_last=True)

    model = UNet().to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    history = list()
    best = float('inf')
    best_step = 0

    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
        history = ck['history']
        best = ck.get('best', float('inf'))
        best_step = ck.get('best_step', step)
        print(f'generator: resuming from step {step}, best val {best:.4f} at step {best_step}')

    model.train()
    running = 0.0
    start_step = step
    t_start = time.time()
    stop = False

    while step < STEPS and not stop:
        for x1, c in loader:
            # Drop the scores on ~10% of samples, so CFG has an unconditional model
            drop = torch.rand(len(x1), device=device) < P_DROP

            loss = fm_loss(model, x1, c, drop)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model)

            step += 1
            running += loss.item()

            if step == start_step + 100:
                ms = (time.time() - t_start) / 100 * 1000
                print(f'{ms:.0f} ms/step, about {ms * (STEPS - step) / 60000:.0f} min left at most')

            if step % LOG_EVERY == 0:
                val = val_loss(ema, x_test, c_test)
                history.append((step, running / LOG_EVERY, val))
                print(f'step {step} train {running / LOG_EVERY:.4f} val_ema {val:.4f}')
                running = 0.0

                # Keep the EMA weights with the lowest held-out loss
                if val < best:
                    best = val
                    best_step = step
                    torch.save(ema.state_dict(), EMA_BEST)
                if step - best_step >= PATIENCE:
                    print(f'generator: no improvement since step {best_step}, stopping')
                    stop = True

            if step % SAVE_EVERY == 0 or stop:
                torch.save({
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'opt': opt.state_dict(),
                    'step': step,
                    'history': history,
                    'best': best,
                    'best_step': best_step,
                }, CKPT)

            if step >= STEPS or stop:
                break

    # Best weights become the final ones; only now, so a crashed run is never mistaken for a finished one
    if not os.path.exists(EMA_BEST):
        torch.save(ema.state_dict(), EMA_BEST)
    os.replace(EMA_BEST, EMA_OUT)
    pd.DataFrame(history, columns=['step', 'train', 'val_ema']).to_csv(LOG, index=False)
    print(f'generator: saved {EMA_OUT} from step {best_step}, val {best:.4f}')


if __name__ == '__main__':
    main()
