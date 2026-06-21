"""One-shot prior quality eval driving the structure iteration. Fast (no synth except optional TexStat).

Reports, vs the codec-reconstruction gold standard (real tokens decoded = what the user says sounds
fine):
  DYNAMICS  — generated control std / real(codec) control std, per channel (loud/centroid/lat0/lat1).
              target ~1.0; <1 = dynamically flat (the perceptual "far from real").
  REACH     — style-space hit@1 / family (does style=t generate track t). must NOT regress.
  DIVERSITY — length-matched unique-token fraction gen vs real.

Usage: python scripts/eval_prior_quality.py --prior <ckpt> --comp <ckpt> [--style_cfg 2] [--temp 1.0]
"""
import sys, glob, math, random, argparse
sys.path.insert(0, '/home/btadeusz/code/ddsp_vae')
import numpy as np, torch
from collections import defaultdict
torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', required=True); ap.add_argument('--comp', required=True)
    ap.add_argument('--style_cfg', type=float, default=2.0)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    dev = 'cuda'
    from ddsp.latent_compressor import LatentCompressor
    from ddsp.prior.prior_discrete import PriorDiscrete
    from ddsp.prior.dataset import PriorTokenSequenceDataset
    from cli.generate_prior_discrete_audio import _sample_tokens
    comp = LatentCompressor.load_from_checkpoint(args.comp, strict=False).to(dev).eval()
    prior = PriorDiscrete.load_from_checkpoint(args.prior, strict=False).to(dev).eval()
    K = int(prior.codebook_size); N = int(prior.num_codebooks); VOC = K * N
    cache = glob.glob('/mnt/mariadata/datasets/mixed-norm/prior_tokens_cache_*.lmdb')[0]
    ds = PriorTokenSequenceDataset(cache, in_memory=False)
    byt = defaultdict(list)
    for i in random.sample(range(len(ds)), 3000):
        tk, cd, te = ds[i]; byt[int(te)].append(tk)
    tracks = sorted(byt)
    names = ['h06', 'h07', 'h08', 'h09', 'h10', 'h11', 'm00', 'm01', 'm02', 'm03', 'm04', 'm05']
    chan = ['loud', 'cent', 'lat0', 'lat1']
    n_tokens = int(math.ceil(args.seconds * (48000 / 128) / comp.compression_ratio))

    def uniq(t):
        ids = (t.long() + (torch.arange(N, device=t.device) * K)).reshape(-1)
        return int(torch.unique(ids).numel()) / VOC

    # real control std per channel + real style centroids
    real_std = {}; s_real = {}; real_u = []
    for t in tracks:
        win = torch.stack(byt[t][:60]).to(dev)
        real = comp.decode_codes(win)[:, :, :4]
        real_std[t] = real.reshape(-1, 4).std(0)
        s_real[t] = prior._encode_style(win).mean(0)
        real_u.append(np.mean([uniq(w[:n_tokens]) for w in win]))
    Creal = torch.stack([s_real[t] for t in tracks])

    # generate per track, collect dynamics ratios, style reach, diversity
    ratios = defaultdict(list); reach = fam = 0; gen_u = []
    s_gen_all = defaultdict(list)
    for t in tracks:
        ref = s_real[t].view(1, -1)
        ch_acc = []
        for sd in range(args.seeds):
            torch.manual_seed(sd); random.seed(sd)
            g = _sample_tokens(prior, n_tokens=n_tokens, primer_len=0, temperature=args.temp,
                               sampling='multinomial', device=dev, top_p=args.top_p, territory=-1,
                               style_vec=ref, style_cfg=args.style_cfg)
            gc = comp.decode_codes(g)[:, :, :4]
            ch_acc.append((gc.reshape(-1, 4).std(0) / (real_std[t] + 1e-6)).cpu().numpy())
            s_gen_all[t].append(prior._encode_style(g).view(-1))
            if sd == 0:
                gen_u.append(uniq(g[0][:n_tokens]))
        r = np.mean(ch_acc, 0)
        for i, ch in enumerate(chan):
            ratios[ch].append(r[i])
        # reach: majority nearest real centroid
        hits = [int(torch.cdist(sg.view(1, -1), Creal).argmin()) for sg in s_gen_all[t]]
        nn = max(set(hits), key=hits.count)
        reach += (nn == t); fam += ((nn < 6) == (t < 6))

    print(f'=== prior quality (style_cfg={args.style_cfg} temp={args.temp} top_p={args.top_p}) ===')
    print('DYNAMICS (gen std / real std, target ~1.0):')
    for ch in chan:
        print(f'  {ch:5s} {np.mean(ratios[ch]):.2f}  (per-track {[round(x,2) for x in ratios[ch]]})')
    print(f'REACH style-space hit@1 = {reach}/{len(tracks)}   FAMILY = {fam}/{len(tracks)}')
    print(f'DIVERSITY uniq gen={np.mean(gen_u):.3f} real={np.mean(real_u):.3f}')


if __name__ == '__main__':
    main()
