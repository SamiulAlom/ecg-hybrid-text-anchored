"""Train/evaluate one configuration. Every reported number in the paper comes from a run of this script.

Example (leave-one-database-out, proposed method):
  python train.py --prep /kaggle/working/prep --mode loso --target GEORGIA --head text --text_emb emb.npz \
      --labels mask_pseudo --hierarchy --mixstyle --seed 0 --out results/
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from data import IMPLIES, build_vocab, drop_annotations, label_matrices, load_prep, split_indices
from labels import LABELS
from models import ECGModel, GradReverse

WIN = 1000  # 10 s at 100 Hz


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--text_emb", default=None, help="npz from embed_labels.py (needed for --head text)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", choices=["loso", "indist"], default="loso")
    ap.add_argument("--target", default="GEORGIA")
    ap.add_argument("--head", choices=["linear", "text", "query", "hybrid"], default="linear")
    ap.add_argument("--labels", choices=["naive", "mask", "mask_pseudo"], default="naive",
                    help="naive: unknown=negative; mask: ignore unknown; mask_pseudo: + EMA-teacher pseudo-labels")
    ap.add_argument("--hierarchy", action="store_true", help="complete implied parent labels")
    ap.add_argument("--dg", choices=["none", "coral", "dann", "groupdro"], default="none")
    ap.add_argument("--dg_w", type=float, default=None, help="DG loss weight (coral 1.0, dann 0.1 by default)")
    ap.add_argument("--mixstyle", action="store_true")
    ap.add_argument("--lead_drop", type=float, default=0.1)
    ap.add_argument("--holdout", nargs="*", default=[], help="SNOMED codes never supervised (open-vocabulary test)")
    ap.add_argument("--min_source", type=int, default=10, help="positives needed for a source to 'annotate' a code")
    ap.add_argument("--drop_rate", type=float, default=0.0, help="E4: fraction of (source, code) annotations removed")
    ap.add_argument("--loss", choices=["bce", "asl"], default="asl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--pseudo_w", type=float, default=0.5)
    ap.add_argument("--pseudo_hi", type=float, default=0.9)
    ap.add_argument("--pseudo_lo", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0, help="model initialisation / batching seed")
    ap.add_argument("--split_seed", type=int, default=0, help="train/val split seed (kept fixed across model seeds)")
    ap.add_argument("--max_train", type=int, default=0, help="debug: subsample training set")
    # ---- method v2 (all off by default: defaults reproduce the original runs exactly) ----
    ap.add_argument("--hier_soft", action="store_true",
                    help="hierarchy as constraint: implied parents completed only where the source annotates the "
                         "parent (no positive-only columns), plus a consistency loss p(parent) >= p(child) and "
                         "hierarchy-coherent teacher targets. Mutually exclusive with --hierarchy")
    ap.add_argument("--hier_w", type=float, default=1.0, help="weight of the hierarchy consistency loss")
    ap.add_argument("--pseudo_soft", action="store_true",
                    help="pseudo-labels as soft teacher probabilities with BCE (instead of hard 0/1 targets with ASL)")
    ap.add_argument("--pseudo_start", type=int, default=0, help="epoch at which the pseudo-label loss starts")
    ap.add_argument("--pseudo_ramp", type=float, default=0.3,
                    help="linear ramp of the pseudo-label weight, as a fraction of all steps, from pseudo_start")
    ap.add_argument("--ema_warmup", action="store_true",
                    help="EMA decay min(ema, (1+t)/(10+t)): the teacher does not carry random-init weights")
    ap.add_argument("--eval_ema", action="store_true",
                    help="validate/select/test the EMA (teacher) weights instead of the student; generic, any method")
    ap.add_argument("--residual", choices=["none", "query", "pool", "both"], default="none",
                    help="text/query head: class-specific residual (bias + readout) for seen diagnoses only")
    # ---- hybrid head (--head hybrid) ----
    ap.add_argument("--text_w", type=float, default=1.0, help="hybrid: weight of the text-head loss")
    ap.add_argument("--seen_score", choices=["linear", "avg"], default="linear",
                    help="hybrid: seen diagnoses scored by the linear head or the mean of both probabilities "
                         "(unseen diagnoses are always scored by the text head)")
    ap.add_argument("--text_detach", action="store_true",
                    help="hybrid: text head trained on detached tokens (no text-head gradient into the encoder)")
    ap.add_argument("--diag", action="store_true", help="log pseudo-label statistics and teacher val AUROC per epoch")
    return ap.parse_args(argv)


def asl_loss(logits, y, gamma_neg=2.0, clip=0.05):
    p = torch.sigmoid(logits)
    p_m = (p - clip).clamp(min=0)
    lp = torch.log(p.clamp(min=1e-8))
    ln = torch.log((1 - p_m).clamp(min=1e-8))
    return -(y * lp + (1 - y) * ln * p_m.pow(gamma_neg))


def elem_loss(kind, logits, y):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return asl_loss(logits, y)


def coral(feat, dom):
    loss, covs = feat.new_zeros(()), []
    for d in dom.unique():
        f = feat[dom == d]
        if len(f) > 2:
            f = f - f.mean(0, keepdim=True)
            covs.append(f.T @ f / (len(f) - 1))
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + (covs[i] - covs[j]).pow(2).mean()
    return loss


def random_crop(x, n_valid):
    """x [B,12,L] -> [B,12,WIN], each record cropped at a random start inside its valid samples."""
    B, _, L = x.shape
    hi = (n_valid.clamp(min=WIN) - WIN).clamp(max=L - WIN)
    start = (torch.rand(B, device=x.device) * (hi + 1).float()).long()
    idx = start[:, None] + torch.arange(WIN, device=x.device)[None]
    return torch.gather(x, 2, idx[:, None, :].expand(-1, 12, -1))


def augment(x, lead_drop):
    B = x.size(0)
    x = x * torch.empty(B, 1, 1, device=x.device).uniform_(0.8, 1.2)
    x = x + 0.02 * torch.randn_like(x)
    if lead_drop > 0:
        x = x * (torch.rand(B, 12, 1, device=x.device) > lead_drop).float()
    return x


def predict(model, X, nvalid, idx, text, device, bs=256, stride=500):
    """Mean probability over 10 s windows (stride 5 s) inside each record's valid part."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.from_numpy(np.asarray(X[b], dtype=np.float32)).to(device).clamp(-5, 5)
            nv = torch.from_numpy(nvalid[b]).to(device).clamp(min=WIN)
            acc, cnt = 0, torch.zeros(len(b), 1, device=device)
            for k in range(0, xb.shape[2] - WIN + 1, stride):
                valid = ((k + WIN) <= nv).float()[:, None] if k > 0 else torch.ones(len(b), 1, device=device)
                if valid.sum() == 0:
                    break
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    lg, _ = model(xb[:, :, k:k + WIN], text)
                acc = acc + torch.sigmoid(lg.float()) * valid
                cnt = cnt + valid
            out.append((acc / cnt).cpu().numpy())
    return np.concatenate(out)


def evaluate(P, Y, A, cols, thresholds=None, min_pos=5):
    """Macro metrics over labels annotated in the evaluated records (A==1) with >= min_pos positives.
    f1: threshold tuned on source validation data; f1_oracle: best threshold on the evaluated data (upper bound)."""
    per = {}
    grid = np.linspace(0.02, 0.98, 49)
    for j in cols:
        m = A[:, j] > 0
        y, p = Y[m, j], P[m, j]
        if m.sum() == 0 or y.sum() < min_pos or y.sum() == len(y):
            continue
        thr = 0.5 if thresholds is None else thresholds.get(j, 0.5)
        per[j] = dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                      f1=f1_score(y, p >= thr), f1_oracle=max(f1_score(y, p >= t) for t in grid),
                      n_pos=int(y.sum()), n=int(m.sum()), prev=float(y.mean()))
    if not per:
        return {}, per
    macro = {k: float(np.mean([v[k] for v in per.values()])) for k in ("auroc", "auprc", "f1", "f1_oracle")}
    macro["n_labels"] = len(per)
    return macro, per


def tune_thresholds(P, Y, A, cols):
    th, grid = {}, np.linspace(0.02, 0.98, 49)
    for j in cols:
        m = A[:, j] > 0
        if m.sum() == 0 or Y[m, j].sum() == 0:
            continue
        th[j] = float(grid[int(np.argmax([f1_score(Y[m, j], P[m, j] >= t) for t in grid]))])
    return th


def complete_matrices(Y, A, vocab, complete):
    """Matrix version of label_matrices(..., complete_hierarchy=True): one-step child->parent completion of Y,
    and implied positives marked known (A = max(A, Y)). Equal to label_matrices when nothing was removed."""
    Y, A = Y.copy(), A.copy()
    if not complete:
        return Y, A
    Y_in = Y.copy()
    for c, ps in IMPLIES.items():
        for p in ps:
            if c in vocab and p in vocab:
                Y[:, vocab.index(p)] = np.maximum(Y[:, vocab.index(p)], Y_in[:, vocab.index(c)])
    return Y, np.maximum(A, Y)


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    name = a.tag or f"{a.mode}-{a.target}-{a.head}-{a.labels}{'-h' if a.hierarchy else ''}-{a.dg}" \
                    f"{'-mix' if a.mixstyle else ''}-ld{a.lead_drop}{'-ho' if a.holdout else ''}-s{a.seed}"
    t0 = time.time()

    X, meta = load_prep(a.prep)
    nvalid = meta.n100.values.astype(np.int64)
    vocab, annotated, _ = build_vocab(meta, min_source=a.min_source)
    Y0, A0 = label_matrices(meta, vocab, annotated)  # original labels: all evaluation uses these
    Y, A = label_matrices(meta, vocab, annotated, complete_hierarchy=a.hierarchy)  # training targets
    if a.hier_soft:
        assert not a.hierarchy, "--hier_soft replaces --hierarchy"
        Y, _ = label_matrices(meta, vocab, annotated, complete_hierarchy=True)
        A = A0.copy()  # completed parents count only where the source annotates the parent
    dropped = []
    if a.drop_rate > 0:
        # E2: the removed (source, diagnosis) pairs are drawn from the RAW annotation (independent of method flags),
        # removed from the raw labels, and only then is the hierarchy completed (so no parent is completed from a
        # removed child annotation)
        Yr, Ar, dropped = drop_annotations(Y0, A0, meta, a.drop_rate, a.split_seed, vocab=vocab)
        Y, A = complete_matrices(Yr, Ar, vocab, a.hierarchy or a.hier_soft)
        if a.hier_soft:
            A = Ar
    tr, va, te = split_indices(meta, a.mode, a.target, seed=a.split_seed)
    if a.max_train:
        tr = np.random.default_rng(a.split_seed).permutation(tr)[:a.max_train]
    tr = np.sort(tr)
    C = len(vocab)
    hold = [vocab.index(c) for c in a.holdout if c in vocab]
    train_cols = [j for j in range(C) if j not in hold]  # held-out labels never enter the training forward pass
    A_train = np.ones_like(A) if a.labels == "naive" else A.copy()  # naive: unknown treated as negative
    # labels with no annotated record in the training sources get no pseudo-labels (no supervision to bootstrap)
    # (raw annotation before completion; in E2 removed (source, diagnosis) pairs count as never annotated)
    A_raw = A0.copy()
    for s_, j_ in dropped:
        A_raw[meta.source.values == s_, j_] = 0
    no_sup = np.where(A_raw[tr].sum(0) == 0)[0]

    text = None
    if a.head in ("text", "hybrid"):
        z = np.load(a.text_emb)
        emb = dict(zip(z["codes"].tolist(), z["emb"]))
        text = torch.tensor(np.stack([emb[c] for c in vocab]), dtype=torch.float32, device=dev)
    sources = sorted(meta.source.iloc[tr].unique())
    dom_id = meta.source.map({s: i for i, s in enumerate(sources)}).fillna(-1).astype(int).values
    model = ECGModel(C, a.head, text_dim=text.shape[1] if text is not None else 768, mixstyle=a.mixstyle,
                     n_domains=len(sources) if a.dg == "dann" else 0, residual=a.residual).to(dev)
    if a.residual != "none":
        assert a.head in ("text", "query"), "--residual needs --head text/query"
        seen_mask = torch.zeros(C, device=dev)
        seen_mask[[j for j in train_cols if j not in set(no_sup)]] = 1
        model.head.seen.copy_(seen_mask)
    hybrid = a.head == "hybrid"
    if hybrid:
        assert a.residual == "none", "hybrid uses the pure text head"
        model.seen.copy_(torch.tensor([1.0 if (j in train_cols and j not in set(no_sup)) else 0.0 for j in range(C)],
                                      device=dev))
        model.seen_score, model.text_detach = a.seen_score, a.text_detach
    use_pseudo = a.labels == "mask_pseudo"
    teacher = copy.deepcopy(model).eval() if (use_pseudo or a.eval_ema) else None
    # implication pairs (child, parent) as positions inside train_cols
    pos_in = {j: k for k, j in enumerate(train_cols)}
    hier_pairs = [(pos_in[vocab.index(c)], pos_in[vocab.index(p)]) for c, ps in IMPLIES.items() for p in ps
                  if c in vocab and p in vocab and vocab.index(c) in pos_in and vocab.index(p) in pos_in]
    hc = torch.tensor([c for c, _ in hier_pairs], device=dev, dtype=torch.long)
    hp = torch.tensor([p for _, p in hier_pairs], device=dev, dtype=torch.long)
    dg_w = a.dg_w if a.dg_w is not None else {"coral": 1.0, "dann": 0.1}.get(a.dg, 0.0)

    print(f"[{name}] device={dev} labels={C} train={len(tr)} val={len(va)} test={len(te)} sources={sources} "
          f"no_sup={[LABELS[vocab[j]][0] for j in no_sup]} dropped={len(dropped)}", flush=True)
    Xtr = torch.from_numpy(np.asarray(X[tr])).to(dev)  # float16 on device
    NVtr = torch.from_numpy(nvalid[tr]).to(dev)
    cols_t = torch.tensor(train_cols, device=dev)
    Ytr = torch.from_numpy(Y[tr][:, train_cols]).to(dev)
    Atr = torch.from_numpy(A_train[tr][:, train_cols]).to(dev)
    sup = torch.ones(len(train_cols), device=dev)
    sup[[train_cols.index(j) for j in no_sup if j in train_cols]] = 0
    Dtr = torch.from_numpy(dom_id[tr]).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    q_dro = torch.ones(len(sources), device=dev) / len(sources)
    best, best_state, step, history = -1, None, 0, []
    start_step = a.pseudo_start * (len(tr) // a.bs)

    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        dgs = dict(n=0.0, pos=0.0, neg=0.0, unk=0.0, pl=0.0, hv=0.0)  # --diag accumulators
        for i in range(0, len(tr) - a.bs + 1, a.bs):
            b = perm[i:i + a.bs]
            xc = random_crop(Xtr[b].float().clamp(-5, 5), NVtr[b])
            xb = augment(xc, a.lead_drop)
            yb, ab, db = Ytr[b], Atr[b], Dtr[b]
            txt_logits = None
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                if hybrid:  # text forward skipped when unused, so text_w=0 consumes no extra random numbers
                    logits, txt_logits, feat = model.forward_both(xb, text, cols_t,
                                                                  with_text=a.text_w > 0 or use_pseudo)
                else:
                    logits, feat = model(xb, text, cols_t)
            logits, feat = logits.float(), feat.float()
            L = elem_loss(a.loss, logits, yb)
            per_sample = (L * ab).sum(1) / ab.sum(1).clamp(min=1)
            if txt_logits is not None:
                txt_logits = txt_logits.float()
            if a.dg == "groupdro":
                gl = torch.stack([per_sample[db == d].mean() if (db == d).any() else per_sample.new_zeros(())
                                  for d in range(len(sources))])
                q_dro = q_dro * torch.exp(0.01 * gl.detach()); q_dro = q_dro / q_dro.sum()
                loss = (q_dro * gl).sum()
            else:
                loss = per_sample.mean()
            if hybrid and a.text_w > 0:  # same targets and mask as the linear head
                Lt = elem_loss(a.loss, txt_logits, yb)
                loss = loss + a.text_w * ((Lt * ab).sum(1) / ab.sum(1).clamp(min=1)).mean()
            if use_pseudo and step >= start_step:
                with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    pt = torch.sigmoid(teacher(xc, text, cols_t)[0].float())  # teacher sees the un-augmented crop
                if a.hier_soft and hier_pairs:  # hierarchy-coherent teacher targets: p(parent) >= p(child)
                    pt = pt.clone()
                    for c_, p_ in hier_pairs:
                        pt[:, p_] = torch.maximum(pt[:, p_], pt[:, c_])
                conf = (1 - ab) * sup * ((pt > a.pseudo_hi) | (pt < a.pseudo_lo)).float()
                if a.pseudo_start == 0 and a.pseudo_ramp == 0.3:
                    ramp = min(1.0, step / max(1, 0.3 * steps))  # original expression (bit-identical)
                else:
                    ramp = min(1.0, (step - start_step) / max(1, a.pseudo_ramp * steps))
                pl_logits = txt_logits if hybrid else logits  # hybrid: pseudo-labels train the text branch only
                if a.pseudo_soft:  # distil teacher probabilities; fixed point is p_student = p_teacher
                    pl = F.binary_cross_entropy_with_logits(pl_logits, pt, reduction="none")
                else:
                    pl = elem_loss(a.loss, pl_logits, (pt > 0.5).float())
                pseudo_term = (pl * conf).sum() / conf.sum().clamp(min=1)
                loss = loss + a.pseudo_w * ramp * pseudo_term
                if a.diag:
                    unk = (1 - ab) * sup
                    dgs["unk"] += unk.sum().item()
                    dgs["pos"] += (unk * (pt > a.pseudo_hi).float()).sum().item()
                    dgs["neg"] += (unk * (pt < a.pseudo_lo).float()).sum().item()
                    dgs["pl"] += pseudo_term.item()
                    dgs["n"] += 1
            if a.hier_soft and hier_pairs:  # implication as a label-free consistency constraint on every record
                pr = torch.sigmoid(logits)
                hviol = F.relu(pr[:, hc] - pr[:, hp]).mean()
                loss = loss + a.hier_w * hviol
                if a.diag:
                    dgs["hv"] += hviol.item()
            if a.dg == "coral":
                loss = loss + dg_w * coral(feat, db)
            elif a.dg == "dann":
                loss = loss + dg_w * F.cross_entropy(model.dom(GradReverse.apply(feat, 1.0)), db)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if teacher is not None:
                decay = min(a.ema, (1 + step) / (10 + step)) if a.ema_warmup else a.ema
                with torch.no_grad():
                    for pt_, ps_ in zip(teacher.state_dict().values(), model.state_dict().values()):
                        if pt_.dtype.is_floating_point:
                            pt_.mul_(decay).add_(ps_.detach(), alpha=1 - decay)
                        else:
                            pt_.copy_(ps_)
            tot += loss.item()
        sel = teacher if a.eval_ema else model  # weights that are validated, selected and tested
        Pv = predict(sel, X, nvalid, va, text, dev)
        mv, _ = evaluate(Pv, Y0[va], A0[va], train_cols)
        history.append(dict(epoch=ep, loss=tot / max(1, len(tr) // a.bs), val_auroc=mv.get("auroc")))
        if hold:  # zero-shot proxy on VALIDATION records: diagnoses withheld from training (no test data)
            history[-1]["val_holdout_auroc"] = evaluate(Pv, Y0[va], A0[va], hold)[0].get("auroc")
        if a.diag:
            nb = max(1, len(tr) // a.bs)
            history[-1].update(pseudo_pos_frac=dgs["pos"] / max(1.0, dgs["unk"]),
                               pseudo_neg_frac=dgs["neg"] / max(1.0, dgs["unk"]),
                               pseudo_loss=dgs["pl"] / max(1.0, dgs["n"]), hier_viol=dgs["hv"] / nb)
            other = model if a.eval_ema else teacher
            if other is not None:
                key = "student_val_auroc" if a.eval_ema else "teacher_val_auroc"
                history[-1][key] = evaluate(predict(other, X, nvalid, va, text, dev),
                                            Y0[va], A0[va], train_cols)[0].get("auroc")
        extra = {k: v for k, v in history[-1].items() if k not in ("epoch", "loss", "val_auroc")}
        print(f"ep {ep:02d} loss {history[-1]['loss']:.4f} val_auroc {mv.get('auroc', float('nan')):.4f} "
              f"t={time.time() - t0:.0f}s" + (f" {extra}" if extra else ""), flush=True)
        if mv.get("auroc", -1) > best:
            best, best_state = mv["auroc"], copy.deepcopy(sel.state_dict())

    if best_state is None and a.eval_ema:
        best_state = teacher.state_dict()
    if best_state is not None:  # None only if validation had no evaluable label (tiny debug runs)
        model.load_state_dict(best_state)
    Pv = predict(model, X, nvalid, va, text, dev)
    th = tune_thresholds(Pv, Y0[va], A0[va], train_cols)
    Pt = predict(model, X, nvalid, te, text, dev)
    # "seen" = annotated by at least one training source (method-independent); held-out labels are unseen
    seen = [j for j in train_cols if j not in set(no_sup)]
    unseen = [j for j in range(C) if j not in seen]
    m_all, per = evaluate(Pt, Y0[te], A0[te], list(range(C)), th)
    m_seen, _ = evaluate(Pt, Y0[te], A0[te], seen, th)
    m_unseen, _ = evaluate(Pt, Y0[te], A0[te], unseen, th)
    res = dict(name=name, config=vars(a), vocab=vocab, best_val_auroc=best, history=history,
               test_all=m_all, test_seen=m_seen, test_unseen=m_unseen,
               seen=[vocab[j] for j in seen], unseen=[vocab[j] for j in unseen],
               dropped=[(s, vocab[j]) for s, j in dropped],
               per_label={LABELS[vocab[j]][0]: v for j, v in per.items()},
               minutes=(time.time() - t0) / 60)
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(a.out, name + "_test.npz"), P=Pt.astype(np.float16), Y=Y0[te].astype(np.uint8),
                        A=A0[te].astype(np.uint8), idx=te)
    print(f"[{name}] TEST all={m_all} seen={m_seen} unseen={m_unseen} ({res['minutes']:.1f} min)", flush=True)
    return res


if __name__ == "__main__":
    main()
